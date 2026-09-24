"""Same scientific experiment on three H100 workers with verified local inputs."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import time
import modal

ROOT = Path(__file__).resolve().parents[2] if modal.is_local() else Path('/root')
P = ROOT / 'research/pilot'
H = ROOT / 'research/h100'
app = modal.App('accent-selection-h100-extension')
data = modal.Volume.from_name('accent-selection-pilot-data')
outputs = modal.Volume.from_name('accent-selection-pilot-results')
models = modal.Volume.from_name('ast-asr-cache')
RATE = .001097 + 8*.0000131 + 32*.00000222
image = (modal.Image.debian_slim(python_version='3.12').apt_install('ffmpeg')
    .pip_install('torch==2.13.0', 'transformers==5.16.1', 'accelerate==1.11.0',
                 'datasets==4.4.1', 'torchcodec==0.16.0', 'soundfile==0.14.0',
                 'librosa==1.0.0', 'numpy==2.5.2')
    .pip_install('jiwer==4.0.0')
    .env({'HF_HOME':'/cache/huggingface','HF_HUB_ENABLE_HF_TRANSFER':'0'})
    .pip_install('pytest==9.1.1')
    .add_local_file(P/'training.py','/root/research/pilot/training.py')
    .add_local_file(P/'test_training.py','/root/research/pilot/test_training.py')
    .add_local_file(P/'frozen_english_normalizer.py','/root/research/pilot/frozen_english_normalizer.py')
    .add_local_file(P/'feature_cache.py','/root/research/pilot/feature_cache.py')
    .add_local_file(ROOT/'paper/data/whisper_normalizer.json','/root/normalizer.json'))

FILES = ('summary.json','provenance.json','manifest.json','history.jsonl',
         'baseline_dev_predictions.jsonl','final_dev_predictions.jsonl',
         'baseline_dev_metrics.json','final_dev_metrics.json')


@app.function(image=image,gpu='H100',cpu=(8,8),memory=(32768,32768),
              timeout=1200,startup_timeout=180,retries=0,max_containers=3,
              scaledown_window=2,
              volumes={'/data':data,'/outputs':outputs,'/cache/huggingface':models})
def train_arm(plan):
    import importlib.util
    import sys
    import threading
    from concurrent.futures import ThreadPoolExecutor
    start=time.monotonic()
    spec=importlib.util.spec_from_file_location('h100_frozen_training','/root/research/pilot/training.py')
    core=importlib.util.module_from_spec(spec);sys.modules[spec.name]=core;spec.loader.exec_module(core)
    if core.sha256_file(Path(core.__file__)) != plan['training_code_sha256']:
        raise ValueError('Frozen training source changed')
    data.reload()
    rows=plan['rows'];train,dev=core.validate_rows(rows)
    needed={row['utt_id']:row for row in dev}
    for cfg in plan['configs']:
        indices=core.IndexCycler(len(train),cfg['seed']).take(cfg['num_updates']*cfg['batch_size'])
        needed.update({train[i]['utt_id']:train[i] for i in indices})
    local_audio=Path('/tmp/h100-audio');local_features=Path('/tmp/h100-features')
    local_features.mkdir(parents=True,exist_ok=True)
    def stage(row):
        if row['split'] not in ('train','dev'):raise ValueError('Test access forbidden')
        receipt=plan['cache_receipts'][row['utt_id']]
        sources=[(Path('/data/l2arctic')/row['audio_path'],local_audio/row['audio_path'],receipt['audio_sha256']),
                 (Path('/data/feature_cache/frozen-v1')/receipt['feature_file'],local_features/receipt['feature_file'],receipt['feature_sha256'])]
        for source,target,expected in sources:
            payload=source.read_bytes()
            if hashlib.sha256(payload).hexdigest()!=expected:raise ValueError(f'Input checksum mismatch: {row["utt_id"]}')
            target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(payload)
        return row['utt_id']
    print(json.dumps({'stage':'input_staging','arm':plan['arm'],'rows':len(needed),'test_access':False}),flush=True)
    with ThreadPoolExecutor(max_workers=32) as pool: staged=list(pool.map(stage,needed.values()))
    staging={'status':'verified_local_copies','rows':len(staged),'utt_ids_sha256':core.canonical_hash(sorted(staged)),
             'seconds':time.monotonic()-start,'model_calls':0,'test_access':False}
    print(json.dumps({'arm':plan['arm'],'staging':staging}),flush=True)
    result=[]
    for config in plan['configs']:
        if time.time()>=plan['compute_cutoff_epoch']:raise TimeoutError('Compute cutoff before next seed')
        name=f"{plan['arm']}-seed{config['seed']}"
        folder=Path('/outputs/h100-v1')/name
        cfg={**config,'feature_cache_dir':str(local_features),'normalizer_path':'/root/normalizer.json'}
        stop=threading.Event()
        def progress():
            while not stop.wait(20):
                log=folder/'history.jsonl';update=0
                if log.exists():
                    for line in log.read_text().splitlines():
                        try:e=json.loads(line)
                        except ValueError:continue
                        if e.get('event')=='train_update':update=e['update']
                print(f'H100 {name}: {update}/600 updates; worker elapsed {time.monotonic()-start:.0f}s',flush=True)
        thread=threading.Thread(target=progress,daemon=True);thread.start()
        try:summary=core.train(cfg,rows,local_audio,folder)
        finally:stop.set();thread.join(timeout=1)
        if 'H100' not in summary['runtime']['device_name']:raise ValueError('Expected H100 hardware')
        summary['checkpoint_model_sha256']=core.sha256_file(folder/'checkpoint_last/model.safetensors')
        summary['input_staging']=staging
        (folder/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
        outputs.commit()
        result.append({'name':name,'files':{f:(folder/f).read_text() for f in FILES}})
        print(json.dumps({'completed':name,'macro_wer':summary['final_dev_metrics']['english']['macro_language_wer'],
                          'updates':summary['completed_updates'],'checkpoint_sha256':summary['checkpoint_model_sha256']}),flush=True)
    seconds=time.monotonic()-start
    return json.loads(json.dumps({'arm':plan['arm'],'runs':result,'wall_seconds':seconds,'estimated_resource_cost_usd':seconds*RATE}))


def dump(path,value):
    Path(path).write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n')


def prepare():
    from datetime import datetime,timezone
    from research.pilot import loop,workflow
    from research.pilot.training import canonical_hash
    schedule=json.loads((H/'schedule.json').read_text())
    old=loop.status(P/'ledger')
    for relative,expected in schedule['frozen_inputs'].items():
        if loop.file_sha(ROOT/relative)!=expected:raise ValueError('Frozen scientific input changed')
    rows=[json.loads(l) for l in (P/'data/manifest.jsonl').read_text().splitlines()]
    dev=[r for r in rows if r['split']=='dev']
    receipts={r['utt_id']:r for path in (P/'feature_receipts').glob('*.json') for r in json.loads(path.read_text())['files']}
    plans=[];jobs={}
    for arm in ('full_gold','filtered10_gold','random_matched_gold'):
        train=[json.loads(l) for l in (P/'selections'/f'{arm}.jsonl').read_text().splitlines()]
        configs=[workflow.training_config(arm,seed) for seed in (0,1,2)]
        plan={'arm':arm,'configs':configs,'rows':train+dev,
              'training_code_sha256':loop.file_sha(P/'training.py'),
              'cache_receipts':{r['utt_id']:receipts[r['utt_id']] for r in train+dev},
              'compute_cutoff_epoch':json.loads((H/'deadline.json').read_text())['experiment_cutoff_epoch']}
        plans.append(plan)
        jobs[arm]={'kind':'infrastructure','dependencies':[],'parameters':{'configs':configs,'plan_sha256':canonical_hash(plan)},
                   'required_evidence':['audit','receipt'],'success_checks':[{'source':'receipt','pointer':'/audited_run_count','op':'eq','value':3}],
                   'dev_queries':{f'seed{s}_{label}':{'dataset':'development','evaluator':'training','purpose':'Fixed H100 baseline/final, previously exposed development speakers'} for s in (0,1,2) for label in ('baseline','final')}}
    jobs['assessment']={'kind':'scientific','dependencies':list(jobs),'parameters':{},'required_evidence':['audit','decision'],
        'success_checks':[{'source':'audit','pointer':'/status','op':'eq','value':'valid_complete'},
                          {'source':'decision','pointer':'/pilot_positive','op':'eq','value':True}],'dev_queries':{}}
    artifacts=json.loads((P/'ledger/contract.json').read_text())['artifacts']
    artifacts={k:{'path':v['path'],'role':v['role']} for k,v in artifacts.items()}
    artifacts['schedule']={'path':str(H/'schedule.json'),'role':'input'}
    artifacts['h100_runner']={'path':str(H/'run.py'),'role':'code'}
    artifacts['h100_deadline']={'path':str(H/'deadline.json'),'role':'input'}
    contract={'schema_version':1,'experiment_id':schedule['experiment_id'],'max_jobs':4,'max_running':3,
              'max_dev_queries':18,'final_test_policy':'sealed','artifacts':artifacts,'jobs':jobs,
              'gates':{'replication_ready':{'required_jobs':['assessment'],'checks':[]}},
              'registration_note':f'New H100 execution; prior dev accesses actual={old["dev_queries_actual_reported"]}, conservatively charged={old["dev_queries_consumed"]}. All eighteen new accesses disclosed; not independent confirmation.'}
    dump(H/'contract.json',contract);loop.initialize(H/'ledger',H/'contract.json')
    for name in jobs:loop.apply(H/'ledger','register',{'job_id':name,'origin':'prospective'})
    dump(H/'plans.json',plans)
    dump(H/'registration.json',{'registered_at':datetime.now(timezone.utc).isoformat(),'prior_evaluation_accounting':old,
                              'new_query_reservations':18,'scientific_settings_changed':False,'hardware_changed':'All9 rerun onH100',
                              'staging':'Exact selected exposure union, checksummed audio/features copied tolocalSSD; no new model queries'})
    return plans


def ingest(result):
    from research.pilot import loop
    from research.pilot.evaluate_runs import evaluate
    arm=result['arm'];directories=[]
    for row in result['runs']:
        directory=H/'runs'/row['name'];directory.mkdir(parents=True,exist_ok=False)
        for name,payload in row['files'].items():(directory/name).write_text(payload)
        directories.append(directory)
    audit=evaluate(directories,P/'data/manifest.jsonl',P/'selections',json.loads((H/'schedule.json').read_text()))
    if len(audit['runs'])!=3 or {r['arm_name'] for r in audit['runs']}!={arm}:raise ValueError('Arm receipt incomplete')
    audit_path=H/f'{arm}.audit.json';dump(audit_path,audit)
    receipt_path=H/f'{arm}.receipt.json';dump(receipt_path,{'arm':arm,'audited_run_count':3,'wall_seconds':result['wall_seconds'],'estimated_resource_cost_usd':result['estimated_resource_cost_usd']})
    for directory in directories:
        summary=json.loads((directory/'summary.json').read_text());seed=summary['config']['seed']
        for label,fingerprint in [('baseline',summary['initial_state_sha256']),('final',summary['checkpoint_model_sha256'])]:
            loop.apply(H/'ledger','query',{'job_id':arm,'query_id':f'seed{seed}_{label}','accounting':'receipt_after_run',
                'checkpoint_sha256':fingerprint,'reason':'Fixed H100 evaluation receipt independently recomputed','receipt':{'path':str(audit_path)}})
    ids=[f'seed{s}_{label}' for s in (0,1,2) for label in ('baseline','final')]
    loop.apply(H/'ledger','record',{'job_id':arm,'outcome':'passed','evidence':{'audit':{'path':str(audit_path)},'receipt':{'path':str(receipt_path)}},'dev_query_ids':ids,'actual_dev_query_ids':ids})
    budget=json.loads((P/'budget.json').read_text())
    job=next(j for j in budget['new_jobs'] if j['job_id']==f'h100-v1/{arm}')
    job.update(status='completed',wall_seconds=result['wall_seconds'],estimated_resource_cost_usd=result['estimated_resource_cost_usd'])
    dump(P/'budget.json',budget)


def finalize():
    import statistics
    from research.pilot import loop
    from research.pilot.evaluate_runs import evaluate,evidence_markdown
    schedule=json.loads((H/'schedule.json').read_text())
    dirs=[H/'runs'/f"{r['arm_name']}-seed{r['seed']}" for r in schedule['runs']]
    audit=evaluate(dirs,P/'data/manifest.jsonl',P/'selections',schedule)
    if audit['status']!='valid_complete':raise ValueError('Required H100 runs incomplete')
    e=[r['filtered_minus_random_macro_wer_pp'] for r in audit['paired_contrasts'] if r['normalization']=='english']
    m=[r['filtered_minus_random_macro_wer_pp'] for r in audit['paired_contrasts'] if r['normalization']=='minimal']
    decision={'valid_complete':True,'mean_english_contrast_pp':statistics.mean(e),'mean_minimal_contrast_pp':statistics.mean(m),
              'positive_seed_count':sum(v>0 for v in e),'scope':'Complete same-H100 exploratorypilot; same previously exposeddevelopment speakers; not independent confirmation.'}
    decision['pilot_positive']=statistics.mean(e)>=.5 and sum(v>0 for v in e)>=2 and statistics.mean(m)>=0
    out=H/'results';out.mkdir(exist_ok=True);dump(out/'audit.json',audit);dump(out/'decision.json',decision)
    (out/'evidence.md').write_text(evidence_markdown(audit))
    loop.apply(H/'ledger','start',{'job_id':'assessment'})
    loop.apply(H/'ledger','record',{'job_id':'assessment','outcome':'passed' if decision['pilot_positive'] else 'inconclusive',
        'evidence':{'audit':{'path':str(out/'audit.json')},'decision':{'path':str(out/'decision.json')}},'dev_query_ids':[],'actual_dev_query_ids':[],
        'reason':'Frozen numerical pilot criterion, all9 H100 runs retained'})
    print(json.dumps(decision,indent=2),flush=True)


@app.local_entrypoint()
def main():
    from research.pilot import loop
    plans=prepare()
    budget=json.loads((P/'budget.json').read_text());reserved=budget['prior_work_carryover_allowance_usd']+sum(j['reserved_usd'] for j in budget['new_jobs'])
    if reserved+5.4>budget['authorized_total_usd']:raise ValueError('H100 bounds exceed original spending authorization')
    for plan in plans:
        budget['new_jobs'].append({'job_id':f'h100-v1/{plan["arm"]}','reserved_usd':1.8,'status':'reserved','gpu':'H100','hard_timeout_seconds':1200,'startup_timeout_seconds':180})
    dump(P/'budget.json',budget)
    for plan in plans:
        arm=plan['arm'];loop.apply(H/'ledger','start',{'job_id':arm,'external_job_id':f'h100-v1/{arm}'})
        for seed in (0,1,2):
            for label in ('baseline','final'):loop.apply(H/'ledger','reserve_query',{'job_id':arm,'query_id':f'seed{seed}_{label}','reason':'Reserve all6 fixed evaluations before arm worker dispatch'})
    failures=[]
    for result in train_arm.map(plans,order_outputs=False,return_exceptions=True):
        if isinstance(result,Exception):failures.append(str(result));continue
        try:ingest(result)
        except Exception as error:failures.append(str(error))
    if failures:
        dump(H/'failures.json',failures);raise RuntimeError(str(failures))
    finalize()


if __name__=='__main__':
    raise SystemExit('Run through the registered Modal app local entrypoint.')
