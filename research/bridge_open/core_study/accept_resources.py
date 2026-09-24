"""Independent acceptance and descriptive summary for fixed-panel resource runs."""
from __future__ import annotations
import argparse, gzip, hashlib, json, math, statistics, tarfile
from functools import lru_cache
from pathlib import Path
from typing import Any
from research.bridge_open.core_study.resource_profile import FIXED_PANEL_IDS, FIXED_PANEL_MANIFEST_SHA256, read_fixed_panel
from research.bridge_open.scoring import normalize_v1, score

ROOT=Path(__file__).resolve().parents[3]
SUITE_KEYS=("whisper_small","whisper_turbo","whisper_large_v3","qwen_0_6b","qwen_1_7b")
OMNI_COMMIT="259f334195795db97812a8220cf793a988ece0de"
SUITE_COMMIT_PREFIX="25c8ef2"
FIXED_HARNESS_SHA256="aeb16dffecd15b393044751ecc7309189325ce25e81ee750625d0c7c2ab5486f"
EXPECTED_MODELS={
    'omni':('facebook/omniASR-CTC-1B','8c22e3ffdaa4aab6431b128b84b991a7d9c2515c','bfloat16'),
    'whisper_small':('openai/whisper-small','973afd24965f72e36ca33b3055d56a652f456b4d','float32'),
    'whisper_turbo':('openai/whisper-large-v3-turbo','41f01f3fe87f28c78e2fbf8b568835947dd65ed9','float32'),
    'whisper_large_v3':('openai/whisper-large-v3','06f233fe06e710322aca913c1bc4249a0d71fce1','float32'),
    'qwen_0_6b':('Qwen/Qwen3-ASR-0.6B','5eb144179a02acc5e5ba31e748d22b0cf3e303b0','bfloat16'),
    'qwen_1_7b':('Qwen/Qwen3-ASR-1.7B','7278e1e70fe206f11671096ffdd38061171dd6e5','bfloat16'),
}

def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def require(ok, message):
    if not ok: raise ValueError(message)
def load(path): return json.loads(Path(path).read_text())

@lru_cache(maxsize=None)
def _archive(run):
    run=Path(run)
    receipt=load(run/'orx_source_receipt.json'); require(receipt.get('provider_status')=='done','provider is not done')
    name=receipt.get('portable_archive'); require(isinstance(name,str) and Path(name).name==name,'portable archive must be a basename')
    archive=run/name; require(archive.is_file() and sha(archive)==receipt.get('portable_archive_sha256'),'portable archive hash mismatch')
    raw=gzip.decompress(archive.read_bytes()); require(hashlib.sha256(raw).hexdigest()==receipt.get('sourceDigest'),'source archive digest mismatch')
    payload={}
    with tarfile.open(archive,'r:gz') as tar:
        for item in tar:
            name=item.name.removeprefix('./')
            require(name and not Path(name).is_absolute() and '..' not in Path(name).parts,'archive contains traversal path')
            if item.isfile():
                require(name not in payload,f'archive contains duplicate member {name}')
                stream=tar.extractfile(item); require(stream is not None,f'cannot extract {name}')
                payload[name]=stream.read()
    require(payload,'portable archive contains no source files')
    return receipt,payload

def _source_hashes(payload, hashes):
    require(isinstance(hashes,dict) and hashes,'missing source hashes')
    for name,digest in hashes.items():
        require(name in payload,f'archive misses hashed source {name}')
        require(hashlib.sha256(payload[name]).hexdigest()==digest,f'archived source hash mismatch: {name}')

def _archived_config(payload, basename):
    matches=[(name,data) for name,data in payload.items() if Path(name).name==basename]
    require(len(matches)==1,f'archive lacks unique {basename}')
    try: return json.loads(matches[0][1])
    except json.JSONDecodeError as exc: raise ValueError(f'archived {basename} is invalid JSON') from exc

def _gpu(environment):
    name=str(environment.get('gpu_name',environment.get('gpu',''))); props=environment.get('device_properties',{})
    memory=props.get('total_memory_bytes')
    require('a100' in name.lower(),'measured GPU is not A100')
    require(isinstance(memory,int) and 70*2**30 <= memory <= 90*2**30,'measured GPU memory is not approximately 80 GiB')
    version=str(environment.get('torch_version',environment.get('torch','')))
    require(version.startswith('2.8.0'),'unexpected PyTorch runtime version')
    require(environment.get('cuda_version',environment.get('cuda')),'missing CUDA version')

def _check_calls(profile, sources, gpu_memory):
    study=profile.get('study',{}); require(study.get('panel_manifest_sha256')==FIXED_PANEL_MANIFEST_SHA256,'profile manifest mismatch')
    require(study.get('conversation_ids')==list(FIXED_PANEL_IDS),'profile panel IDs mismatch')
    warmup,timed=profile.get('warmup_calls'),profile.get('timed_calls')
    require(isinstance(warmup,list) and len(warmup)==12,'warmup must retain all 12 calls')
    require(isinstance(timed,list) and len(timed)==36,'timed section must retain all 36 calls')
    require([r.get('conversation_id') for r in warmup]==list(FIXED_PANEL_IDS),'warmup order mismatch')
    for number in range(1,4):
        rows=[r for r in timed if r.get('pass_index')==number]
        require([r.get('conversation_id') for r in rows]==list(FIXED_PANEL_IDS),f'timed pass {number} order mismatch')
    repeats={cid:[] for cid in FIXED_PANEL_IDS}; statuses={cid:[] for cid in FIXED_PANEL_IDS}; failures=0; warmup_without_hash=0
    for records,is_timed in ((warmup,False),(timed,True)):
      for record in records:
        status=record.get('status'); text=record.get('text'); digest=record.get('text_sha256')
        require(status in {'success','failed'},'call status must be success or failed')
        source=sources[record['conversation_id']]
        if is_timed: require(record.get('audio_sha256')==source['audio_sha256'],'timed call audio hash mismatch or absent')
        elif record.get('audio_sha256') is None: warmup_without_hash+=1
        else: require(record['audio_sha256']==source['audio_sha256'],'warmup call audio hash mismatch')
        duration=float(record.get('duration_seconds',0)); require(math.isclose(duration,float(source['duration_seconds']),rel_tol=0,abs_tol=1/16000),'call duration differs from fixed source')
        if status=='failed': require(isinstance(record.get('failure'),dict),'failed call lacks failure evidence'); failures+=1
        else:
            require(isinstance(text,str),'successful call lacks text')
            require(digest==hashlib.sha256(text.encode()).hexdigest(),'text hash mismatch')
        if is_timed:
            seconds=float(record.get('end_to_end_inference_seconds',0)); rtf=float(record.get('real_time_factor',-1))
            require(math.isfinite(duration) and math.isfinite(seconds) and duration>0 and seconds>0,'nonpositive/invalid timing')
            require(math.isclose(rtf,seconds/duration,rel_tol=1e-9,abs_tol=1e-12),'RTF arithmetic mismatch')
            allocated,reserved=record.get('resident_baseline_memory_bytes',{}).get('allocated_bytes'),record.get('resident_baseline_memory_bytes',{}).get('reserved_bytes')
            peak_allocated,peak_reserved=record.get('peak_cuda_allocated_bytes'),record.get('peak_cuda_reserved_bytes')
            require(all(isinstance(value,int) for value in (allocated,reserved,peak_allocated,peak_reserved)),'missing CUDA memory evidence')
            require(0<=allocated<=peak_allocated<=peak_reserved<=gpu_memory and 0<=reserved<=peak_reserved,'invalid CUDA baseline/peak ordering')
            statuses[record['conversation_id']].append(status)
            if status=='success': repeats[record['conversation_id']].append(text)
    differences=[cid for cid,values in repeats.items() if len(set(values))>1]
    mixed=[cid for cid,values in statuses.items() if 0<values.count('failed')<3]
    declared=profile.get('summary',{}).get('failed_calls_including_warmup'); require(declared==failures,'failure count drops calls')
    warmup_binding=('verified shared preloaded inputs, no per-call warmup hash' if warmup_without_hash else 'per-call warmup audio hashes verified')
    return timed,failures,{'repeat_different_ids':differences,'mixed_failure_ids':mixed,'warmup_source_binding':warmup_binding}

def _quality(run, prefix, profile, rows):
    path=run/(f'{prefix}_quality_predictions.jsonl' if prefix else 'quality_predictions.jsonl')
    records=[json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    require([r.get('conversation_id') for r in records]==list(FIXED_PANEL_IDS),'quality IDs/order mismatch')
    final=[r for r in profile['timed_calls'] if r['pass_index']==3]
    errors=words=0
    for source, timing, record in zip(rows,final,records,strict=True):
        require(record.get('timed_pass_index')==3,'quality uses a pass other than 3')
        require(record.get('audio_sha256')==source['audio_sha256'] and record.get('reference_sha256')==source['reference_sha256'],'quality source hash mismatch')
        require(record.get('text_raw')==timing.get('text') and record.get('text_sha256')==timing.get('text_sha256'),'quality text differs from pass3 timing')
        ref_path=ROOT/'data/bridge_open/svarah_native_shard02_v2/bundle/references'/Path(source['reference_path']).name
        require(sha(ref_path)==source['reference_sha256'],'local reference hash mismatch')
        if timing.get('status')=='success':
            require(record.get('metrics') is not None,'successful pass3 call lacks quality metrics')
        if record.get('metrics') is not None:
            ref=load(ref_path)['reference_text']
            expected=score(ref,record['text_raw']); require(record['metrics']==expected,'quality metric recomputation mismatch')
            errors+=sum(expected[k] for k in ('substitutions','deletions','insertions')); words+=expected['ref_words']
        elif timing.get('status')=='failed':
            require(record.get('status')=='inference_error' and record.get('failure')==timing.get('failure'),'failed timing call was not propagated to quality')
    return {'quality_records':len(records),'pass3_micro_wer':errors/words if words else None,'pass3_word_errors':errors,'pass3_reference_words':words}

def _summary(profile, timed):
    passes=[]
    for number in range(1,4):
        rows=[r for r in timed if r['pass_index']==number]; wall=sum(r['end_to_end_inference_seconds'] for r in rows); audio=sum(r['duration_seconds'] for r in rows)
        passes.append({'pass_index':number,'corpus_rtf':wall/audio,'wall_seconds':wall,'median_clip_rtf':statistics.median(r['real_time_factor'] for r in rows),'peak_cuda_allocated_bytes':max(r['peak_cuda_allocated_bytes'] for r in rows),'peak_cuda_reserved_bytes':max(r['peak_cuda_reserved_bytes'] for r in rows)})
    return {'passes':passes,'median_clip_rtf':statistics.median(r['real_time_factor'] for r in timed),'median_pass_wall_seconds':statistics.median(r['wall_seconds'] for r in passes),'peak_cuda_allocated_bytes':max(r['peak_cuda_allocated_bytes'] for r in timed),'peak_cuda_reserved_bytes':max(r['peak_cuda_reserved_bytes'] for r in timed),'repeated_timing_note':'Three passes are repeated timing measurements, not 36 independent quality/reference observations.'}

def _one(run, prefix, sources, expected=None, commit=None):
    run=Path(run); require((run/'remote_validation.json').is_file() and load(run/'remote_validation.json').get('valid') is True,'remote validation is not valid')
    require((run/'container_receipt.json').is_file() and ((run/'artifact_transport.json').is_file() or (run/'transport_receipt.json').is_file()),'missing top-level container/transport receipt')
    receipt,payload=_archive(str(run)); require((receipt.get('commit')==commit) if commit and len(commit)==40 else (str(receipt.get('commit','')).startswith(commit) if commit else True),'unexpected archived source commit')
    profile=load(run/(f'{prefix}_resource_profile.json' if prefix else 'resource_profile.json'))
    runtime=load(run/(f'{prefix}_environment.json' if prefix else 'environment.json'))
    _source_hashes(payload,runtime.get('source_sha256'))
    require(runtime['source_sha256'].get('research/bridge_open/core_study/resource_profile.py')==FIXED_HARNESS_SHA256,'unexpected fixed resource harness hash')
    env=profile.get('environment',{}); _gpu(env); timed,failures,variation=_check_calls(profile,sources,env['device_properties']['total_memory_bytes'])
    model=profile.get('model_residency',{}); require(model.get('resident_before_measurement') is True and isinstance(model.get('parameter_count'),int) and model['parameter_count']>0 and isinstance(model.get('model_hashes'),dict) and model['model_hashes'],'invalid model residency/parameter/weight evidence')
    require(profile.get('pipeline_configuration',{}).get('batch_size')==1,'pipeline batch size is not one')
    if prefix:
        require(runtime.get('parameter_count')==model['parameter_count'] and runtime.get('model_file_sha256')==model['model_hashes'],'per-model runtime evidence disagrees with profile')
        require(runtime.get('model_eval') is True and runtime.get('actual_model_dtype')==f"torch.{model['declared_precision']}",'actual suite eval/dtype evidence disagrees with profile')
        packages=runtime.get('packages',{}); require(str(packages.get('torch','')).startswith('2.8.0') and packages.get('transformers')=='4.57.6','suite package evidence mismatch')
        if prefix.startswith('qwen_'): require(packages.get('qwen-asr')=='0.0.6','Qwen package evidence mismatch')
    else:
        evidence=runtime.get('runtime_evidence',{})
        require(evidence.get('model_training') is False and evidence.get('parameter_count')==model['parameter_count'] and evidence.get('parameter_dtypes')==['torch.bfloat16'] and evidence.get('pipeline_source_sha256'),'Omni runtime pipeline/eval/dtype/parameter evidence disagrees with profile')
        require(runtime.get('asset_sha256')==model['model_hashes'],'Omni runtime weight evidence disagrees with profile')
    if expected:
        for key,value in expected.items(): require(model.get(key)==value,f'model evidence mismatch: {key}')
    # Parameter count must be derived by the exact archived Python runner, not only reported in JSON.
    code_names=list(runtime['source_sha256'])
    code_text='\n'.join(payload[name].decode('utf-8','ignore') for name in code_names if name.endswith('.py'))
    require('numel()' in code_text,'archive lacks parameter-count derivation code')
    return profile, {'provider_receipt':receipt,'failure_calls':failures,**variation,**_summary(profile,timed)},payload

def _batch8_comparison(prefix, quality):
    candidates={'qwen_1_7b':ROOT/'research/bridge_open/artifacts/qwen_svarah_full_v1','qwen_0_6b':ROOT/'research/bridge_open/artifacts/core_qwen06_svarah_v1'}
    other=candidates.get(prefix)
    if not other or not (other/'predictions.jsonl').is_file(): return {'status':'unavailable'}
    baseline={r['conversation_id']:r.get('text_raw') for r in (json.loads(x) for x in (other/'predictions.jsonl').read_text().splitlines())}
    current={r['conversation_id']:r.get('text_raw') for r in quality}
    shared=sorted(set(baseline)&set(current)); return {'status':'descriptive_only','shared_ids':shared,'matching_ids':[cid for cid in shared if baseline[cid]==current[cid]],'different_ids':[cid for cid in shared if baseline[cid]!=current[cid]],'note':'Batch-1 resource texts are compared descriptively with existing batch-8 output; equality is not an acceptance requirement.'}

def validate(omni_dir, suite_dir):
    rows=read_fixed_panel(); sources={row['conversation_id']:row for row in rows}
    roster={row['model_id']:row['revision'] for row in load(ROOT/'research/bridge_open/model_roster.json')['models']}
    for model_id,revision,_dtype in EXPECTED_MODELS.values(): require(roster.get(model_id)==revision,f'local roster revision mismatch for {model_id}')
    omni_profile,omni,omni_payload=_one(omni_dir,'',sources,dict(zip(('model_id','model_revision','declared_precision'),EXPECTED_MODELS['omni'])),OMNI_COMMIT)
    omni_config=_archived_config(omni_payload,'omni_resource_config.json')
    require(tuple(omni_config.get(k) for k in ('model_id','model_revision','dtype'))==EXPECTED_MODELS['omni'] and omni_config.get('batch_size')==1 and omni_config.get('public_api_calls_per_input_per_phase')==1,'archived Omni config differs from pinned contract')
    omni['quality']=_quality(Path(omni_dir),'',omni_profile,rows)
    suite={'models':{}}
    suite_payload=None
    for key in SUITE_KEYS:
        profile,summary,payload=_one(suite_dir,key,sources,dict(zip(('model_id','model_revision','declared_precision'),EXPECTED_MODELS[key])),SUITE_COMMIT_PREFIX)
        suite_payload=suite_payload or payload
        search=profile['model_residency']['declared_search']
        if key.startswith('whisper_'):
            require(search.get('num_beams')==5 and search.get('max_new_tokens')==440 and search.get('return_timestamps') is False and search.get('do_sample') is False,'Whisper direct-generate beam5/max440/no-timestamps/greedy contract changed')
        else:
            require(search.get('num_beams')==1 and search.get('max_new_tokens')==512 and search.get('language')=='English' and search.get('do_sample') is False,'Qwen greedy/English/max512 contract changed')
        quality=_quality(Path(suite_dir),key,profile,rows); summary['quality']=quality; summary['batch8_comparison']=_batch8_comparison(key,[json.loads(x) for x in (Path(suite_dir)/f'{key}_quality_predictions.jsonl').read_text().splitlines()]); suite['models'][key]=summary
    suite_config=_archived_config(suite_payload,'resource_suite_config.json')
    actual={row.get('key'):(row.get('model_id'),row.get('revision'),row.get('dtype')) for row in suite_config.get('models',[])}
    expected={key:EXPECTED_MODELS[key] for key in SUITE_KEYS}
    require(actual==expected and suite_config.get('wall_guard_seconds')==1500 and suite_config.get('timeout_seconds')==1800,'archived suite config differs from pinned six-model contract')
    return {'valid':True,'panel_manifest_sha256':FIXED_PANEL_MANIFEST_SHA256,'omni':omni,'suite':suite,'billing_note':'No total billing is inferred from elapsed resource measurements.'}

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('omni_dir'); p.add_argument('suite_dir'); p.add_argument('--output'); a=p.parse_args(); result=validate(a.omni_dir,a.suite_dir)
    if a.output: Path(a.output).write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
    print(json.dumps(result,indent=2,sort_keys=True))
