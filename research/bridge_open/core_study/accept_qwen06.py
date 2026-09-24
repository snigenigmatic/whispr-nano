"""Independently accept the two frozen Qwen 0.6B core conditions."""
from __future__ import annotations
import argparse, gzip, hashlib, importlib.util, json, math, tarfile
from pathlib import Path
from research.bridge_open.scoring import score, normalize_v1
from research.bridge_open.accept_svarah_runs import portable_archive_path, require as req, require_equal as eq
from research.bridge_open.validate_svarah_inputs import validate as validate_svarah_source

ROOT=Path(__file__).resolve().parents[3]
MODEL='Qwen/Qwen3-ASR-0.6B'; REV='5eb144179a02acc5e5ba31e748d22b0cf3e303b0'
ART=ROOT/'research/bridge_open/artifacts'
CODE={'run_qwen.py':'b7e2c3843fcab145bd500c7e3be529314fdf7f5a075207c683ec86a03b6c4315','scoring.py':'a63c0fdce68e33810bf233c813a5ebad0f3692e6718aad2828a802b2e776ff3c','qwen_trace.py':'6331d3c4636351769a4db35812fc6cac5e4186ec14bfe532bdceb220d60a4a3f'}
CONDITIONS={
 'svarah':{'count':518,'language':'en','transcribe_language':'English','commit':'91257030585090522739144688e4c1b99fcf68ef','manifest':ROOT/'data/bridge_open/svarah_native_shard02_full_v1/bundle/manifest.jsonl'},
 'hindi':{'count':418,'language':'hi','transcribe_language':'Hindi','commit':'9886373','manifest':ROOT/'data/bridge_open/fleurs_test_v1/corpus-v2.jsonl'}}

def blob_sha(x):return hashlib.sha256(x).hexdigest()
def sha(path):return blob_sha(Path(path).read_bytes())
def read(path):return json.loads(Path(path).read_text())
def rows(path):return [json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()]

def archive_payloads(run, receipt):
    eq(receipt.get('provider_status'),'done','Provider is not complete')
    archive=portable_archive_path(run,receipt); blob=archive.read_bytes()
    eq(blob_sha(blob),receipt.get('portable_archive_sha256'),'Archive digest')
    eq(blob_sha(gzip.decompress(blob)),receipt.get('sourceDigest'),'Source tar digest')
    out={}
    with tarfile.open(archive,'r:gz') as tf:
        for member in tf:
            if not member.isfile():continue
            name=member.name.removeprefix('./');req(name not in out,'Duplicate archive member')
            out[name]=tf.extractfile(member).read()
    return out

def summary_check(summary,preds,track):
    spec=CONDITIONS[track];n=spec['count'];eq(len(preds),n,'Prediction count')
    errors=sum(sum(r['metrics'][k] for k in ['substitutions','deletions','insertions']) for r in preds)
    words=sum(r['metrics']['ref_words'] for r in preds);chars=sum(r['metrics']['ref_chars'] for r in preds)
    cerr=sum(r['metrics']['character_errors'] for r in preds);req(words>0 and chars>0,'Empty references')
    eq(summary.get('attempted'),n,'Summary attempted');eq(summary.get('scored'),n,'Summary scored')
    eq(summary.get('successful'),sum(r['status']=='success' for r in preds),'Summary successful')
    p=summary.get('per_language',{}).get(spec['language']);req(isinstance(p,dict),'Missing per-language summary')
    for k,v in {'attempted_count':n,'scored_count':n,'word_errors':errors,'ref_words':words}.items():eq(p.get(k),v,'Summary '+k)
    for k,v in {'micro_wer':errors/words,'micro_cer':cerr/chars,'mean_conversation_wer':sum(r['metrics']['wer'] for r in preds)/n,'mean_conversation_wil':sum(r['metrics']['wil'] for r in preds)/n}.items():
        req(isinstance(p.get(k),(int,float)) and math.isclose(p[k],v,rel_tol=1e-12,abs_tol=1e-12),'Summary '+k)
    return errors,words,cerr,chars

def validate(run,track):
    run=Path(run);spec=CONDITIONS[track];n=spec['count'];manifest=spec['manifest']
    cfg,env,summary=[read(run/(s+'.json')) for s in ['config','environment','summary']]
    req(read(run/'remote_validation.json').get('valid') is True,'Invalid remote evidence')
    receipt=read(run/'orx_source_receipt.json');req(receipt.get('commit','').startswith(spec['commit']),'Unexpected executed commit')
    payloads=archive_payloads(run,receipt)
    committed=json.loads(payloads['research/bridge_open/run_config.json'])
    ignore={'output_dir','manifest_path'}
    eq({k:v for k,v in cfg.items() if k not in ignore},{k:v for k,v in committed.items() if k not in ignore},'Effective config changed')
    for k,v in {'model_id':MODEL,'model_revision':REV,'batch_size':8,'num_beams':1,'max_new_tokens':512,'torch_dtype':'bfloat16','transcribe_language':spec['transcribe_language'],'inference_api':'Qwen3ASRModel.transcribe','supported_languages':[spec['language']]}.items():eq(cfg.get(k),v,'Config '+k)
    eq(cfg.get('manifest_sha256'),sha(manifest),'Source manifest hash');eq(env.get('manifest_sha256'),sha(manifest),'Environment manifest hash')
    eq(payloads[cfg['source_manifest']],manifest.read_bytes(),'Archived source manifest')
    eq(env.get('config_sha256'),sha(run/'config.json'),'Config digest');eq(env.get('actual_model_dtype'),'torch.bfloat16','Model dtype');req(env.get('model_eval') is True,'Eval mode')
    eq(env.get('code_sha256'),CODE,'Trusted inference code')
    for name,digest in CODE.items():eq(blob_sha(payloads['research/bridge_open/'+name]),digest,'Archived code '+name)
    for name,digest in read(run/'hf_receipt.json').get('source_sha256',{}).items():eq(blob_sha(payloads[name]),digest,'Worker source '+name)
    assets=read(Path(__file__).with_name('qwen06_remote_assets.json'));eq(assets['revision'],REV,'Asset pin')
    eq(env.get('weight_sha256'),{k:v['sha256'] for k,v in assets['files'].items() if k.endswith('.safetensors')},'Weights do not match pinned HF LFS')
    for name,digest in env.get('auxiliary_file_sha256',{}).items():
        req(name in assets['files'],'Unknown auxiliary asset');eq(digest,assets['files'][name]['sha256'],'Auxiliary asset '+name)
    req(bool(env.get('auxiliary_file_sha256')),'Missing auxiliary files')
    eq(env.get('packages',{}).get('qwen-asr'),'0.0.6','Qwen API package version')
    gen=env.get('effective_generation_config',{});eq(gen.get('do_sample'),False,'Sampling enabled');eq(gen.get('num_beams'),1,'Search differs')
    parent=read(ART/'qwen_native_full_hi_v1/environment.json')
    for name,digest in parent.get('qwen_asr_package_file_sha256',{}).items():
        if name.endswith('.py'):eq(env.get('qwen_asr_package_file_sha256',{}).get(name),digest,'Qwen API code changed')
    source=[r for r in rows(manifest) if r['language']==spec['language']];eq(len(source),n,'Source cohort count')
    byid={r['conversation_id']:r for r in source};eq(len(byid),n,'Source duplicate IDs')
    eq(set(cfg['conversation_ids']),set(byid),'Config IDs');eq(len(cfg['conversation_ids']),n,'Config duplicates')
    if track=='svarah':
        audit=validate_svarah_source(manifest.parent,n)
        eq(json.loads(payloads[cfg['input_acceptance_receipt']]),audit,'Archived independent source validation')
        eq(blob_sha(payloads[cfg['input_acceptance_receipt']]),cfg['input_acceptance_receipt_sha256'],'Source receipt digest')
        prepared=json.loads(payloads[cfg['input_receipt_path']]);eq(prepared.get('cohort_count'),n,'Wrong preparation cohort');eq(prepared.get('cohort_manifest_sha256'),sha(manifest),'Wrong preparation manifest')
    preds=rows(run/'predictions.jsonl');eq(len(preds),n,'Prediction count');eq(len({r['conversation_id'] for r in preds}),n,'Duplicate prediction IDs');eq({r['conversation_id'] for r in preds},set(byid),'Prediction cohort')
    # Execute only the already hash-verified trace validator from the immutable source snapshot.
    trace_path=run/'executed_qwen_trace.py';eq(sha(trace_path),CODE['qwen_trace.py'],'Trace validator digest')
    sp=importlib.util.spec_from_file_location('verified_qwen06_trace',trace_path);trace=importlib.util.module_from_spec(sp);sp.loader.exec_module(trace)
    for p in preds:
        s=byid[p['conversation_id']]
        if track=='svarah':
            audio=manifest.parent/'wav'/Path(s['audio_path']).name;ref=manifest.parent/'references'/Path(s['reference_path']).name
            for pk,hk in [('audio_path','audio_sha256'),('reference_path','reference_sha256')]:eq(blob_sha(payloads[s[pk]]),s[hk],'Archived input digest')
        else:audio=Path(s['audio_path']);ref=Path(s['reference_path'])
        eq(sha(audio),s['audio_sha256'],'Local audio digest');eq(sha(ref),s['reference_sha256'],'Local reference digest')
        for k,v in {'audio_sha256':s['audio_sha256'],'reference_sha256':s['reference_sha256'],'model_id':MODEL,'model_revision':REV,'language':spec['language']}.items():eq(p.get(k),v,'Prediction '+k)
        req(p.get('status') in ('success','empty_output'),'Incomplete inference status')
        eq(p.get('metrics'),score(read(ref)['reference_text'],p['text_raw']),'Recomputed metrics');eq(p.get('text_for_scoring'),normalize_v1(p['text_raw']),'Normalized text')
        eq(len(p.get('chunks',[])),1,'Full-waveform chunks');c=p['chunks'][0]
        req(c.get('start_seconds')==0 and abs(c.get('end_seconds',-1)-s['duration_seconds'])<1e-5,'Waveform bounds')
        req(not trace.validate_trace(c.get('generation_trace'),cfg),'Invalid raw generation trace')
    errors,words,cerr,chars=summary_check(summary,preds,track)
    result=dict(valid=True,track=track,model=MODEL,revision=REV,n=n,word_errors=errors,reference_words=words,micro_wer=errors/words,character_errors=cerr,reference_chars=chars,micro_cer=cerr/chars,receipt=receipt,weight_sha256=env['weight_sha256'],scope='Fixed public cohort; source/runtime acceptance, no held-out method or speaker-population inference')
    (run/'local_independent_acceptance.json').write_text(json.dumps(result,indent=2)+'\n');return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('run');p.add_argument('--track',choices=CONDITIONS,required=True);a=p.parse_args();print(json.dumps(validate(a.run,a.track),indent=2))
