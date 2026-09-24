import gzip, hashlib, json, tarfile
from pathlib import Path

import pytest

from research.bridge_open.core_study import accept_resources as acceptance
from research.bridge_open.core_study.resource_profile import FIXED_PANEL_IDS, read_fixed_panel
from research.bridge_open.scoring import score


ROOT=Path(__file__).resolve().parents[1]


def write_json(path, value): path.write_text(json.dumps(value,indent=2)+"\n")


def profile(model_id, precision, runtime=False, key=''):
    durations={row['conversation_id']:row['duration_seconds'] for row in read_fixed_panel()}
    calls=[]
    for pass_index in range(1,4):
        for ordinal,cid in enumerate(FIXED_PANEL_IDS):
            calls.append({'conversation_id':cid,'audio_sha256':next(row['audio_sha256'] for row in read_fixed_panel() if row['conversation_id']==cid),'pass_index':pass_index,'ordinal':ordinal,'status':'success','text':'unit text','text_sha256':hashlib.sha256(b'unit text').hexdigest(),'failure':None,'duration_seconds':durations[cid],'end_to_end_inference_seconds':.5,'real_time_factor':.5/durations[cid],'resident_baseline_memory_bytes':{'allocated_bytes':50,'reserved_bytes':75},'peak_cuda_allocated_bytes':100,'peak_cuda_reserved_bytes':200})
    search={'num_beams':1,'max_new_tokens':512,'language':'English','do_sample':False} if key.startswith('qwen_') else {'num_beams':5,'max_new_tokens':440,'return_timestamps':False,'do_sample':False}
    model={'model_id':model_id,'model_revision':acceptance.EXPECTED_MODELS[key or 'omni'][1],'model_hashes':{'weights':'abc'},'parameter_count':7,'declared_precision':precision,'declared_search':search,'resident_before_measurement':True}
    if runtime: model['runtime_evidence']={'model_training':False,'parameter_count':7,'pipeline_source_sha256':'x'}
    return {'schema_version':'core-resource-profile/v1','study':{'panel_manifest_sha256':acceptance.FIXED_PANEL_MANIFEST_SHA256,'conversation_ids':list(FIXED_PANEL_IDS)},'environment':{'gpu_name':'NVIDIA A100-SXM4-80GB','device_properties':{'total_memory_bytes':80*2**30},'torch_version':'2.8.0','cuda_version':'12.8'},'model_residency':model,'pipeline_configuration':{'batch_size':1},'warmup_calls':[{'conversation_id':cid,'audio_sha256':next(row['audio_sha256'] for row in read_fixed_panel() if row['conversation_id']==cid),'status':'success','text':'unit text','text_sha256':hashlib.sha256(b'unit text').hexdigest(),'failure':None,'duration_seconds':durations[cid]} for cid in FIXED_PANEL_IDS],'timed_calls':calls,'summary':{'failed_calls_including_warmup':0}}


def archive(run, commit, config_name, config):
    source=run/'runner.py'; source.write_text('model.eval()\nsum(parameter.numel() for parameter in model.parameters())')
    config_path=run/config_name; write_json(config_path,config)
    tar=run/'source_snapshot.tar.gz'
    with tarfile.open(tar,'w:gz') as out:
        out.add(source,arcname='runner.py'); out.add(source,arcname='research/bridge_open/core_study/resource_profile.py'); out.add(config_path,arcname=config_name)
    raw=gzip.decompress(tar.read_bytes())
    write_json(run/'orx_source_receipt.json',{'provider_status':'done','commit':commit,'portable_archive':tar.name,'portable_archive_sha256':hashlib.sha256(tar.read_bytes()).hexdigest(),'sourceDigest':hashlib.sha256(raw).hexdigest()})
    digest=hashlib.sha256(source.read_bytes()).hexdigest()
    return {'runner.py':digest,'research/bridge_open/core_study/resource_profile.py':digest}


def quality(rows):
    output=[]
    for row in rows:
        ref=json.loads((ROOT/'data/bridge_open/svarah_native_shard02_v2/bundle/references'/Path(row['reference_path']).name).read_text())['reference_text']
        output.append({'conversation_id':row['conversation_id'],'audio_sha256':row['audio_sha256'],'reference_sha256':row['reference_sha256'],'timed_pass_index':3,'text_raw':'unit text','text_sha256':hashlib.sha256(b'unit text').hexdigest(),'metrics':score(ref,'unit text')})
    return output


def run_dir(path, omni=False):
    path.mkdir(); rows=read_fixed_panel()
    omni_config={'model_id':acceptance.EXPECTED_MODELS['omni'][0],'model_revision':acceptance.EXPECTED_MODELS['omni'][1],'dtype':'bfloat16','batch_size':1,'public_api_calls_per_input_per_phase':1}
    suite_config={'models':[{'key':key,'model_id':value[0],'revision':value[1],'dtype':value[2]} for key,value in acceptance.EXPECTED_MODELS.items() if key!='omni'],'wall_guard_seconds':1500,'timeout_seconds':1800}
    hashes=archive(path,acceptance.OMNI_COMMIT if omni else acceptance.SUITE_COMMIT_PREFIX+'0'*33,'omni_resource_config.json' if omni else 'resource_suite_config.json',omni_config if omni else suite_config)
    acceptance.FIXED_HARNESS_SHA256=hashes['research/bridge_open/core_study/resource_profile.py']
    write_json(path/'remote_validation.json',{'valid':True}); write_json(path/'container_receipt.json',{}); write_json(path/'artifact_transport.json',{})
    if omni:
        model_id, _revision, dtype=acceptance.EXPECTED_MODELS['omni']; p=profile(model_id,dtype,True)
        write_json(path/'resource_profile.json',p); write_json(path/'environment.json',{'source_sha256':hashes,'asset_sha256':{'weights':'abc'},'runtime_evidence':{'model_training':False,'parameter_count':7,'parameter_dtypes':['torch.bfloat16'],'pipeline_source_sha256':'x'}})
        (path/'quality_predictions.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in quality(rows)))
    else:
        for key in acceptance.SUITE_KEYS:
            model_id, _revision, dtype=acceptance.EXPECTED_MODELS[key]; copy=profile(model_id,dtype,False,key)
            write_json(path/f'{key}_resource_profile.json',copy); write_json(path/f'{key}_environment.json',{'source_sha256':hashes,'parameter_count':7,'model_file_sha256':{'weights':'abc'},'model_eval':True,'actual_model_dtype':f"torch.{acceptance.EXPECTED_MODELS[key][2]}",'packages':{'torch':'2.8.0','transformers':'4.57.6','qwen-asr':'0.0.6'}})
            (path/f'{key}_quality_predictions.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in quality(rows)))


def test_accepts_synthetic_complete_resource_layout(tmp_path):
    omni,suite=tmp_path/'omni',tmp_path/'suite'; run_dir(omni,True); run_dir(suite)
    result=acceptance.validate(omni,suite)
    assert result['valid'] is True and result['omni']['quality']['quality_records']==12
    assert result['suite']['models']['whisper_small']['passes'][0]['corpus_rtf']>0


def test_rejects_mutated_timing_text_hash_and_dropped_failure(tmp_path):
    omni,suite=tmp_path/'omni',tmp_path/'suite'; run_dir(omni,True); run_dir(suite)
    path=suite/'whisper_small_resource_profile.json'; data=json.loads(path.read_text()); data['timed_calls'][0]['text_sha256']='wrong'; write_json(path,data)
    with pytest.raises(ValueError,match='text hash'):
        acceptance.validate(omni,suite)


def test_rejects_incomplete_provider_receipt_and_profiled_non_a100(tmp_path):
    omni,suite=tmp_path/'omni',tmp_path/'suite'; run_dir(omni,True); run_dir(suite)
    receipt=json.loads((omni/'orx_source_receipt.json').read_text()); receipt['provider_status']='running'; write_json(omni/'orx_source_receipt.json',receipt)
    with pytest.raises(ValueError,match='provider is not done'):
        acceptance.validate(omni,suite)
    run_dir(tmp_path/'fresh',True)
    bad=tmp_path/'fresh'; profile_path=bad/'resource_profile.json'; data=json.loads(profile_path.read_text()); data['environment']['gpu_name']='CPU'; write_json(profile_path,data)
    with pytest.raises(ValueError,match='not A100'):
        acceptance.validate(bad,suite)


def test_call_check_reports_repeat_difference_and_mixed_failures():
    data=profile(*acceptance.EXPECTED_MODELS['omni'][:1],precision='bfloat16')
    first=data['timed_calls'][0]; first['text']='alternate'; first['text_sha256']=hashlib.sha256(b'alternate').hexdigest()
    failed=data['timed_calls'][12]; failed['status']='failed'; failed['text']=None; failed['text_sha256']=None; failed['failure']={'type':'RuntimeError','message':'planned'}
    data['summary']['failed_calls_including_warmup']=1
    sources={row['conversation_id']:row for row in read_fixed_panel()}
    _timed, failures, report=acceptance._check_calls(data,sources,80*2**30)
    assert failures==1
    assert FIXED_PANEL_IDS[0] in report['repeat_different_ids']
    assert FIXED_PANEL_IDS[0] in report['mixed_failure_ids']
