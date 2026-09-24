"""Freeze native long-form policy and independently verify Turbo inference."""
import copy
import gc
import hashlib
import importlib.metadata
import inspect
import json
import sys
import time
import traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from experiments.diagnostic_v1 import sha,hf_key
from research.bridge_open.artifact_transport import emit
from research.bridge_open.scoring import score


def main():
    import numpy as np
    import soundfile as sf
    import torch
    import whisper
    from huggingface_hub import snapshot_download
    from transformers import WhisperProcessor,WhisperForConditionalGeneration,pipeline
    start=time.monotonic();out=ROOT/'artifacts/audit';out.mkdir(parents=True,exist_ok=False)
    cfg=json.loads((ROOT/'experiments/config.json').read_text())
    rows=json.loads((ROOT/'experiments/cohort.json').read_text())
    assert len(rows)==len({r['conversation_id'] for r in rows})==38
    short=[r for r in rows if r['duration_seconds']<=30];long=[r for r in rows if r['duration_seconds']>30]
    assert len(short)==24 and len(long)==14
    for filename in ['config.json','cohort.json','native_source_receipt.json']:
        (out/filename).write_bytes((ROOT/'experiments'/filename).read_bytes())
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False
    assert torch.cuda.is_available() and torch.__version__.split('+')[0]=='2.8.0'
    def reseed():torch.manual_seed(cfg['seed']);np.random.seed(cfg['seed'])
    def time_guard():
        if time.monotonic()-start>cfg['max_wall_seconds']:raise TimeoutError('Wall-time budget')
    audio={}
    for r in rows:
        assert sha(ROOT/r['staged_audio'])==r['audio_sha256']
        assert sha(ROOT/r['staged_reference'])==r['reference_sha256']
        a,sr=sf.read(ROOT/r['staged_audio'],dtype='float32')
        assert sr==16000 and a.ndim==1 and len(a)>0 and np.isfinite(a).all() and 0<np.abs(a).max()<=1
        assert abs(len(a)/sr-r['duration_seconds'])<1e-4
        audio[r['conversation_id']]=a
    records=[];environments=[]
    def save(arm,r,text,modelcfg,**extra):
        ref=json.loads((ROOT/r['staged_reference']).read_text())['reference_text']
        record=dict(arm=arm,conversation_id=r['conversation_id'],language=r['language'],
            audio_sha256=r['audio_sha256'],reference_sha256=r['reference_sha256'],
            model_id=modelcfg['model_id'],model_revision=modelcfg['model_revision'],
            input_sample_count=len(audio[r['conversation_id']]),sample_rate=16000,
            text=text,metrics=score(ref,text),**extra)
        records.append(record)
        with (out/'predictions.jsonl').open('a') as f:f.write(json.dumps(record,ensure_ascii=False)+'\n')
        print(json.dumps(dict(event='correction_prediction',model=modelcfg['native_model'],arm=arm,
            id=r['conversation_id'],wer=record['metrics']['wer'],elapsed_seconds=time.monotonic()-start)),flush=True)
    for mc in cfg['models']:
        time_guard();reseed()
        snapshot=Path(snapshot_download(mc['model_id'],revision=mc['model_revision'],
            allow_patterns=['*.json','model.safetensors','*.txt'],token=False))
        processor=WhisperProcessor.from_pretrained(snapshot,local_files_only=True)
        hf=WhisperForConditionalGeneration.from_pretrained(snapshot,local_files_only=True,
            torch_dtype=torch.float32,attn_implementation='sdpa').to('cuda').eval()
        hf.generation_config.forced_decoder_ids=None;baseline=copy.deepcopy(hf.generation_config)
        native=whisper.load_model(mc['native_model'],device='cuda',download_root=str(ROOT/'model_cache')).float().eval()
        checkpoint=ROOT/'model_cache'/f"{mc['native_model']}.pt"
        assert sha(checkpoint)==whisper._MODELS[mc['native_model']].split('/')[-2]
        assert not hf.training and not native.training
        assert native.dims.n_mels==processor.feature_extractor.feature_size==128
        assert processor.feature_extractor.sampling_rate==16000
        hs=hf.model.state_dict();ns=native.state_dict();learned=set(dict(native.named_parameters()));weights=[]
        for key,tensor in ns.items():
            if tensor.is_sparse:continue
            mapped=hf_key(key);other=hs[mapped];assert tensor.shape==other.shape
            delta=float((tensor-other).abs().max())
            assert delta <= (0 if key in learned else 1e-5),(key,delta)
            weights.append(dict(native=key,hf=mapped,learned=key in learned,max_abs_delta=delta))
        assert set(hs)=={r['hf'] for r in weights}
        assert torch.equal(hf.proj_out.weight,hf.model.decoder.embed_tokens.weight)
        features=[]
        for r in rows:
            a=audio[r['conversation_id']][:480000]
            f=processor(a,sampling_rate=16000,return_tensors='pt').input_features[0]
            n=whisper.log_mel_spectrogram(whisper.pad_or_trim(a),n_mels=128)
            assert f.shape==n.shape==(128,3000)
            delta=float((f-n).abs().max());assert delta<1e-4
            features.append(dict(conversation_id=r['conversation_id'],max_abs_delta=delta))
        env=dict(**mc,gpu=torch.cuda.get_device_name(0),dtype=str(next(hf.parameters()).dtype),
            native_dtype=str(next(native.parameters()).dtype),hf_training=hf.training,native_training=native.training,
            native_checkpoint_sha256=sha(checkpoint),hf_weight_sha256={p.name:sha(p) for p in snapshot.glob('*.safetensors')},
            hf_auxiliary_sha256={p.name:sha(p) for p in snapshot.iterdir() if p.is_file() and p.suffix in ['.json','.txt']},
            script_sha256=sha(__file__),native_transcribe_signature=str(inspect.signature(native.transcribe)),
            native_source_sha256={n:sha(Path(whisper.__file__).parent/n) for n in ['audio.py','model.py','decoding.py','transcribe.py','tokenizer.py']},
            hf_generation_config=baseline.to_dict(),packages={n:importlib.metadata.version(n) for n in ['torch','transformers','openai-whisper','numpy','soundfile','jiwer']})
        environments.append(env)
        (out/'environment.json').write_text(json.dumps(environments,indent=2)+'\n')
        (out/f"{mc['native_model']}_weights.json").write_text(json.dumps(weights,indent=2)+'\n')
        (out/f"{mc['native_model']}_features.json").write_text(json.dumps(features,indent=2)+'\n')
        print(json.dumps(dict(event='model_gates_passed',model=mc['native_model'],tensors=len(weights),features=len(features))),flush=True)
        def hf_decode(arrays,language):
            hf.generation_config=copy.deepcopy(baseline)
            inputs=processor(arrays,sampling_rate=16000,padding='max_length',max_length=480000,
                truncation=False,return_tensors='pt',return_attention_mask=True).to('cuda')
            with torch.inference_mode():
                result=hf.generate(**inputs,language=language,task='transcribe',num_beams=5,
                    do_sample=False,return_timestamps=False,max_new_tokens=440,return_dict_in_generate=True)
            ids=result.sequences.cpu().tolist()
            return processor.batch_decode(ids,skip_special_tokens=True),ids
        if mc['short_parity']:
            for language in ['hi','bn','as']:
                time_guard();group=[r for r in short if r['language']==language]
                texts,tokens=hf_decode([audio[r['conversation_id']] for r in group],language)
                for r,text,ids in zip(group,texts,tokens):save('hf_batch',r,text,mc,tokens=ids)
            for r in short:
                time_guard();texts,tokens=hf_decode([audio[r['conversation_id']]],r['language'])
                save('hf_single',r,texts[0],mc,tokens=tokens[0])
            pipe=pipeline('automatic-speech-recognition',model=hf,tokenizer=processor.tokenizer,
                feature_extractor=processor.feature_extractor,device=0,torch_dtype=torch.float32)
            for r in short:
                time_guard();hf.generation_config=copy.deepcopy(baseline)
                with torch.inference_mode():
                    result=pipe(dict(array=audio[r['conversation_id']].copy(),sampling_rate=16000),
                        generate_kwargs=dict(language=r['language'],task='transcribe',num_beams=5,do_sample=False,
                                             return_timestamps=False,max_new_tokens=440))
                save('hf_pipeline',r,result['text'],mc)
            del pipe
            for r in short:
                time_guard();mel=whisper.log_mel_spectrogram(whisper.pad_or_trim(audio[r['conversation_id']]),n_mels=128).to('cuda')
                options=whisper.DecodingOptions(language=r['language'],task='transcribe',fp16=False,
                    beam_size=5,temperature=0.,without_timestamps=True,sample_len=440)
                with torch.inference_mode():result=native.decode(mel,options)
                save('native_decode_matched',r,result.text,mc,tokens=result.tokens,options=vars(options),
                    avg_logprob=result.avg_logprob,no_speech_prob=result.no_speech_prob,compression_ratio=result.compression_ratio)
        # Release HF weights before the full-waveform native route. No independent cuts.
        del hf,hs,ns,other,tensor;gc.collect();torch.cuda.empty_cache()
        for r in long:
            time_guard();reseed();attempts=[];original_decode=native.decode
            def traced_decode(mel,options):
                time_guard()
                result=original_decode(mel,options)
                attempts.append(dict(options=vars(options),tokens=result.tokens,text=result.text,
                    avg_logprob=result.avg_logprob,no_speech_prob=result.no_speech_prob,
                    compression_ratio=result.compression_ratio,
                    token_count_reaches_default_sample_limit=len(result.tokens)>=native.dims.n_text_ctx//2))
                return result
            native.decode=traced_decode
            options=copy.deepcopy(cfg['native_transcribe']);options['temperature']=tuple(options['temperature'])
            try:
                with torch.inference_mode():
                    result=native.transcribe(audio[r['conversation_id']].copy(),language=r['language'],**options)
            finally:native.decode=original_decode
            assert result['language']==r['language']
            save('native_longform',r,result['text'],mc,segments=result['segments'],decode_attempts=attempts,
                effective_options=dict(language=r['language'],**cfg['native_transcribe']),
                empty_output=not result['text'].strip(),native_language=result['language'],
                segment_bounds_valid=all(0<=s['start']<=s['end']<=r['duration_seconds']+.03 for s in result['segments']))
        del native,original_decode;gc.collect();torch.cuda.empty_cache()
    expected=24*4+14*2
    assert len(records)==expected
    summary=dict(status='completed',records=len(records),long_per_model=14,short_turbo_parity=24,
        total_wall_seconds=time.monotonic()-start,cost_usd=None,purpose=cfg['purpose'])
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(dict(event='correction_complete',**summary)),flush=True)


if __name__=='__main__':
    out=ROOT/'artifacts/audit'
    if out.exists():raise SystemExit('Refuse stale artifact directory')
    code=0
    try:main()
    except Exception as e:
        code=1;traceback.print_exc();out.mkdir(parents=True,exist_ok=True)
        (out/'failure.json').write_text(json.dumps(dict(type=type(e).__name__,message=str(e)),indent=2)+'\n')
    finally:
        if out.exists():emit(out)
    raise SystemExit(code)
