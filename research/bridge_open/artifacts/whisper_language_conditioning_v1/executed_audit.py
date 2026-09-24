"""Paired native Whisper language-conditioning diagnostic, with full evidence."""
import copy
import hashlib
import importlib.metadata
import json
import sys
import time
import traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from research.bridge_open.scoring import score
from research.bridge_open.artifact_transport import emit
from research.bridge_open.audit_scripts import distribution


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    import numpy as np
    import soundfile as sf
    import torch
    import whisper
    cfg=json.loads((ROOT/'experiments/config.json').read_text())
    assert sha(ROOT/'experiments/cohort.json')==cfg['cohort_sha256']
    rows=json.loads((ROOT/'experiments/cohort.json').read_text())
    assert len(rows)==len({r['conversation_id'] for r in rows})==cfg['expected_n']
    assert cfg['arms']==['forced','automatic'] and len(cfg['models'])==1
    out=ROOT/'artifacts/audit';out.mkdir(parents=True,exist_ok=False)
    for name in ['config.json','cohort.json','native_source_receipt.json']:
        (out/name).write_bytes((ROOT/'experiments'/name).read_bytes())
    audio={}
    for r in rows:
        assert sha(ROOT/r['staged_audio'])==r['audio_sha256']
        assert sha(ROOT/r['staged_reference'])==r['reference_sha256']
        a,sr=sf.read(ROOT/r['staged_audio'],dtype='float32')
        assert sr==16000 and a.ndim==1 and len(a)>0 and np.isfinite(a).all() and 0<np.abs(a).max()<=1
        assert abs(len(a)/sr-r['duration_seconds'])<1e-4
        audio[r['conversation_id']]=a
    assert torch.cuda.is_available() and torch.__version__.split('+')[0]=='2.8.0'
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False
    started=time.monotonic()
    def guard():
        if time.monotonic()-started>cfg['max_wall_seconds']:raise TimeoutError('Job wall-time limit')
    mc=cfg['models'][0]
    model=whisper.load_model(mc['native_model'],device='cuda').float().eval()
    assert not model.training and str(next(model.parameters()).dtype)=='torch.float32'
    checkpoint=Path.home()/'.cache/whisper'/f"{mc['native_model']}.pt"
    assert sha(checkpoint)==mc['expected_native_checkpoint_sha256']
    environment=dict(model=mc,gpu=torch.cuda.get_device_name(0),model_eval=not model.training,
        actual_dtype=str(next(model.parameters()).dtype),checkpoint_sha256=sha(checkpoint),
        script_sha256=sha(__file__),scorer_sha256=sha(ROOT/'research/bridge_open/scoring.py'),
        native_source_sha256={n:sha(Path(whisper.__file__).parent/n) for n in ['audio.py','model.py','decoding.py','transcribe.py','tokenizer.py']},
        packages={n:importlib.metadata.version(n) for n in ['torch','openai-whisper','numpy','soundfile','jiwer']},
        model_load_seconds=time.monotonic()-started,measured_cost_usd=None)
    (out/'environment.json').write_text(json.dumps(environment,indent=2)+'\n')
    print(json.dumps(dict(event='language_audit_started',config=cfg,environment=environment)),flush=True)
    records=[]
    with (out/'predictions.jsonl').open('x') as f:
        for r in rows:
            guard();a=audio[r['conversation_id']]
            with torch.inference_mode():
                mel=whisper.log_mel_spectrogram(whisper.pad_or_trim(a),n_mels=model.dims.n_mels).to('cuda')
                _,probs=model.detect_language(mel)
            detected=max(probs,key=probs.get)
            for arm in cfg['arms']:
                guard();torch.manual_seed(cfg['seed']);np.random.seed(cfg['seed'])
                attempts=[];original=model.decode;begin=time.monotonic()
                def observed(mel,options):
                    guard();result=original(mel,options)
                    attempts.append(dict(options=vars(options),tokens=result.tokens,text=result.text,
                        avg_logprob=result.avg_logprob,no_speech_prob=result.no_speech_prob,
                        compression_ratio=result.compression_ratio))
                    return result
                model.decode=observed
                options=copy.deepcopy(cfg['native_transcribe']);options['temperature']=tuple(options['temperature'])
                language=r['language'] if arm=='forced' else None
                try:
                    with torch.inference_mode():result=model.transcribe(a.copy(),language=language,**options)
                finally:model.decode=original
                assert result['language']==(r['language'] if arm=='forced' else detected)
                ref=json.loads((ROOT/r['staged_reference']).read_text())['reference_text']
                rd,hd=distribution(ref),distribution(result['text'])
                record=dict(conversation_id=r['conversation_id'],language=r['language'],arm=arm,
                    model_id=mc['model_id'],model_revision=mc['model_revision'],
                    audio_sha256=r['audio_sha256'],reference_sha256=r['reference_sha256'],
                    sample_rate=16000,input_sample_count=len(a),duration_seconds=len(a)/16000,
                    effective_options=dict(language=language,**cfg['native_transcribe']),
                    independent_detected_language=detected,language_probabilities=probs,
                    output_language=result['language'],text_raw=result['text'],segments=result['segments'],
                    decode_attempts=attempts,termination_evidence='native public decode strips EOS; cap count unknown',
                    reference_script=rd,hypothesis_script=hd,
                    script_mismatch=bool(rd['dominant_script'] and hd['dominant_script'] and rd['dominant_script']!=hd['dominant_script']),
                    metrics=score(ref,result['text']),wall_seconds=time.monotonic()-begin,
                    status='native_api_completed',segment_bounds_valid=all(0<=s['start']<=s['end']<=len(a)/16000+.03 for s in result['segments']))
                f.write(json.dumps(record,ensure_ascii=False)+'\n');f.flush();records.append(record)
                print(json.dumps(dict(event='pair_arm_complete',id=r['conversation_id'],arm=arm,language=r['language'],
                    output_language=result['language'],wer=record['metrics']['wer'],script_mismatch=record['script_mismatch'])),flush=True)
    assert len(records)==cfg['expected_n']*2
    summary=dict(status='completed',records=len(records),total_wall_seconds=time.monotonic()-started,measured_cost_usd=None,groups=[])
    for lang in ['hi','bn','as']:
        for arm in cfg['arms']:
            g=[r for r in records if r['language']==lang and r['arm']==arm]
            errors=sum(r['metrics'][k] for r in g for k in ['substitutions','deletions','insertions']);words=sum(r['metrics']['ref_words'] for r in g)
            summary['groups'].append(dict(language=lang,arm=arm,n=len(g),word_errors=errors,ref_words=words,
                micro_wer=errors/words,script_mismatches=sum(r['script_mismatch'] for r in g)))
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(dict(event='language_audit_complete',**summary)),flush=True)

if __name__=='__main__':
    out=ROOT/'artifacts/audit';code=0
    if out.exists():raise SystemExit('Refuse stale artifacts')
    try:main()
    except Exception as e:
        code=1;traceback.print_exc();out.mkdir(parents=True,exist_ok=True)
        (out/'failure.json').write_text(json.dumps(dict(type=type(e).__name__,message=str(e)),indent=2)+'\n')
    finally:
        if out.exists():emit(out)
    raise SystemExit(code)
