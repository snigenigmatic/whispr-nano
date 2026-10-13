"""Reuse pinned cloud predictions; verify repeated words with model alignment."""
from collections import Counter
import base64
import hashlib
import importlib.metadata
import json
from pathlib import Path
import time
import zlib
import numpy as np
import torch
import whisper
from whisper.timing import find_alignment
from whisper.tokenizer import get_tokenizer
from huggingface_hub import HfApi
from publication_study.loop_repair import suppress_adjacent_word_loops
from publication_study.alignment_repair import repair_zero_duration
from research.lahaja.cloud_study import prepare,manifest,sha,score,summary
from research.lahaja.artifact_transport import emit,recover

CONFIG=Path(__file__).with_suffix('.json')


def main():
    cfg=json.loads(CONFIG.read_text())
    assert torch.cuda.is_available(),'CUDA cloud execution required'
    torch.set_num_threads(4);torch.manual_seed(20261013)
    started=time.monotonic();rows=prepare();ma=manifest(rows)
    out=Path.home()/'.local/share/openresearch'/cfg['run_id'];out.mkdir(parents=True,exist_ok=True)
    (out/'manifest.json').write_text(json.dumps(ma,sort_keys=True,ensure_ascii=False))
    assert sha((out/'manifest.json').read_bytes())==cfg['manifest_sha256']
    cache=out.parent/(cfg['run_id']+'-source')
    log=out.parent/(cfg['run_id']+'-source.log')
    log.write_text('\n'.join(HfApi().fetch_job_logs(job_id=cfg['source_job_id'],namespace='snigenigmatic',follow=False)))
    recover(log,cache)
    p=cache/'predictions.jsonl'
    assert sha(p.read_bytes())==cfg['source_predictions_sha256']
    source={r['id']:r for r in map(json.loads,p.read_text().splitlines())}
    assert len(source)==len(rows)==1101 and json.loads((cache/'summary.json').read_text())['status']=='complete'
    source_env=json.loads((cache/'environment.json').read_text())
    assert source_env['config']['mode']==cfg['source_mode']
    model=whisper.load_model(cfg['alignment_model_name'],device='cuda').eval()
    tokenizers={language:get_tokenizer(model.is_multilingual,num_languages=model.num_languages,language=language,task='transcribe') for language in ('hi','en')}
    env=dict(config=cfg,decoder=source_env['decoder'],source_environment=source_env,
        dataset_revision=source_env['dataset_revision'],gpu=torch.cuda.get_device_name(0),
        manifest_sha256=sha((out/'manifest.json').read_bytes()),
        repair_policy_sha256=source_env['repair_policy_sha256'],
        alignment_policy_sha256=sha((Path(__file__).parents[2]/'src/publication_study/alignment_repair.py').read_bytes()),
        alignment_model_url=whisper._MODELS[cfg['alignment_model_name']],
        packages={n:importlib.metadata.version(n) for n in ['torch','openai-whisper','transformers','numpy','jiwer']},
        code_sha256=sha(Path(__file__).read_bytes()))
    (out/'environment.json').write_text(json.dumps(env,indent=2));print(json.dumps(dict(event='preflight',**env)),flush=True)
    predictions=[];batch=[];t0=time.monotonic()
    try:
        with (out/'predictions.jsonl').open('w') as fh,torch.inference_mode():
            for i,row in enumerate(rows,1):
                if time.monotonic()-t0>cfg['max_inference_seconds']:raise TimeoutError('Alignment time budget reached')
                base=source[row['id']]
                assert (base['reference'],base['audio_sha256'],base['sp_id'])==(row['reference'],row['audio_sha256'],row['sp_id'])
                r=dict(base);hyp=base['prediction'];_,trigger=suppress_adjacent_word_loops(hyp)
                repaired=hyp;alignment=[];meta=dict(status='not_triggered',removed=0)
                ts=time.monotonic()
                if trigger:
                    tokenizer=tokenizers[row['language']];tokens=tokenizer.encode(hyp)
                    if len(tokens)>model.dims.n_text_ctx-len(tokenizer.sot_sequence)-2:
                        meta=dict(status='abstain_context_length',removed=0)
                    else:
                        try:
                            mel=whisper.log_mel_spectrogram(row['audio'],n_mels=model.dims.n_mels)
                            frames=mel.shape[-1]
                            assert frames<=3000
                            mel=whisper.pad_or_trim(mel,3000).to('cuda',dtype=torch.float16)
                            aligned=find_alignment(model,tokenizer,tokens,mel,frames)
                            alignment=[dict(word=w.word,start=float(w.start),end=float(w.end),probability=float(w.probability)) for w in aligned]
                            assert all(np.isfinite(w[k]) for w in alignment for k in ('start','end','probability'))
                            repaired,meta=repair_zero_duration(hyp,alignment)
                        except Exception as e:
                            # Keep the clip and unmodified output. Do not hide alignment failures.
                            meta=dict(status='alignment_error',removed=0,error=f'{type(e).__name__}: {e}')
                torch.cuda.synchronize()
                r.update(repaired_prediction=repaired,removed_words=meta['removed'],
                    repair_counts=score(row['reference'],repaired),alignment=alignment,alignment_status=meta,
                    alignment_seconds=time.monotonic()-ts)
                predictions.append(r);batch.append(r);fh.write(json.dumps(r,ensure_ascii=False)+'\n');fh.flush()
                if i%25==0 or i==len(rows):
                    raw=json.dumps(batch,ensure_ascii=False).encode()
                    print(json.dumps(dict(event='prediction_chunk',end=i,sha256=sha(raw),payload=base64.b64encode(zlib.compress(raw)).decode())),flush=True);batch=[]
                    print(json.dumps(dict(event='progress',done=i,total=len(rows),elapsed_s=round(time.monotonic()-started,1),
                        statuses=dict(Counter(r['alignment_status']['status'] for r in predictions)))),flush=True)
    finally:
        result=dict(status='complete' if len(predictions)==len(rows) else 'partial',expected=len(rows),completed=len(predictions),
            cohorts=summary(predictions),elapsed_s=time.monotonic()-started,predictions_sha256=sha((out/'predictions.jsonl').read_bytes()),
            alignment_statuses=dict(Counter(r['alignment_status']['status'] for r in predictions)))
        (out/'summary.json').write_text(json.dumps(result,indent=2));emit(out)
    return result
