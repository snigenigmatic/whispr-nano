"""Auditable cloud decoder study. References never enter model generation.

Fresh Lahaja recordings share heldout speakers: this is an audio-disjoint
robustness extension, not a new speaker-disjoint confirmation. Speech Commands
stress uses concatenated human recordings, not natural conversational repeats.
"""
from __future__ import annotations
import base64
from collections import Counter, defaultdict
import hashlib
import importlib.metadata
import io
import json
from pathlib import Path
import random
import tarfile
import time
import urllib.request
import zlib

import jiwer
import numpy as np
import pyarrow.parquet as pq
import soundfile as sf
import torch
from huggingface_hub import hf_hub_download
from transformers import AutoProcessor, WhisperForConditionalGeneration
from research.lahaja import baseline as b
from research.lahaja.artifact_transport import emit
from publication_study.loop_repair import suppress_adjacent_word_loops

CONFIG = Path(__file__).with_suffix('.json')
COMMAND_URL = 'https://storage.googleapis.com/download.tensorflow.org/data/speech_commands_test_set_v0.02.tar.gz'
COMMAND_SHA = 'cc2a00c1147c2254e9be3fa0f779d8c17421dc349b86366567a8edfa9acd51df'
WORDS = ('yes','no','up','down','left','right','on','off','stop','go')


def sha(data):
    return hashlib.sha256(data).hexdigest()


def prepare():
    token=b._token()
    dev=hf_hub_download(b.DATASET_ID,b.DEV_FILE,revision=b.DATASET_REVISION,repo_type='dataset',token=token)
    dev_speakers=set(pq.read_table(dev,columns=['sp_id']).column('sp_id').to_pylist())
    paths={f:Path(hf_hub_download(b.DATASET_ID,f,revision=b.DATASET_REVISION,repo_type='dataset',token=token)) for f in b.DATA_FILES}
    all_rows,old=b._metadata(paths,dev_speakers)
    old_keys={(r['data_file'],r['row_index']) for r in old}
    fresh=b._pick([r for r in all_rows if (r['data_file'],r['row_index']) not in old_keys])
    assert not old_keys & {(r['data_file'],r['row_index']) for r in fresh}
    assert len(old)==405 and len({r['sp_id'] for r in old})==106
    audio=b._audio(paths,{(r['data_file'],r['row_index']) for r in old+fresh})
    records=[]
    for cohort,rows in [('lahaja_replication',old),('lahaja_fresh_audio',fresh)]:
        for r in rows:
            a,sr=audio[(r['data_file'],r['row_index'])]
            assert sr==16000 and abs(len(a)/sr-r['duration'])<0.12
            records.append(dict(id=f"{r['data_file']}:{r['row_index']}",cohort=cohort,
                sp_id=r['sp_id'],reference=r['verbatim'],language='hi',audio=a,
                audio_sha256=sha(a.astype('<f4').tobytes()),data_file=r['data_file'],row_index=r['row_index'],
                scenario=r['scenario'],native_state=r['native_state'],duration_s=len(a)/sr))
    archive=Path.home()/'.cache/whispr-nano/speech_commands_test_set_v0.02.tar.gz'
    archive.parent.mkdir(parents=True,exist_ok=True)
    if not archive.exists():
        with urllib.request.urlopen(COMMAND_URL,timeout=120) as src,archive.open('wb') as dst:
            while block:=src.read(1024*1024): dst.write(block)
    assert sha(archive.read_bytes())==COMMAND_SHA
    with tarfile.open(archive) as tf:
        members=[m for m in tf.getmembers() if m.isfile() and m.name.endswith('.wav')]
        for word in WORDS:
            candidates=[m for m in members if Path(m.name).parent.name==word]
            candidates.sort(key=lambda m:sha(('20261013|'+m.name).encode()))
            picked=[];speakers=set()
            for m in candidates:
                speaker=Path(m.name).name.split('_nohash_')[0]
                if speaker in speakers:continue
                speakers.add(speaker);picked.append((m,speaker))
                if len(picked)==10:break
            assert len(picked)==10
            for m,speaker in picked:
                raw=tf.extractfile(m).read()
                a,sr=sf.read(io.BytesIO(raw),dtype='float32',always_2d=True)
                assert sr==16000 and a.shape[1]==1 and np.isfinite(a).all()
                a=a[:,0]
                for count in (1,5,8):
                    # Keep original silence and insert exactly 150 ms between copies.
                    pieces=[]
                    for k in range(count):
                        if k:pieces.append(np.zeros(2400,dtype=np.float32))
                        pieces.append(a)
                    repeated=np.concatenate(pieces)
                    records.append(dict(id=f'{m.name}|repeat={count}',cohort=f'commands_repeat_{count}',
                        sp_id=speaker,source_id=m.name,source_audio_sha256=sha(raw),word=word,n_repeats=count,
                        reference=' '.join([word]*count),language='en',audio=repeated,
                        audio_sha256=sha(repeated.astype('<f4').tobytes()),duration_s=len(repeated)/16000))
    return records


def manifest(records):
    return [{k:v for k,v in r.items() if k!='audio'} for r in records]


def score(ref,hyp):
    c=jiwer.process_words(b._normalize(ref),b._normalize(hyp))
    return dict(ref_words=c.hits+c.substitutions+c.deletions,
                word_errors=c.substitutions+c.deletions+c.insertions,
                substitutions=c.substitutions,deletions=c.deletions,insertions=c.insertions)


def summary(rows):
    out={}
    for cohort in sorted({r['cohort'] for r in rows}):
        rr=[r for r in rows if r['cohort']==cohort]
        n=sum(r['raw_counts']['ref_words'] for r in rr)
        e=sum(r['raw_counts']['word_errors'] for r in rr)
        er=sum(r['repair_counts']['word_errors'] for r in rr)
        out[cohort]=dict(clips=len(rr),speakers=len({r['sp_id'] for r in rr}),ref_words=n,
            raw_errors=e,repair_errors=er,raw_wer=e/n if n else None,repair_wer=er/n if n else None,
            raw_exact=sum(b._normalize(r['prediction'])==b._normalize(r['reference']) for r in rr),
            repair_exact=sum(b._normalize(r['repaired_prediction'])==b._normalize(r['reference']) for r in rr),
            raw_empty=sum(not b._normalize(r['prediction']) for r in rr),
            repair_empty=sum(not b._normalize(r['repaired_prediction']) for r in rr),
            altered=sum(r['removed_words']>0 for r in rr),
            helped=sum(r['repair_counts']['word_errors']<r['raw_counts']['word_errors'] for r in rr),
            harmed=sum(r['repair_counts']['word_errors']>r['raw_counts']['word_errors'] for r in rr))
    return out


def make_decoder(mode):
    if mode=='hf_beam5':
        proc=AutoProcessor.from_pretrained(b.MODEL_ID,revision=b.MODEL_REVISION)
        model=WhisperForConditionalGeneration.from_pretrained(b.MODEL_ID,revision=b.MODEL_REVISION,
            dtype=torch.float16,attn_implementation='sdpa').to('cuda').eval()
        def decode(audio,language):
            inputs=proc(audio,sampling_rate=16000,return_tensors='pt')
            features=inputs.input_features.to('cuda',dtype=torch.float16)
            tokens=model.generate(features,language=language,task='transcribe',num_beams=5,
                do_sample=False,max_new_tokens=256)
            return proc.batch_decode(tokens,skip_special_tokens=True)[0].strip(),dict(generated_tokens=int(tokens.shape[-1]))
        return decode,dict(model=b.MODEL_ID,revision=b.MODEL_REVISION,dtype='float16',num_beams=5,
            max_new_tokens=256,attention_mask='not passed; exact prior single-clip padded recipe')
    if mode in ('official_default','official_no_speech_skip','official_turbo'):
        import whisper
        model_name='turbo' if mode=='official_turbo' else 'small'
        model=whisper.load_model(model_name,device='cuda').eval()
        url=whisper._MODELS[model_name]
        def decode(audio,language):
            result=model.transcribe(audio,language=language,task='transcribe',verbose=None,
                beam_size=5,best_of=5,fp16=True,
                no_speech_threshold=None if mode=='official_no_speech_skip' else .6)
            meta=[{k:s[k] for k in ('start','end','temperature','avg_logprob','compression_ratio','no_speech_prob')} for s in result['segments']]
            return result['text'].strip(),dict(segments=meta)
        return decode,dict(model='openai-whisper/'+model_name,url=url,checkpoint_sha256=url.split('/')[-2],
            dtype='float16',temperature=[0,.2,.4,.6,.8,1.],beam_size=5,best_of=5,
            compression_ratio_threshold=2.4,logprob_threshold=-1.,no_speech_threshold=None if mode=='official_no_speech_skip' else .6,
            condition_on_previous_text=True,implementation='openai-whisper==20250625')
    raise ValueError(mode)


def main():
    cfg=json.loads(CONFIG.read_text())
    assert torch.cuda.is_available(), 'Cloud CUDA required; no local fallback'
    torch.set_num_threads(4)
    torch.manual_seed(cfg['seed']); np.random.seed(cfg['seed']); random.seed(cfg['seed'])
    started=time.monotonic()
    from research.lahaja.short_hindi import prepare as prepare_short
    rows=prepare_short();ma=manifest(rows)
    digest=sha(json.dumps(ma,sort_keys=True,ensure_ascii=False).encode())
    assert digest==cfg['expected_manifest_sha256'],digest
    out=Path.home()/'.local/share/openresearch'/cfg['run_id'];out.mkdir(parents=True,exist_ok=True)
    (out/'manifest.json').write_text(json.dumps(ma,sort_keys=True,ensure_ascii=False))
    decode,dcfg=make_decoder(cfg['mode'])
    env=dict(config=cfg,decoder=dcfg,dataset_revision=b.DATASET_REVISION,
        dataset_protocol='short_hindi_v1',silence_revision='8126be4a4b83afefeef9394ff1312b635a499d82',
        gpu=torch.cuda.get_device_name(0),packages={p:importlib.metadata.version(p) for p in
            ['torch','transformers','numpy','pyarrow','soundfile','openai-whisper','jiwer','huggingface-hub']},
        code_sha256=sha(Path(__file__).read_bytes()),manifest_sha256=sha((out/'manifest.json').read_bytes()),
        repair_policy_sha256=sha((Path(__file__).parents[2]/'src/publication_study/loop_repair.py').read_bytes()))
    (out/'environment.json').write_text(json.dumps(env,indent=2));print(json.dumps(dict(event='preflight',**env)),flush=True)
    predictions=[];batch=[];start_decode=time.monotonic()
    try:
        with (out/'predictions.jsonl').open('w') as fh, torch.inference_mode():
            for pos,row in enumerate(rows,1):
                if time.monotonic()-start_decode>cfg['max_inference_seconds']:
                    raise TimeoutError('Committed inference time budget reached')
                # Seed separately per recording so prior clip fallbacks do not alter sampling.
                torch.manual_seed(int(sha(row['id'].encode())[:8],16))
                t=time.monotonic();hyp,meta=decode(row['audio'],row['language'])
                torch.cuda.synchronize();elapsed=time.monotonic()-t
                repaired,removed=suppress_adjacent_word_loops(hyp)
                r={k:v for k,v in row.items() if k!='audio'}
                r.update(prediction=hyp,repaired_prediction=repaired,removed_words=removed,
                    raw_counts=score(row['reference'],hyp),repair_counts=score(row['reference'],repaired),
                    decode_seconds=elapsed,decoder_metadata=meta)
                fh.write(json.dumps(r,ensure_ascii=False)+'\n');fh.flush();predictions.append(r);batch.append(r)
                if pos%25==0 or pos==len(rows):
                    raw=json.dumps(batch,ensure_ascii=False).encode()
                    print(json.dumps(dict(event='prediction_chunk',end=pos,sha256=sha(raw),
                        payload=base64.b64encode(zlib.compress(raw)).decode())),flush=True);batch=[]
                    print(json.dumps(dict(event='progress',done=pos,total=len(rows),cohort=row['cohort'],
                        elapsed_s=round(time.monotonic()-started,1))),flush=True)
    finally:
        result=dict(status='complete' if len(predictions)==len(rows) else 'partial',
            expected=len(rows),completed=len(predictions),cohorts=summary(predictions),
            elapsed_s=time.monotonic()-started,predictions_sha256=sha((out/'predictions.jsonl').read_bytes()))
        (out/'summary.json').write_text(json.dumps(result,indent=2))
        emit(out)
    return result
