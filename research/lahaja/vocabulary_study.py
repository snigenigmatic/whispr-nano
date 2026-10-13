"""Fixed development lexicon plus original-output escape candidate, on CUDA."""
from __future__ import annotations
import hashlib
import importlib.metadata
import json
from pathlib import Path
import time
import numpy as np
import pyarrow.parquet as pq
import torch
import whisper
from huggingface_hub import HfApi,hf_hub_download
from research.lahaja import baseline as b
from research.lahaja.artifact_transport import emit,recover
from research.lahaja.short_hindi import prepare
from publication_study.candidate_scores import summed_target_logprobs,choose


def sha(data):return hashlib.sha256(data).hexdigest()


def development_vocabulary():
    path=hf_hub_download(b.DATASET_ID,b.DEV_FILE,revision=b.DATASET_REVISION,repo_type='dataset',token=b._token())
    rows=pq.read_table(path,columns=['verbatim','duration','sp_id']).to_pylist()
    words=sorted({b._normalize(r['verbatim']) for r in rows
                  if 0<float(r['duration'])<2 and len(b._normalize(r['verbatim']).split())==1})
    assert len(words)==92 and sha(json.dumps(words,ensure_ascii=False).encode())=='9ee472963446475f3bfd67081bf357ea703f001c7772598bc14d15a85632e849'
    return words,set(r['sp_id'] for r in rows)


class CandidateScorer:
    def __init__(self,model):
        self.model=model
        self.tokenizer=whisper.tokenizer.get_tokenizer(model.is_multilingual,num_languages=model.num_languages,language='hi',task='transcribe')
        self.prefix=list(self.tokenizer.sot_sequence_including_notimestamps)

    def tokens(self,text):
        return self.prefix+(self.tokenizer.encode(' '+text) if text else [])+[self.tokenizer.eot]

    def encode_audio(self,audio):
        assert audio.dtype==np.float32 and audio.ndim==1 and np.isfinite(audio).all() and len(audio)<=30*16000
        mel=whisper.log_mel_spectrogram(whisper.pad_or_trim(audio),n_mels=self.model.dims.n_mels).to('cuda').half()
        return self.model.embed_audio(mel.unsqueeze(0))

    def score(self,features,texts,batch_size=16):
        all_scores=[];counts=[]
        for start in range(0,len(texts),batch_size):
            seq=[self.tokens(t) for t in texts[start:start+batch_size]]
            assert all(len(t)-1<=self.model.dims.n_text_ctx for t in seq)
            lengths=torch.tensor([len(t) for t in seq],device='cuda')
            inputs=torch.full((len(seq),int(lengths.max().item())),self.tokenizer.eot,device='cuda',dtype=torch.long)
            for i,t in enumerate(seq):inputs[i,:len(t)]=torch.tensor(t,device='cuda')
            logits=self.model.logits(inputs[:,:-1],features.expand(len(seq),-1,-1))
            scores=summed_target_logprobs(logits,inputs,lengths,len(self.prefix))
            all_scores.extend(scores.cpu().tolist());counts.extend((lengths-len(self.prefix)).cpu().tolist())
        return all_scores,counts


def main():
    cfg=json.loads(Path(__file__).with_name('cloud_study.json').read_text())
    assert torch.cuda.is_available(), 'GPU required; no local neural inference'
    torch.set_num_threads(4);torch.manual_seed(cfg['seed']);np.random.seed(cfg['seed'])
    started=time.monotonic();out=Path.home()/'.local/share/openresearch'/cfg['run_id'];out.mkdir(parents=True,exist_ok=False)
    vocabulary,dev_speakers=development_vocabulary()
    all_rows=prepare();manifest=[{k:v for k,v in r.items() if k!='audio'} for r in all_rows]
    assert sha(json.dumps(manifest,sort_keys=True,ensure_ascii=False).encode())==cfg['expected_manifest_sha256']
    rows=[r for r in all_rows if r['cohort'] in ('hindi_short_natural','silence_control')]
    assert len(rows)==450 and not dev_speakers & {r['sp_id'] for r in rows}
    source=out.parent/(cfg['run_id']+'-baseline');log=out.parent/(cfg['run_id']+'-baseline.log')
    log.write_text('\n'.join(HfApi().fetch_job_logs(job_id=cfg['baseline_job_id'],namespace='snigenigmatic',follow=False)))
    recover(log,source)
    assert sha((source/'predictions.jsonl').read_bytes())==cfg['baseline_predictions_sha256']
    baseline={r['id']:r for r in map(json.loads,(source/'predictions.jsonl').read_text().splitlines())}
    baseline_env=json.loads((source/'environment.json').read_text())
    for row in rows:
        old=baseline[row['id']]
        assert all(old[k]==v for k,v in row.items() if k!='audio')
    (out/'manifest.json').write_text(json.dumps([{k:v for k,v in r.items() if k!='audio'} for r in rows],ensure_ascii=False,sort_keys=True))
    (out/'vocabulary.json').write_text(json.dumps(vocabulary,ensure_ascii=False))
    model=whisper.load_model('small',device='cuda').eval();scorer=CandidateScorer(model)
    assert whisper._MODELS['small']==baseline_env['decoder']['url']
    env=dict(config=cfg,baseline_environment=baseline_env,model_url=whisper._MODELS['small'],
        gpu=torch.cuda.get_device_name(0),packages={p:importlib.metadata.version(p) for p in
            ['torch','openai-whisper','numpy','pyarrow','soundfile','huggingface-hub']},
        code_sha256=sha(Path(__file__).read_bytes()),scoring_sha256=sha((Path(__file__).parents[2]/'src/publication_study/candidate_scores.py').read_bytes()),
        vocabulary_sha256=sha((out/'vocabulary.json').read_bytes()),manifest_sha256=sha((out/'manifest.json').read_bytes()),
        prefix_tokens=scorer.prefix,primary_contrast='mean_prior_corrected versus mean_ll',
        scope='isolated-word recognition with development vocabulary, original-ASR and empty escape candidates',
        probability='unmodified full-vocabulary softmax; score candidate tokens and EOT, exclude forced prefix and padding')
    results=[];fixed=['']+vocabulary;cache={}
    with torch.inference_mode():
        blank=scorer.encode_audio(np.zeros(480000,dtype=np.float32))
        blank_scores,blank_lengths=scorer.score(blank,fixed)
        cache.update({t:(s,n) for t,s,n in zip(fixed,blank_scores,blank_lengths)})
        # Check vectorized evaluation against individual candidates before any outcomes.
        differences=[]
        for i in (0,1,16):
            single,_=scorer.score(blank,[fixed[i]],batch_size=1)
            differences.append(abs(single[0]-blank_scores[i]))
            assert differences[-1]<.12, 'Batched/single log probability mismatch'
        env['batch_single_max_tolerance']=.12
        env['batch_single_observed_differences']=differences
        (out/'environment.json').write_text(json.dumps(env,indent=2));print(json.dumps(dict(event='candidate_preflight',inputs=len(rows),vocabulary=len(vocabulary),**env)),flush=True)
        inference_start=time.monotonic()
        try:
            with (out/'predictions.jsonl').open('w') as fh:
                for pos,row in enumerate(rows,1):
                    if time.monotonic()-inference_start>cfg['max_inference_seconds']:raise TimeoutError('Candidate scoring inference budget')
                    raw=baseline[row['id']]['prediction'];fallback=b._normalize(raw)
                    candidates=fixed+([fallback] if fallback not in fixed else [])
                    record={k:v for k,v in row.items() if k!='audio'}
                    record.update(baseline_prediction=raw,in_development_vocabulary=b._normalize(row['reference']) in vocabulary)
                    if any(len(scorer.tokens(t))-1>model.dims.n_text_ctx for t in candidates):
                        record.update(abstention='original output exceeds model context',predictions={k:raw for k in ('sum_ll','mean_ll','sum_prior_corrected','mean_prior_corrected')})
                    else:
                        start=time.monotonic()
                        if fallback not in cache:
                            z,n=scorer.score(blank,[fallback]);cache[fallback]=(z[0],n[0])
                        features=scorer.encode_audio(row['audio']);scores,lengths=scorer.score(features,candidates)
                        null=[cache[t][0] for t in candidates]
                        assert lengths==[cache[t][1] for t in candidates]
                        winners=choose(scores,null,lengths)
                        record.update(candidates=candidates,log_likelihoods=scores,null_log_likelihoods=null,token_counts=lengths,
                            selected_indices=winners,predictions={k:candidates[i] for k,i in winners.items()},scoring_seconds=time.monotonic()-start)
                    results.append(record);fh.write(json.dumps(record,ensure_ascii=False)+'\n');fh.flush()
                    if pos%25==0:print(json.dumps(dict(event='candidate_progress',completed=pos,expected=len(rows),elapsed=time.monotonic()-inference_start)),flush=True)
        finally:
            summary=dict(status='complete' if len(results)==len(rows) else 'partial',expected=len(rows),completed=len(results),
                elapsed_seconds=time.monotonic()-started,predictions_sha256=sha((out/'predictions.jsonl').read_bytes()))
            (out/'summary.json').write_text(json.dumps(summary,indent=2));emit(out)
    return summary
