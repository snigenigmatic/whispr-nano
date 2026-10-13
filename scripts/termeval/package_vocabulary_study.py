"""Independently reconstruct every candidate decision and paired outcome."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import numpy as np
from package_cloud_decoder_study import recover,sha,normalize,counts,paired
from package_short_hindi_study import describe,validate as validate_baseline

MODES=('sum_ll','mean_ll','sum_prior_corrected','mean_prior_corrected')
VOCABULARY='9ee472963446475f3bfd67081bf357ea703f001c7772598bc14d15a85632e849'


def exact_effect(rows,a,b):
    grouped=defaultdict(lambda:np.zeros(3,dtype=np.int64))
    for r in rows:
        ref=normalize(r['reference'])
        grouped[r['sp_id']]+=np.array([normalize(r[a])==ref,normalize(r[b])==ref,1])
    arr=np.array([grouped[k] for k in sorted(grouped)])
    totals=arr.sum(0);rng=np.random.default_rng(20261013)
    draws=rng.integers(0,len(arr),size=(4000,len(arr)));rep=arr[draws].sum(1)
    return dict(first_correct=int(totals[0]),second_correct=int(totals[1]),
        difference_points=float(100*(totals[1]-totals[0])/totals[2]),
        paired_speaker_bootstrap_95_ci_points=np.percentile(100*(rep[:,1]-rep[:,0])/rep[:,2],[2.5,97.5]).tolist(),
        correct_gained=sum(normalize(r[a])!=normalize(r['reference'])==normalize(r[b]) for r in rows),
        correct_lost=sum(normalize(r[a])==normalize(r['reference'])!=normalize(r[b]) for r in rows))


def verify_choices(r,vocab):
    raw=normalize(r['baseline_prediction']);expected=['']+vocab+([raw] if raw not in ['']+vocab else [])
    assert r['candidates']==expected, 'Candidate list differs from development vocabulary and baseline only'
    s=np.asarray(r['log_likelihoods']);z=np.asarray(r['null_log_likelihoods']);n=np.asarray(r['token_counts'])
    assert s.shape==z.shape==n.shape==(len(expected),)
    assert np.isfinite(s).all() and np.isfinite(z).all() and (s<=0).all() and (z<=0).all()
    assert (n>=1).all() and np.equal(n,np.floor(n)).all() and (n<=445).all()
    values=dict(sum_ll=s,mean_ll=s/n,sum_prior_corrected=s-z,mean_prior_corrected=(s-z)/n)
    winners={k:int(v.argmax()) for k,v in values.items()}
    assert r['selected_indices']==winners, 'Exported decisions disagree with likelihoods'
    assert r['predictions']=={k:expected[i] for k,i in winners.items()}
    assert np.isfinite(r['scoring_seconds']) and r['scoring_seconds']>=0


def main():
    p=argparse.ArgumentParser();p.add_argument('--log',required=True);p.add_argument('--baseline',required=True);p.add_argument('--out',required=True);args=p.parse_args()
    log=Path(args.log);folder=Path.home()/'.local/share/openresearch/recovered-cloud'/log.stem
    if not folder.exists():recover(log,folder)
    env=json.loads((folder/'environment.json').read_text());summary=json.loads((folder/'summary.json').read_text())
    manifest=json.loads((folder/'manifest.json').read_text());vocab=json.loads((folder/'vocabulary.json').read_text())
    rows=[json.loads(l) for l in (folder/'predictions.jsonl').read_text().splitlines()]
    assert summary['status']=='complete' and summary['completed']==summary['expected']==len(rows)==450
    assert sha(folder/'predictions.jsonl')==summary['predictions_sha256']
    assert sha(folder/'manifest.json')==env['manifest_sha256']
    assert sha(folder/'vocabulary.json')==env['vocabulary_sha256']==VOCABULARY
    assert sorted(set(vocab))==vocab and len(vocab)==92
    # Independent tokenizer implementation, pinned to the matching HF checkpoint.
    # This loads text vocabulary files only; no model weights or local inference.
    from transformers import AutoTokenizer
    tokenizer_revision='973afd24965f72e36ca33b3055d56a652f456b4d'
    tokenizer=AutoTokenizer.from_pretrained('openai/whisper-small',revision=tokenizer_revision,local_files_only=True)
    assert env['prefix_tokens']==tokenizer.convert_tokens_to_ids(['<|startoftranscript|>','<|hi|>','<|transcribe|>','<|notimestamps|>'])
    token_lengths={}
    evidence=json.loads(Path(args.baseline).read_text())['runs']['official_default']
    old,receipt=validate_baseline(Path(evidence['private_artifacts']))
    assert receipt['predictions_sha256']==env['config']['baseline_predictions_sha256']==evidence['predictions_sha256']
    assert receipt['environment']==env['baseline_environment']
    old={r['id']:r for r in old if r['cohort'] in ('hindi_short_natural','silence_control')}
    assert set(old)=={r['id'] for r in rows} and len({r['id'] for r in rows})==450
    for r,m in zip(rows,manifest):
        previous=old[r['id']]
        assert all(r[k]==v==previous[k] for k,v in m.items())
        assert r['baseline_prediction']==previous['prediction']
        assert r['in_development_vocabulary']==(normalize(r['reference']) in vocab)
        # This run's context abstention must be explicitly handled before acceptance.
        assert 'abstention' not in r, 'Context abstention requires a separate token-length audit'
        verify_choices(r,vocab)
        for text in r['candidates']:
            if text not in token_lengths:
                token_lengths[text]=len(tokenizer.encode(' '+text,add_special_tokens=False))+1 if text else 1
        assert r['token_counts']==[token_lengths[t] for t in r['candidates']]
        r['baseline']=r['baseline_prediction'];r.update(r['predictions']);r['raw_counts']=counts(r['reference'],r['baseline'])
    cohorts={};contrasts={}
    groups=dict(all_speech=[r for r in rows if r['cohort']=='hindi_short_natural'],
        in_vocabulary=[r for r in rows if r['in_development_vocabulary']],
        out_of_vocabulary=[r for r in rows if r['cohort']=='hindi_short_natural' and not r['in_development_vocabulary']],
        silence=[r for r in rows if r['cohort']=='silence_control'])
    assert {k:len(v) for k,v in groups.items()}==dict(all_speech=414,in_vocabulary=196,out_of_vocabulary=218,silence=36)
    for name,rr in groups.items():
        cohorts[name]={mode:describe(rr,mode) for mode in ('baseline',)+MODES}
        if name=='silence':continue
        contrasts[name]={}
        for a,b in [('mean_ll','mean_prior_corrected')]+[('baseline',m) for m in MODES]+[('sum_ll','sum_prior_corrected')]:
            contrasts[name][f'{b}_minus_{a}']=dict(exact=exact_effect(rr,a,b),
                wer=paired(rr,lambda r:counts(r['reference'],r[a])['word_errors'],lambda r:counts(r['reference'],r[b])['word_errors']))
    result=dict(status='independently_validated',run_id=log.stem,environment=env,summary=summary,
        private_artifacts=str(folder),log_sha256=sha(log),validation_script_sha256=sha(__file__),
        source_evidence_sha256=sha(args.baseline),cohorts=cohorts,contrasts=contrasts,
        independent_tokenizer=dict(model='openai/whisper-small',revision=tokenizer_revision,implementation='Transformers AutoTokenizer',eot=tokenizer.eos_token_id),
        bootstrap=dict(seed=20261013,resamples=4000,cluster='speaker'),
        limits=['Outcome-motivated exploratory candidate scoring on previously exposed evaluation speakers; not an independent confirmation.',
                'Fixed development vocabulary covers 196/414 speech clips; this is an isolated-word recognition task, not general ASR.',
                'Digital-zero likelihood is a proxy for a language prior; prior correction is not asserted to be novel.',
                'Constructed silence controls do not estimate a population false-positive rate.'])
    Path(args.out).write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n')
    print(json.dumps({'cohorts':cohorts,'primary':contrasts['all_speech']['mean_prior_corrected_minus_mean_ll']},indent=2))


if __name__=='__main__':main()
