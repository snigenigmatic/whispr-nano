"""Independently rescore exported cloud ASR artifacts; publish only aggregates."""
from __future__ import annotations
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sys
import unicodedata
import jiwer
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'src'))
from publication_study.artifact_transport import recover
from publication_study.loop_repair import suppress_adjacent_word_loops
from publication_study.alignment_repair import repair_zero_duration,duration_character_cap
POLICY='4a89775ecae07364dd69a661e1ac612a3ffd10c7991f76305b45ae48c8873867'
PINNED={'lahaja':'18dfa27387197bc02b29740d6be96ea266019823e72b27eb592e5450a84854b6',
        'commands':'4d2eca6827649275174c7ea18031a92eff0002e0de702967dbb2ae0a2d4bea37'}


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def normalize(s):
    return ' '.join(''.join(c if unicodedata.category(c)[0] in 'LMN' else ' '
        for c in unicodedata.normalize('NFC',s).lower()).split())

def counts(ref,hyp):
    r=jiwer.process_words(normalize(ref),normalize(hyp))
    return dict(ref_words=r.hits+r.substitutions+r.deletions,word_errors=r.substitutions+r.deletions+r.insertions,
        substitutions=r.substitutions,deletions=r.deletions,insertions=r.insertions)

def paired(rows,first,second):
    grouped=defaultdict(lambda:np.zeros(3,dtype=np.int64))
    for r in rows:grouped[r['sp_id']]+=np.array([first(r),second(r),r['raw_counts']['ref_words']])
    a=np.array([grouped[k] for k in sorted(grouped)])
    totals=a.sum(axis=0);rng=np.random.default_rng(20261013)
    draws=rng.integers(0,len(a),size=(4000,len(a)));replicates=a[draws].sum(axis=1)
    delta=100*(replicates[:,1]-replicates[:,0])/replicates[:,2]
    return dict(clips=len(rows),speakers=len(a),reference_words=int(totals[2]),
        first_word_errors=int(totals[0]),second_word_errors=int(totals[1]),
        first_wer_percent=100*totals[0]/totals[2],second_wer_percent=100*totals[1]/totals[2],
        difference_points=100*(totals[1]-totals[0])/totals[2],
        paired_speaker_bootstrap_95_ci_points=np.percentile(delta,[2.5,97.5]).tolist(),
        helped=sum(second(r)<first(r) for r in rows),harmed=sum(second(r)>first(r) for r in rows),
        tied=sum(second(r)==first(r) for r in rows))

def validate(folder):
    rows=[json.loads(l) for l in (folder/'predictions.jsonl').read_text().splitlines()]
    summary=json.loads((folder/'summary.json').read_text());env=json.loads((folder/'environment.json').read_text())
    manifest=json.loads((folder/'manifest.json').read_text())
    assert summary['status']=='complete' and summary['expected']==summary['completed']==len(rows)==1101
    assert len({r['id'] for r in rows})==len(rows)
    assert sha(folder/'predictions.jsonl')==summary['predictions_sha256']
    assert sha(folder/'manifest.json')==env['manifest_sha256']
    assert env['repair_policy_sha256']==sha(ROOT/'src/publication_study/loop_repair.py')==POLICY
    for key in PINNED:
        selected=[r for r in manifest if r['cohort'].startswith(key)]
        assert hashlib.sha256(json.dumps(selected,sort_keys=True,ensure_ascii=False).encode()).hexdigest()==PINNED[key]
    for r,m in zip(rows,manifest):
        assert all(r[k]==v for k,v in m.items())
        if env['config']['mode'].startswith('alignment_zero'):
            assert env['alignment_policy_sha256']==sha(ROOT/'src/publication_study/alignment_repair.py')
            state=r['alignment_status']['status']
            assert state!='alignment_error', 'A run with alignment exceptions cannot establish the intended hypothesis'
            if state in ('not_triggered','abstain_context_length'):
                repaired,removed=r['prediction'],0
                if state=='not_triggered':assert suppress_adjacent_word_loops(r['prediction'])[1]==0
            else:
                assert all(np.isfinite(w[k]) for w in r['alignment'] for k in ('start','end','probability'))
                repaired,meta=repair_zero_duration(r['prediction'],r['alignment']);removed=meta['removed']
                assert meta==r['alignment_status']
        else:
            repaired,removed=suppress_adjacent_word_loops(r['prediction'])
        assert (repaired,removed)==(r['repaired_prediction'],r['removed_words'])
        assert counts(r['reference'],r['prediction'])==r['raw_counts']
        assert counts(r['reference'],repaired)==r['repair_counts']
    cohorts={}
    for name in sorted({r['cohort'] for r in rows}):
        rr=[r for r in rows if r['cohort']==name]
        result=paired(rr,lambda r:r['raw_counts']['word_errors'],lambda r:r['repair_counts']['word_errors'])
        result['altered']=sum(r['removed_words']>0 for r in rr)
        result['audio_seconds']=sum(r['duration_s'] for r in rr)
        result['decode_seconds']=sum(r['decode_seconds'] for r in rr)
        logged=summary['cohorts'][name]
        assert (result['clips'],result['reference_words'],result['first_word_errors'],result['second_word_errors'],result['altered'],result['helped'],result['harmed'])==(
            logged['clips'],logged['ref_words'],logged['raw_errors'],logged['repair_errors'],logged['altered'],logged['helped'],logged['harmed'])
        if env['config']['mode'].startswith('alignment_zero'):
            result['alignment_seconds']=sum(r['alignment_seconds'] for r in rr)
            result['alignment_statuses']={state:sum(r['alignment_status']['status']==state for r in rr) for state in sorted({r['alignment_status']['status'] for r in rr})}
        cohorts[name]=result
    return rows,dict(environment=env,predictions_sha256=sha(folder/'predictions.jsonl'),cohorts=cohorts)


def main():
    p=argparse.ArgumentParser();p.add_argument('--log',action='append',required=True);p.add_argument('--out',required=True);args=p.parse_args()
    outputs={};all_rows={}
    for path in args.log:
        log=Path(path);folder=Path.home()/'.local/share/openresearch/recovered-cloud'/log.stem
        if not folder.exists():recover(log,folder)
        rows,result=validate(folder);mode=result['environment']['config']['mode']
        assert mode not in outputs
        result['run_id']=log.stem;result['log_sha256']=sha(log);result['private_artifacts']=str(folder)
        outputs[mode]=result;all_rows[mode]=rows
    for mode,rows in all_rows.items():
        if not mode.startswith('alignment_zero'):continue
        cfg=outputs[mode]['environment']['config'];source_mode=cfg['source_mode']
        if source_mode in all_rows:
            assert outputs[source_mode]['predictions_sha256']==cfg['source_predictions_sha256']
            source={r['id']:r for r in all_rows[source_mode]}
            for r in rows:
                assert (r['prediction'],r['raw_counts'],r['audio_sha256'])==(source[r['id']]['prediction'],source[r['id']]['raw_counts'],source[r['id']]['audio_sha256'])
    comparisons={}
    if {'hf_beam5','official_default'}<=set(all_rows):
        b={r['id']:r for r in all_rows['hf_beam5']};o={r['id']:r for r in all_rows['official_default']}
        assert set(b)==set(o)
        for key in b:
            assert (b[key]['reference'],b[key]['audio_sha256'],b[key]['sp_id'])==(o[key]['reference'],o[key]['audio_sha256'],o[key]['sp_id'])
        for cohort in outputs['hf_beam5']['cohorts']:
            rr=[r for r in b.values() if r['cohort']==cohort]
            comparisons[cohort]=paired(rr,lambda r:r['raw_counts']['word_errors'],lambda r:o[r['id']]['raw_counts']['word_errors'])
    cascades={}
    if 'hf_beam5' in all_rows:
        base=all_rows['hf_beam5']
        for alternative in ('official_default','hf_turbo_beam5'):
            if alternative not in all_rows:continue
            alt={r['id']:r for r in all_rows[alternative]}
            assert set(alt)=={r['id'] for r in base}
            for r in base:
                assert (r['reference'],r['audio_sha256'],r['sp_id'])==(alt[r['id']]['reference'],alt[r['id']]['audio_sha256'],alt[r['id']]['sp_id'])
            cohorts={}
            for cohort in outputs['hf_beam5']['cohorts']:
                rr=[r for r in base if r['cohort']==cohort]
                def cascade_errors(r):
                    return alt[r['id']]['raw_counts']['word_errors'] if r['removed_words']>0 else r['raw_counts']['word_errors']
                entry=paired(rr,lambda r:r['raw_counts']['word_errors'],cascade_errors)
                entry['triggered']=sum(r['removed_words']>0 for r in rr)
                if 'official_default' in all_rows:
                    official={r['id']:r for r in all_rows['official_default']}
                    entry['versus_official_small']=paired(rr,lambda r:official[r['id']]['raw_counts']['word_errors'],cascade_errors)
                cohorts[cohort]=entry
            cascades[alternative]=cohorts
    # Published length control, applied to raw hypotheses without reference access.
    duration_controls={}
    for mode,rows in all_rows.items():
        if mode.startswith('alignment_zero'):continue
        cohorts={}
        for cohort in outputs[mode]['cohorts']:
            rr=[r for r in rows if r['cohort']==cohort]
            capped={r['id']:duration_character_cap(r['prediction'],r['duration_s']) for r in rr}
            errors={r['id']:counts(r['reference'],capped[r['id']])['word_errors'] for r in rr}
            entry=paired(rr,lambda r:r['raw_counts']['word_errors'],lambda r:errors[r['id']])
            entry['altered']=sum(capped[r['id']]!=r['prediction'] for r in rr)
            cohorts[cohort]=entry
        duration_controls[mode]=cohorts
    factorial={}
    for first,second in [('hf_turbo_beam5','official_turbo'),('official_default','official_turbo')]:
        if not {first,second}<=set(all_rows):continue
        a=all_rows[first];b={r['id']:r for r in all_rows[second]}
        assert {r['id'] for r in a}==set(b)
        assert outputs[first]['environment']['manifest_sha256']==outputs[second]['environment']['manifest_sha256']
        factorial[f'{second}_minus_{first}']={cohort:paired([r for r in a if r['cohort']==cohort],
            lambda r:r['raw_counts']['word_errors'],lambda r:b[r['id']]['raw_counts']['word_errors'])
            for cohort in outputs[first]['cohorts']}
    result=dict(status='independently_validated',runs=outputs,official_minus_hf_beam5=comparisons,
        prespecified_loop_trigger_cascades=cascades,
        published_duration_character_cap=duration_controls,additional_decoder_comparisons=factorial,
        bootstrap=dict(seed=20261013,resamples=4000,cluster='speaker'),
        limits=['Fresh Lahaja clips share speakers with the previous heldout sample; audio-disjoint only.',
                'The repetition stress is constructed from labelled human keyword recordings, not natural repeated conversation.',
                'The two decoder stacks change multiple settings and implementations; this is a recipe comparison.',
                'Different GPU types preclude interpreting raw timing ratios as decoder speedups.',
                'This Hindi result does not establish an Indian English improvement.',
                'Frozen repair was selected on prior Lahaja development; no tuning on these outcomes.',
                'No publication novelty or acceptance is established by a WER reduction.'],
        validation_script_sha256=sha(__file__))
    out=Path(args.out);out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n')
    print(json.dumps({'out':str(out),'cohorts':{m:r['cohorts'] for m,r in outputs.items()}},indent=2))

if __name__=='__main__':main()
