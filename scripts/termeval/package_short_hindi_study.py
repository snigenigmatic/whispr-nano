"""Validate the frozen short-Hindi ablation; silence has no WER denominator."""
from __future__ import annotations
import argparse
from collections import Counter
import json
from pathlib import Path
import numpy as np
from package_cloud_decoder_study import (
    ROOT, POLICY, recover, sha, normalize, counts, paired, suppress_adjacent_word_loops,
    duration_character_cap,
)

MANIFEST='ab7da06da199001f450d4958a25cffda2a5b452579f4261b07a08bac82db000a'
EXPECTED={'hindi_short_natural':414,'hindi_repeat_5':102,'hindi_repeat_8':102,'silence_control':36}


def describe(rows, field):
    edits=[counts(r['reference'],r[field]) for r in rows]
    n=sum(c['ref_words'] for c in edits);e=sum(c['word_errors'] for c in edits)
    empty=sum(not normalize(r[field]) for r in rows)
    exact=sum(normalize(r[field])==normalize(r['reference']) for r in rows)
    return dict(clips=len(rows),reference_words=n,word_errors=e,wer_percent=100*e/n if n else None,
        exact=exact,exact_percent=100*exact/len(rows),empty=empty,emitted=len(rows)-empty,
        substitutions=sum(c['substitutions'] for c in edits),
        deletions=sum(c['deletions'] for c in edits),insertions=sum(c['insertions'] for c in edits))


def validate(folder):
    rows=[json.loads(l) for l in (folder/'predictions.jsonl').read_text().splitlines()]
    env=json.loads((folder/'environment.json').read_text());s=json.loads((folder/'summary.json').read_text())
    ma=json.loads((folder/'manifest.json').read_text())
    assert s['status']=='complete' and s['expected']==s['completed']==len(rows)==654
    assert sha(folder/'manifest.json')==env['manifest_sha256']==MANIFEST
    assert sha(folder/'predictions.jsonl')==s['predictions_sha256']
    assert env['repair_policy_sha256']==POLICY==sha(ROOT/'src/publication_study/loop_repair.py')
    assert env['dataset_protocol']=='short_hindi_v1'
    assert env['dataset_revision']=='d4ffd2ecbdd933e37c917ddcf620eef159ceb3a7'
    assert env['silence_revision']=='8126be4a4b83afefeef9394ff1312b635a499d82'
    assert Counter(r['cohort'] for r in rows)==EXPECTED
    assert len({r['id'] for r in rows})==654
    sources={r['id']:r for r in rows if r['cohort']=='hindi_short_natural'}
    assert len({r['sp_id'] for r in sources.values()})==102
    silence=[r for r in rows if r['cohort']=='silence_control']
    assert len({r['audio_sha256'] for r in silence})==36
    assert sum(len(r['equivalent_source_ids']) for r in silence)==42
    for r,m in zip(rows,ma):
        assert all(r[k]==v for k,v in m.items())
        assert 0<r['duration_s']<=25 and np.isfinite(r['decode_seconds']) and r['decode_seconds']>=0
        repaired,removed=suppress_adjacent_word_loops(r['prediction'])
        assert (repaired,removed)==(r['repaired_prediction'],r['removed_words'])
        assert counts(r['reference'],r['prediction'])==r['raw_counts']
        assert counts(r['reference'],repaired)==r['repair_counts']
        if r['cohort']=='hindi_short_natural':
            assert r['duration_s']<2 and len(normalize(r['reference']).split())==1
        elif r['cohort'].startswith('hindi_repeat_'):
            src=sources[r['source_id']];n=r['n_repeats']
            assert normalize(r['reference'])==' '.join([normalize(src['reference'])]*n)
            assert r['sp_id']==src['sp_id'] and abs(r['duration_s']-(n*src['duration_s']+.15*(n-1)))<1e-6
        else:assert not normalize(r['reference'])
        r['duration_capped_prediction']=duration_character_cap(r['prediction'],r['duration_s'])
    cohorts={}
    for name in EXPECTED:
        rr=[r for r in rows if r['cohort']==name]
        raw=describe(rr,'prediction');repair=describe(rr,'repaired_prediction');cap=describe(rr,'duration_capped_prediction')
        log=s['cohorts'][name]
        assert (raw['clips'],raw['reference_words'],raw['word_errors'],repair['word_errors'],raw['exact'],raw['empty'])==(
            log['clips'],log['ref_words'],log['raw_errors'],log['repair_errors'],log['raw_exact'],log['raw_empty'])
        assert (log['raw_wer'] is None)==(raw['wer_percent'] is None)
        cohorts[name]=dict(raw=raw,text_loop_repair=repair,duration_cap=cap)
        if raw['reference_words']:
            cohorts[name]['text_loop_repair_effect']=paired(rr,lambda r:r['raw_counts']['word_errors'],lambda r:r['repair_counts']['word_errors'])
            cap_errors={r['id']:counts(r['reference'],r['duration_capped_prediction'])['word_errors'] for r in rr}
            cohorts[name]['duration_cap_effect']=paired(rr,lambda r:r['raw_counts']['word_errors'],lambda r:cap_errors[r['id']])
    return rows,dict(environment=env,predictions_sha256=sha(folder/'predictions.jsonl'),cohorts=cohorts)


def main():
    p=argparse.ArgumentParser();p.add_argument('--log',action='append',required=True)
    p.add_argument('--long-study',required=True);p.add_argument('--out',required=True);args=p.parse_args()
    outputs={};all_rows={}
    for file in args.log:
        log=Path(file);folder=Path.home()/'.local/share/openresearch/recovered-cloud'/log.stem
        if not folder.exists():recover(log,folder)
        rows,result=validate(folder);mode=result['environment']['config']['mode']
        assert mode not in outputs
        result.update(run_id=log.stem,log_sha256=sha(log),private_artifacts=str(folder))
        outputs[mode]=result;all_rows[mode]=rows
    # Audit recording disjointness against the complete preceding long-clip run.
    previous=json.loads(Path(args.long_study).read_text())
    previous_folder=Path(previous['runs']['official_default']['private_artifacts'])
    previous_manifest=json.loads((previous_folder/'manifest.json').read_text())
    previous_ids={r['id'] for r in previous_manifest if r['cohort'].startswith('lahaja')}
    for rows in all_rows.values():assert not previous_ids & {r['id'] for r in rows}
    comparison={}
    if {'official_default','official_no_speech_skip'}<=set(outputs):
        a=all_rows['official_default'];b={r['id']:r for r in all_rows['official_no_speech_skip']}
        ca=dict(outputs['official_default']['environment']['decoder']);cb=dict(outputs['official_no_speech_skip']['environment']['decoder'])
        assert ca.pop('no_speech_threshold')==.6 and cb.pop('no_speech_threshold') is None and ca==cb
        assert outputs['official_default']['environment']['gpu']==outputs['official_no_speech_skip']['environment']['gpu']
        assert outputs['official_default']['environment']['packages']==outputs['official_no_speech_skip']['environment']['packages']
        for r in a:assert (r['audio_sha256'],r['reference'],r['sp_id'])==(b[r['id']]['audio_sha256'],b[r['id']]['reference'],b[r['id']]['sp_id'])
        for name in EXPECTED:
            rr=[r for r in a if r['cohort']==name]
            if name=='silence_control':
                comparison[name]=dict(default=describe(rr,'prediction'),skip_disabled=describe([b[r['id']] for r in rr],'prediction'),
                    interpretation='36 constructed/calibration inputs; descriptive counts only, not a population false-positive rate.')
            else:
                comparison[name]=paired(rr,lambda r:r['raw_counts']['word_errors'],lambda r:b[r['id']]['raw_counts']['word_errors'])
            comparison[name]['changed_outputs']=sum(r['prediction']!=b[r['id']]['prediction'] for r in rr)
            comparison[name]['recovered_from_empty']=sum(not normalize(r['prediction']) and bool(normalize(b[r['id']]['prediction'])) for r in rr)
    result=dict(status='independently_validated',runs=outputs,no_speech_skip_ablation=comparison,
        recording_overlap_with_long_study=0,bootstrap=dict(seed=20261013,resamples=4000,cluster='speaker'),
        limits=['One-word clips selected using released transcript and duration metadata; no speaker-independent confirmation beyond the earlier development exclusion.',
                'Constructed repetitions reuse one source recording per speaker; not natural conversational repetition.',
                'Silence controls are artificial/calibration floors explicitly cropped at 25 seconds and byte-deduplicated; empty-reference WER is undefined.',
                'This ablation was motivated after inspecting prior long-clip and Commands outcomes. Intervals are exploratory, without multiplicity adjustment.',
                'Disabling speech skipping is an existing decoder option, not a new ASR method.'],
        validation_script_sha256=sha(__file__))
    Path(args.out).write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n')
    print(json.dumps({'out':args.out,'comparison':comparison,'cohorts':{k:v['cohorts'] for k,v in outputs.items()}},indent=2))


if __name__=='__main__':main()
