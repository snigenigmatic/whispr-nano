"""Predeclared one-word task diagnostic over existing, validated cloud outputs."""
import argparse
from collections import Counter
import json
from pathlib import Path
import numpy as np
from package_short_hindi_study import validate,normalize,sha


def select_word(hypothesis):
    words=normalize(hypothesis).split()
    if not words:return ''
    frequency=Counter(words)
    # max() retains the first occurrence when its key ties.
    return max(words,key=lambda word:frequency[word])


def analyze(rows):
    sources={r['id']:r for r in rows if r['cohort']=='hindi_short_natural'}
    ids=sorted({r['source_id'] for r in rows if r['cohort']=='hindi_repeat_5'})
    assert len(ids)==102 and len({sources[i]['sp_id'] for i in ids})==102
    base=[sources[i] for i in ids]
    correct=np.array([select_word(r['prediction'])==normalize(r['reference']) for r in base])
    out={'source_clips':102,'source_speakers':102,'baseline_correct':int(correct.sum()),
         'baseline_accuracy_percent':float(correct.mean()*100),
         'baseline_decode_seconds':sum(r['decode_seconds'] for r in base),'copies':{}}
    for n in (5,8):
        repeated={r['source_id']:r for r in rows if r['cohort']==f'hindi_repeat_{n}'}
        alternative=np.array([select_word(repeated[i]['prediction'])==normalize(sources[i]['reference']) for i in ids])
        delta=alternative.astype(int)-correct.astype(int)
        rng=np.random.default_rng(20261013);draws=rng.integers(0,len(ids),size=(4000,len(ids)))
        intervals=np.percentile(100*delta[draws].mean(axis=1),[2.5,97.5]).tolist()
        seconds=sum(repeated[i]['decode_seconds'] for i in ids)
        out['copies'][str(n)]=dict(correct=int(alternative.sum()),accuracy_percent=float(100*alternative.mean()),
            difference_points=float(100*delta.mean()),paired_speaker_bootstrap_95_ci_points=intervals,
            helped=int((delta>0).sum()),harmed=int((delta<0).sum()),tied=int((delta==0).sum()),
            decode_seconds=seconds,measured_decode_time_multiplier=seconds/out['baseline_decode_seconds'])
    return out


def main():
    p=argparse.ArgumentParser();p.add_argument('--evidence',required=True);p.add_argument('--out',required=True);a=p.parse_args()
    evidence=json.loads(Path(a.evidence).read_text());assert evidence['status']=='independently_validated'
    result={}
    for mode,run in evidence['runs'].items():
        folder=Path(run['private_artifacts']);rows,verified=validate(folder)
        assert verified['predictions_sha256']==run['predictions_sha256']
        result[mode]=analyze(rows)
    Path(a.out).write_text(json.dumps(dict(status='independently_validated',conditions=result,
        selector='most frequent normalized token; first occurrence breaks ties; empty remains empty',
        scope='Explicit isolated-word recognition only. Not a general ASR postprocessor.',
        source_evidence_sha256=sha(a.evidence),analysis_script_sha256=sha(__file__)),indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
