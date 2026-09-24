"""Validate and compare post-result diagnostic arms on identical examples."""
import argparse
import hashlib
import json
from pathlib import Path

from research.bridge_open.scoring import score


def analyze(out, source):
    out, source = Path(out), Path(source)
    assert not (out/'failure.json').exists(), 'Diagnostic run recorded a failure'
    cohort = json.loads((out/'cohort.json').read_text())
    rows = {r['conversation_id']: r for r in cohort}
    predictions = [json.loads(line) for line in (out/'predictions.jsonl').read_text().splitlines()]
    short = {k for k, r in rows.items() if r['duration_seconds'] <= 30}
    long = set(rows)-short
    expected = {
        'hf_original_batch': short, 'hf_original_fixed30': long,
        'hf_single': short, 'hf_pipeline': short,
        'native_decode_matched': short, 'native_transcribe_default': set(rows),
    }
    by_arm = {arm: {} for arm in expected}
    for p in predictions:
        arm, ident = p['arm'], p['conversation_id']
        assert ident not in by_arm[arm], 'Duplicate prediction'
        row = rows[ident]
        reference = source/row['staged_reference']
        assert hashlib.sha256(reference.read_bytes()).hexdigest() == row['reference_sha256'] == p['reference_sha256']
        assert p['audio_sha256'] == row['audio_sha256']
        reference_text = json.loads(reference.read_text())['reference_text']
        assert score(reference_text, p['text']) == p['metrics'], 'Metric mismatch'
        by_arm[arm][ident] = p
    for arm, ids in expected.items():
        assert set(by_arm[arm]) == ids, (arm, len(by_arm[arm]), len(ids))
    def aggregate(group):
        errors = sum(p['metrics'][k] for p in group for k in ['substitutions','deletions','insertions'])
        words = sum(p['metrics']['ref_words'] for p in group)
        chars = sum(p['metrics']['ref_chars'] for p in group)
        return dict(n=len(group), word_errors=errors, ref_words=words, wer=errors/words,
                    cer=sum(p['metrics']['character_errors'] for p in group)/chars)
    short_scores = {}
    for arm, records in by_arm.items():
        if arm == 'hf_original_fixed30': continue
        short_scores[arm] = {lang: aggregate([p for k,p in records.items() if k in short and p['language']==lang])
                             for lang in ['hi','bn','as']}
    matches = {}
    original = by_arm['hf_original_batch']
    for arm, records in by_arm.items():
        if arm == 'hf_original_fixed30': continue
        matches[arm] = {lang: sum(records[k]['text'].strip() == original[k]['text'].strip()
                                 for k in short if rows[k]['language']==lang) for lang in ['hi','bn','as']}
    result = dict(artifact_and_metric_checks_passed=True, full_study_performance_claims_cleared=False,
                  purpose='Post-result diagnostic, not held-out performance',
                  records=len(predictions), original_batch_reproduced=sum(p['original_text_equal'] for p in original.values()),
                  short_n=len(short), short_scores=short_scores, text_matches_to_original_batch=matches,
                  long_examples={k: {arm: by_arm[arm][k]['metrics'] for arm in ['hf_original_fixed30','native_transcribe_default']} for k in sorted(long)})
    (out/'independent_analysis.json').write_text(json.dumps(result, indent=2)+'\n')
    return result


if __name__ == '__main__':
    p=argparse.ArgumentParser();p.add_argument('out');p.add_argument('source')
    args=p.parse_args();print(json.dumps(analyze(args.out,args.source),indent=2))
