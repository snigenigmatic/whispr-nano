"""Descriptive failure tails on frozen, already-exposed benchmark predictions."""
import hashlib
import json
from pathlib import Path

import numpy as np

from research.bridge_open.audit_scripts import distribution
from research.bridge_open.scoring import normalize_v1

ROOT = Path(__file__).resolve().parents[2]
ART = ROOT / 'research/bridge_open/artifacts'
SOURCES = {
    'Whisper Turbo': ART / 'fleurs_whisper_corrected_v3/turbo_predictions.jsonl',
    'Whisper Large-v3': ART / 'fleurs_whisper_corrected_v3/large-v3_predictions.jsonl',
    'Qwen3-ASR-1.7B': ART / 'qwen_native_full_hi_v1/predictions.jsonl',
    'OmniASR CTC-1B': ART / 'fleurs_omni_comparison_v1/predictions.jsonl',
}


def read(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def errors(row):
    return sum(row['metrics'][k] for k in ('substitutions', 'deletions', 'insertions'))


def main():
    out = ART / 'failure_tails_v1'
    out.mkdir(exist_ok=False)
    all_rows = {name: read(path) for name, path in SOURCES.items()}
    if sum(len(rows) for rows in all_rows.values()) != 7384:
        raise ValueError('Unexpected primary prediction coverage')
    manifest = {r['conversation_id']: r for r in read(ROOT / 'data/bridge_open/fleurs_test_v1/corpus-v2.jsonl')}
    groups = []
    for name, rows in all_rows.items():
        expected = {cid for cid, r in manifest.items() if name != 'Qwen3-ASR-1.7B' or r['language'] == 'hi'}
        ids = [r['conversation_id'] for r in rows]
        if len(ids) != len(set(ids)) or set(ids) != expected:
            raise ValueError(f'Incomplete or duplicate cohort for {name}')
        for r in rows:
            source = manifest[r['conversation_id']]
            if any(r[key] != source[key] for key in ('audio_sha256', 'reference_sha256', 'language')):
                raise ValueError('Source identity mismatch')
            if r['metrics']['ref_words'] <= 0 or not np.isclose(r['metrics']['wer'], errors(r)/r['metrics']['ref_words']):
                raise ValueError('Metric-count mismatch')
        for lang in sorted({r['language'] for r in rows}):
            selected = [r for r in rows if r['language'] == lang]
            wer = [r['metrics']['wer'] for r in selected]
            groups.append(dict(model=name, language=lang, n=len(selected),
                micro_wer=sum(errors(r) for r in selected)/sum(r['metrics']['ref_words'] for r in selected),
                macro_wer=float(np.mean(wer)), median_utterance_wer=float(np.median(wer)),
                p95_utterance_wer=float(np.quantile(wer, .95)),
                wer_ge_1_count=sum(errors(r) >= r['metrics']['ref_words'] for r in selected),
                zero_word_error_count=sum(errors(r) == 0 for r in selected),
                normalized_empty_count=sum(not normalize_v1(r['text_raw']) for r in selected)))
    pairs = []
    omni = {r['conversation_id']: r for r in all_rows['OmniASR CTC-1B']}
    for name, rows in all_rows.items():
        if name == 'OmniASR CTC-1B':
            continue
        for r in rows:
            o = omni[r['conversation_id']]
            if o['metrics']['ref_words'] != r['metrics']['ref_words']:
                raise ValueError('Reference denominators differ between paired systems')
            delta = errors(o)-errors(r)
            pairs.append(dict(conversation_id=r['conversation_id'], language=r['language'], comparator=name,
                omni_minus_comparator_word_errors=delta, ref_words=r['metrics']['ref_words'],
                outcome='omni_fewer_errors' if delta < 0 else 'omni_more_errors' if delta > 0 else 'equal_error_count',
                omni_script=distribution(o['text_raw'])['dominant_script'],
                comparator_script=distribution(r['text_raw'])['dominant_script']))
    comparisons = []
    for name in SOURCES:
        if name == 'OmniASR CTC-1B':
            continue
        for lang in ('hi', 'bn', 'as'):
            selected = [r for r in pairs if r['comparator'] == name and r['language'] == lang]
            if selected:
                comparisons.append(dict(comparator=name, language=lang, n=len(selected),
                    fewer=sum(r['outcome'] == 'omni_fewer_errors' for r in selected),
                    equal=sum(r['outcome'] == 'equal_error_count' for r in selected),
                    more=sum(r['outcome'] == 'omni_more_errors' for r in selected)))
    result = dict(scope='Post-hoc descriptive analysis on exposed FLEURS; no causal or demographic claim',
        definitions={'wer_ge_1': 'word edit count >= reference word count; not a semantic failure label',
            'p95': 'NumPy linear 0.95 quantile of recording-level WER; no confidence interval',
            'pair_outcome': 'Compare word-edit counts on exactly the same reference; equal count does not mean equal transcript'},
        groups=groups, paired_counts=comparisons,
        source_hashes={name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in SOURCES.items()})
    (out/'analysis.json').write_text(json.dumps(result, indent=2)+'\n')
    (out/'paired_counts.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in pairs))
    report = ['# Descriptive failure tails on the frozen FLEURS benchmark', '',
        'This post-hoc analysis reuses accepted predictions; no new inference or model selection was performed. The threshold WER ≥ 100% means word edits meet or exceed the reference word count. It does not certify a semantic failure. Percentiles describe recordings in this corpus, not a population confidence interval.', '',
        '| System | Language | n | Micro WER | Median WER | 95th percentile WER | WER ≥ 100% (count) | Zero word errors (count) |',
        '|---|---|---:|---:|---:|---:|---:|---:|']
    for g in groups:
        report.append(f"| {g['model']} | {g['language']} | {g['n']} | {g['micro_wer']:.2%} | {g['median_utterance_wer']:.2%} | {g['p95_utterance_wer']:.2%} | {g['wer_ge_1_count']} | {g['zero_word_error_count']} |")
    report += ['', '## Paired recording counts', '',
        '| Omni versus | Language | Fewer word edits | Equal word edits | More word edits |', '|---|---|---:|---:|---:|']
    for c in comparisons:
        report.append(f"| {c['comparator']} | {c['language']} | {c['fewer']} | {c['equal']} | {c['more']} |")
    report += ['', 'These system configurations differ in model training, precision and decoding. Pair counts weight recordings equally; micro WER weights reference words. Neither establishes demographic fairness, human semantic accuracy, nor a trained improvement. Input/source hashes and total cohort coverage were rechecked. The blinded review sample remains fixed independently of this additional analysis.', '']
    (out/'RESULTS.md').write_text('\n'.join(report))
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
