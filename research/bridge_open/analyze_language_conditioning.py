"""Rescore a complete paired language diagnostic without dropping failures."""
import argparse
import gzip
import hashlib
import json
from collections import Counter
from pathlib import Path

from research.bridge_open.audit_scripts import distribution
from research.bridge_open.scoring import score

ROOT = Path(__file__).resolve().parents[2]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def lines(path):
    return [json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()]


def validate(run):
    run = Path(run)
    cfg, env, summary = [read(run / (name + '.json')) for name in ('config', 'environment', 'summary')]
    receipt = read(run / 'orx_source_receipt.json')
    assert receipt['provider_status'] == 'done'
    archive = run / receipt['portable_archive']
    assert sha(archive) == receipt['portable_archive_sha256']
    assert hashlib.sha256(gzip.decompress(archive.read_bytes())).hexdigest() == receipt['sourceDigest']
    assert sha(run / 'executed_audit.py') == env['script_sha256']
    assert sha(run / 'executed_scoring.py') == env['scorer_sha256']
    assert sha(run / 'cohort.json') == cfg['cohort_sha256']
    assert env['model_eval'] and env['actual_dtype'] == 'torch.float32'
    assert env['checkpoint_sha256'] == cfg['models'][0]['expected_native_checkpoint_sha256']
    sources = {r['conversation_id']: r for r in lines(ROOT / 'data/bridge_open/fleurs_test_v1/corpus-v2.jsonl')}
    cohort = read(run / 'cohort.json')
    expected = {(r['conversation_id'], arm) for r in cohort for arm in cfg['arms']}
    records = lines(run / 'predictions.jsonl')
    assert len(cohort) == len({r['conversation_id'] for r in cohort}) == cfg['expected_n']
    assert len(records) == len(expected) == summary['records']
    assert {(r['conversation_id'], r['arm']) for r in records} == expected
    pairs = {}
    for r in records:
        source = sources[r['conversation_id']]
        assert r['status'] == 'native_api_completed' and r['language'] == source['language']
        assert r['model_id'] == cfg['models'][0]['model_id'] and r['model_revision'] == cfg['models'][0]['model_revision']
        for key, path in [('audio_sha256', 'audio_path'), ('reference_sha256', 'reference_path')]:
            assert r[key] == source[key] == sha(source[path])
        assert r['sample_rate'] == 16000 and abs(r['duration_seconds'] - source['duration_seconds']) < 1e-4
        assert abs(r['input_sample_count'] / 16000 - source['duration_seconds']) < 1e-4
        ref = read(source['reference_path'])['reference_text']
        assert r['metrics'] == score(ref, r['text_raw'])
        assert r['reference_script'] == distribution(ref)
        assert r['hypothesis_script'] == distribution(r['text_raw'])
        rd, hd = r['reference_script']['dominant_script'], r['hypothesis_script']['dominant_script']
        assert r['script_mismatch'] == bool(rd and hd and rd != hd)
        probs = r['language_probabilities']
        assert all(0 <= p <= 1 for p in probs.values()) and abs(sum(probs.values()) - 1) < 1e-5
        assert r['independent_detected_language'] == max(probs, key=probs.get)
        language = r['language'] if r['arm'] == 'forced' else None
        assert r['effective_options'] == dict(language=language, **cfg['native_transcribe'])
        output_language = language or r['independent_detected_language']
        assert r['output_language'] == output_language
        assert r['decode_attempts'] and all(a['options']['language'] == output_language for a in r['decode_attempts'])
        bounds = all(0 <= s['start'] <= s['end'] <= r['duration_seconds'] + .03 for s in r['segments'])
        assert r['segment_bounds_valid'] == bounds
        pairs.setdefault(r['conversation_id'], {})[r['arm']] = r
    for pair in pairs.values():
        assert pair['forced']['language_probabilities'] == pair['automatic']['language_probabilities']
    return records, pairs


def aggregate(rows):
    errors = sum(r['metrics'][k] for r in rows for k in ('substitutions', 'deletions', 'insertions'))
    words = sum(r['metrics']['ref_words'] for r in rows)
    return dict(n=len(rows), word_errors=errors, reference_words=words, micro_wer=errors / words,
                script_mismatches=sum(r['script_mismatch'] for r in rows),
                unknown_script=sum(r['hypothesis_script']['dominant_script'] is None for r in rows),
                output_languages=dict(Counter(r['output_language'] for r in rows)),
                timestamp_flags=sum(not r['segment_bounds_valid'] for r in rows))


def analyze(run, out):
    records, pairs = validate(run)
    groups = []
    for scope in ('all_diagnostic', 'sampled_short', 'all_long'):
        for lang in ('hi', 'bn', 'as'):
            selected = [r for r in records if r['language'] == lang and (
                scope == 'all_diagnostic' or (r['duration_seconds'] <= 30) == (scope == 'sampled_short'))]
            arm_stats = {arm: aggregate([r for r in selected if r['arm'] == arm]) for arm in ('forced', 'automatic')}
            groups.append(dict(scope=scope, language=lang, **arm_stats,
                               automatic_minus_forced=arm_stats['automatic']['micro_wer'] - arm_stats['forced']['micro_wer']))
    changes = [dict(conversation_id=cid, language=p['forced']['language'], duration_seconds=p['forced']['duration_seconds'],
                    detected_language=p['automatic']['output_language'],
                    forced_text=p['forced']['text_raw'], automatic_text=p['automatic']['text_raw'],
                    forced_wer=p['forced']['metrics']['wer'], automatic_wer=p['automatic']['metrics']['wer'])
               for cid, p in pairs.items()]
    out = Path(out); out.mkdir(parents=True, exist_ok=False)
    result = dict(status='validated_complete_diagnostic', recordings=len(pairs), groups=groups,
                  interpretation='Exposed-test mechanistic diagnostic; not a population estimate or held-out gain',
                  measured_cost_usd=None, prediction_sha256=sha(Path(run) / 'predictions.jsonl'))
    (out / 'analysis.json').write_text(json.dumps(result, indent=2) + '\n')
    (out / 'paired_transcripts.json').write_text(json.dumps(changes, ensure_ascii=False, indent=2) + '\n')
    report = ['# Native Whisper language-conditioning diagnostic', '',
              'Both arms use the same checkpoint, float32 evaluation state, complete waveform, seed and native decoding settings. Only the supplied language differs. This deliberately exposed diagnostic does not establish an untouched-test improvement.', '',
              '| Cohort | Language | n | Forced WER | Automatic WER | Difference (pp) | Script mismatches forced / auto | Unknown scripts forced / auto |',
              '|---|---|---:|---:|---:|---:|---:|---:|']
    for g in groups:
        f, a = g['forced'], g['automatic']
        report.append(f"| {g['scope']} | {g['language']} | {f['n']} | {f['micro_wer']:.2%} | {a['micro_wer']:.2%} | {100*g['automatic_minus_forced']:+.2f} | {f['script_mismatches']} / {a['script_mismatches']} | {f['unknown_script']} / {a['unknown_script']} |")
    report += ['', 'The 30 short recordings were selected by a fixed hash order, ten per language. All 14 recordings longer than 30 seconds were added separately. These strata must not be pooled to estimate corpus prevalence.', '',
               'Script matching is a Unicode diagnostic, not a judgment of spoken language or transcript meaning. Native decoding strips EOS from its public result; generation-cap counts remain unknown. Any timestamp flags are retained in analysis.json. No human adjudication is implied.']
    (out / 'RESULTS.md').write_text('\n'.join(report) + '\n')
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('run'); p.add_argument('output'); args = p.parse_args()
    print(json.dumps(analyze(args.run, args.output), indent=2))
