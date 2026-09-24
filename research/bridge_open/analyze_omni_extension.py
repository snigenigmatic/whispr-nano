"""Accept exact-input Omni CTC runs and compare complete language cohorts."""
import argparse
import gzip
import hashlib
import json
import tarfile
from pathlib import Path

from research.bridge_open.audit_scripts import distribution
from research.bridge_open.scoring import score, normalize_v1
from research.bridge_open.analyze_fleurs import _aggregate, cluster_bootstrap_difference

ROOT = Path(__file__).resolve().parents[2]
ART = ROOT / 'research/bridge_open/artifacts'
MANIFEST = ROOT / 'data/bridge_open/fleurs_test_v1/corpus-v2.jsonl'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def lines(path):
    return [json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()]


def validate(run):
    run = Path(run)
    cfg, env, summary = [read(run / (n + '.json')) for n in ('config', 'environment', 'summary')]
    receipt = read(run / 'orx_source_receipt.json')
    assert receipt['provider_status'] == 'done' and read(run / 'remote_validation.json')['valid']
    archive = run / receipt['portable_archive']
    assert sha(archive) == receipt['portable_archive_sha256']
    assert hashlib.sha256(gzip.decompress(archive.read_bytes())).hexdigest() == receipt['sourceDigest']
    with tarfile.open(archive, 'r:gz') as tar:
        members = {m.name.removeprefix('./'): m for m in tar if m.isfile()}
        def archived(name):
            return tar.extractfile(members[name]).read()
        committed = json.loads(archived('research/bridge_open/run_config.json'))
        assert {k: v for k, v in cfg.items() if k not in ('output_dir', 'manifest_path')} == {
            k: v for k, v in committed.items() if k not in ('output_dir', 'manifest_path')}
        for name, h in env['code_sha256'].items():
            assert hashlib.sha256(archived('research/bridge_open/omni_extension/' + name)).hexdigest() == h
    assert cfg['model_id'] == 'facebook/omniASR-CTC-1B'
    assert cfg['model_revision'] == '8c22e3ffdaa4aab6431b128b84b991a7d9c2515c'
    assert cfg['full_waveform'] and cfg['batch_size'] == 1 and cfg['termination_semantics'] == 'ctc_no_eos'
    assert cfg['source_manifest_sha256'] == sha(MANIFEST) == '967d8dca02f9f729c1ed7548dce7283cf9dfc62623815a7c131f71acb13ce2c7'
    assert env['config_sha256'] == sha(run / 'config.json')
    runtime = env['runtime_evidence']
    assert runtime['parameter_dtypes'] == ['torch.bfloat16'] and runtime['model_training'] is False
    assert runtime['official_omnilingual_asr_commit'] == '81f51e224ce9e74b02cc2a3eaf21b2d91d743455'
    assert env['asset_sha256'] == {
        'omniASR-CTC-1B.pt': 'e8564fa59dab7caedbcdb54ab7fb9bd6c96989f4d19add2ad81ddd969716952c',
        'omniASR_tokenizer.model': 'b954cc166b0c9e0271b953fa226fa27ca706a25b7029e84579fe2c60a2b451fe'}
    sources = {r['conversation_id']: r for r in lines(MANIFEST)}
    records = lines(run / 'predictions.jsonl')
    ids = [r['conversation_id'] for r in records]
    assert len(ids) == len(set(ids)) == len(cfg['conversation_ids']) == summary['scored'] == summary['attempted']
    assert set(ids) == set(cfg['conversation_ids']) and summary['runtime_failure_count'] == 0
    for r in records:
        source = sources[r['conversation_id']]
        assert r['status'] in ('success', 'empty_output') and r['language'] == source['language']
        assert r['model_id'] == cfg['model_id'] and r['model_revision'] == cfg['model_revision']
        assert r['termination_semantics'] == 'ctc_no_eos'
        assert r['api_token_agreement'] == (cfg['inference_mode'] == 'acceptance')
        for key, path in [('audio_sha256', 'audio_path'), ('reference_sha256', 'reference_path')]:
            assert r[key] == source[key] == sha(source[path])
        assert len(r['chunks']) == 1
        c = r['chunks'][0]
        assert c['start_seconds'] == 0 and abs(c['end_seconds'] - source['duration_seconds']) < 1e-4
        assert c['eos_present'] is None and c['ctc_raw_token_ids']
        raw = c['ctc_raw_token_ids']
        assert all(type(t) is int and t >= 0 for t in raw)
        assert c['ctc_collapsed_token_ids'] == [t for i, t in enumerate(raw) if not i or raw[i-1] != t]
        assert c['text_raw'].strip() == r['text_raw'] and r['text_for_scoring'] == normalize_v1(r['text_raw'])
        reference = read(source['reference_path'])['reference_text']
        assert r['metrics'] == score(reference, r['text_raw'])
        r['source_sentence_id'] = source['source_sentence_id']
        rd, hd = distribution(reference), distribution(r['text_raw'])
        r['script_mismatch'] = bool(rd['dominant_script'] and hd['dominant_script'] and rd['dominant_script'] != hd['dominant_script'])
        r['unknown_script'] = hd['dominant_script'] is None
    return records, dict(records=len(records), api_agreements=sum(r['api_token_agreement'] for r in records),
                         wall_seconds=sum(r['wall_seconds'] for r in records),
                         audio_seconds=sum(r['duration_seconds'] for r in records),
                         model_load_seconds=env['model_load_seconds'], runtime=runtime)


def compare(runs, out):
    rows, receipts = [], []
    for run in runs:
        r, audit = validate(run); rows.extend(r); receipts.append(dict(run=str(run), audit=audit))
    ids = [r['conversation_id'] for r in rows]
    assert len(ids) == len(set(ids)) == 2322 and set(ids) == {r['conversation_id'] for r in lines(MANIFEST)}
    groups = {}; differences = {}
    for lang in ('hi', 'bn', 'as'):
        selected = [r for r in rows if r['language'] == lang]
        groups[lang] = dict(_aggregate(selected), script_mismatches=sum(r['script_mismatch'] for r in selected),
                            unknown_script=sum(r['unknown_script'] for r in selected))
        differences[lang] = {}
        for label, file in [('Whisper Turbo', 'turbo_predictions.jsonl'), ('Whisper Large-v3', 'large-v3_predictions.jsonl')]:
            comparator = [r for r in lines(ART / 'fleurs_whisper_corrected_v3' / file) if r['language'] == lang]
            differences[lang][label] = cluster_bootstrap_difference(selected, comparator)
        if lang == 'hi':
            qwen = lines(ART / 'qwen_native_full_hi_v1/predictions.jsonl')
            sources = {r['conversation_id']: r for r in lines(MANIFEST)}
            for r in qwen: r['source_sentence_id'] = sources[r['conversation_id']]['source_sentence_id']
            differences[lang]['Qwen3-ASR-1.7B'] = cluster_bootstrap_difference(selected, qwen)
    out = Path(out); out.mkdir(parents=True, exist_ok=False)
    result = dict(groups=groups, omni_minus_comparator=differences, receipts=receipts,
                  scope='Declared system comparison on exposed FLEURS official test; no Indian-English/mobile/fairness inference')
    (out / 'comparison.json').write_text(json.dumps(result, indent=2) + '\n')
    (out / 'predictions.jsonl').write_text('\n'.join(json.dumps(r, ensure_ascii=False) for r in rows) + '\n')
    report = ['# OmniASR CTC complete-cohort comparison', '',
              'All 2,322 FLEURS test recordings use the pinned native CTC pipeline, one full waveform per recording, BF16, evaluation mode, and the same primary scorer as the other systems. The acceptance cohort verified official API/token-decoder equality. Full coverage and independent rescoring passed.', '',
              '| Language | n | Word errors / reference words | WER | CER | Script mismatches |',
              '|---|---:|---:|---:|---:|---:|']
    for lang, g in groups.items():
        report.append(f"| {lang} | {g['utterances']} | {g['word_errors']} / {g['ref_words']} | {g['micro_wer']:.2%} | {g['micro_cer']:.2%} | {g['script_mismatches']} |")
    report += ['', 'Paired differences are Omni minus comparator in percentage points; 95% sentence-cluster bootstrap intervals (2,000 draws, seed 12):']
    for lang, comparisons in differences.items():
        for name, d in comparisons.items():
            report.append(f"- {lang}, {name}: {100*d['difference']:+.2f} [{100*d['ci_95'][0]:+.2f}, {100*d['ci_95'][1]:+.2f}].")
    report += ['', 'These are comparisons of declared configurations, with different precision, decoding and model training. Training-data overlap is not independently ruled out. No human transcript adjudication or speaker-population inference is implied. Script matching does not establish correct meaning. Provider charges remain unverified.']
    (out / 'RESULTS.md').write_text('\n'.join(report) + '\n')
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('output'); p.add_argument('runs', nargs='+'); a = p.parse_args()
    print(json.dumps(compare(a.runs, a.output), indent=2))
