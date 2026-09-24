"""Independently rescore the full Hindi cohort and compare declared ASR systems."""
import argparse
import hashlib
import gzip
import importlib.util
import json
from pathlib import Path
from research.bridge_open.analyze_fleurs import cluster_bootstrap_difference, _aggregate
from research.bridge_open.scoring import normalize_v1, score

ROOT = Path(__file__).resolve().parents[2]
ART = ROOT / 'research/bridge_open/artifacts'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def validate_qwen(run, expected_count=418):
    run = Path(run)
    cfg, env, summary = [read(run / f'{n}.json') for n in ['config', 'environment', 'summary']]
    assert read(run / 'independent_validation.json')['valid'] is True
    assert read(run / 'remote_validation.json')['valid'] is True
    assert env['config_sha256'] == sha(run / 'config.json')
    assert env['model_eval'] is True and env['actual_model_dtype'] == 'torch.bfloat16'
    assert env['effective_generation_config']['do_sample'] is False
    assert env['effective_generation_config']['num_beams'] == 1
    assert env['packages']['qwen-asr'] == '0.0.6'
    assert cfg['model_id'] == 'Qwen/Qwen3-ASR-1.7B'
    assert cfg['model_revision'] == '7278e1e70fe206f11671096ffdd38061171dd6e5'
    assert cfg['inference_api'] == 'Qwen3ASRModel.transcribe'
    assert cfg['batch_size'] == 8 and cfg['max_new_tokens'] == 512
    assert cfg['qwen_postprocessing'] == 'official API text primary; pre-parser generation retained as evidence'
    assert summary['scored'] == summary['attempted'] == expected_count
    committed = read(run / 'committed_run_config.json')
    assert {k:v for k,v in cfg.items() if k != 'output_dir'} == {k:v for k,v in committed.items() if k != 'output_dir'}
    for filename, digest in env['code_sha256'].items():
        assert sha(run / ('executed_' + filename)) == digest
    receipt = read(run / 'orx_source_receipt.json')
    assert receipt['provider_status'] == 'done'
    archive=run / receipt['portable_archive']
    assert sha(archive)==receipt['portable_archive_sha256']
    assert hashlib.sha256(gzip.decompress(archive.read_bytes())).hexdigest()==receipt['sourceDigest']
    manifest = ROOT / 'data/bridge_open/fleurs_test_v1/corpus-v2.jsonl'
    assert sha(manifest) == cfg['manifest_sha256'] == '967d8dca02f9f729c1ed7548dce7283cf9dfc62623815a7c131f71acb13ce2c7'
    byid = {r['conversation_id']:r for r in jsonl(manifest) if r['language'] == 'hi'}
    rows = jsonl(run / 'predictions.jsonl')
    actual = {r['conversation_id'] for r in rows}
    assert len(rows) == len(actual) == expected_count and actual == set(cfg['conversation_ids'])
    if expected_count == 418:
        assert actual == set(byid)
    spec = importlib.util.spec_from_file_location('executed_qwen_trace', run / 'executed_qwen_trace.py')
    trace_module = importlib.util.module_from_spec(spec); spec.loader.exec_module(trace_module)
    groups = {}; caps = eos = empty = 0; raw_primary_mismatches=[]
    for r in rows:
        source = byid[r['conversation_id']]
        assert r['status'] in {'success', 'empty_output'}
        assert r['audio_sha256'] == source['audio_sha256'] == sha(source['audio_path'])
        assert r['reference_sha256'] == source['reference_sha256'] == sha(source['reference_path'])
        assert r['metrics'] == score(read(source['reference_path'])['reference_text'], r['text_raw'])
        assert r['text_for_scoring'] == normalize_v1(r['text_raw'])
        assert len(r['chunks']) == 1
        chunk = r['chunks'][0]
        assert chunk['start_seconds'] == 0 and abs(chunk['end_seconds'] - source['duration_seconds']) < 1e-4
        assert not trace_module.validate_trace(chunk['generation_trace'], cfg)
        trace = chunk['generation_trace'][0]
        eos += trace['eos_present']; caps += trace['cap_observed']; empty += not r['text_for_scoring']
        if trace['raw_decoded_text'].strip()!=r['text_raw'].strip():
            raw_primary_mismatches.append(r['conversation_id'])
        r['source_sentence_id'] = source['source_sentence_id']
        assert r['wall_seconds'] is None and r['real_time_factor'] is None
        groups.setdefault(r['inference_group_id'], []).append(r)
    total_wall = 0; total_audio = 0
    for group in groups.values():
        n = len(group)
        assert {r['group_size'] for r in group} == {n}
        assert {r['chunks'][0]['generation_trace'][0]['record_index'] for r in group} == set(range(n))
        assert {r['chunks'][0]['generation_trace'][0]['batch_size'] for r in group} == {n}
        assert len({r['group_wall_seconds'] for r in group}) == 1
        assert abs(sum(r['duration_seconds'] for r in group) - group[0]['group_audio_seconds']) < 1e-4
        total_wall += group[0]['group_wall_seconds']; total_audio += group[0]['group_audio_seconds']
    audit = dict(records=len(rows), groups=len(groups), observed_eos=eos, observed_caps=caps, normalized_empty=empty,
                 raw_primary_text_match_count=len(rows)-len(raw_primary_mismatches),
                 raw_primary_text_mismatch_ids=raw_primary_mismatches,
                 group_wall_seconds=total_wall, audio_seconds=total_audio,
                 audio_seconds_per_group_wall_second=total_audio/total_wall,
                 timing_scope='one run; API batches plus scoring; excludes loading/setup; not request latency',
                 measured_bill_usd=None, source_receipt=receipt)
    return rows, audit


def compare(run, out):
    out=Path(out)
    qwen, audit=validate_qwen(run)
    models={'Qwen3-ASR-1.7B':qwen}
    sources={r['conversation_id']:r for r in jsonl(ROOT/'data/bridge_open/fleurs_test_v1/corpus-v2.jsonl')}
    for label, filename in [('Whisper Turbo', 'turbo_predictions.jsonl'), ('Whisper Large-v3', 'large-v3_predictions.jsonl')]:
        rows=[r for r in jsonl(ART/'fleurs_whisper_corrected_v3'/filename) if r['language']=='hi']
        assert len(rows)==418 and {r['conversation_id'] for r in rows}=={r['conversation_id'] for r in qwen}
        for r in rows:
            assert r['metrics']==score(read(sources[r['conversation_id']]['reference_path'])['reference_text'], r['text_raw'])
        models[label]=rows
    summary={label:dict(_aggregate(rows),native_api_completed_count=sum(r['status']=='native_api_completed' for r in rows),
        native_termination_unknown=sum(r.get('route')=='native_longform' for r in rows)) for label, rows in models.items()}
    differences={label:cluster_bootstrap_difference(qwen, rows) for label,rows in models.items() if label!='Qwen3-ASR-1.7B'}
    out.mkdir(parents=True,exist_ok=False)
    (out/'comparison.json').write_text(json.dumps(dict(summary=summary,qwen_audit=audit,qwen_minus_comparator=differences),indent=2)+'\n')
    lines=['# Full Hindi comparison', '', 'All systems cover the same 418 FLEURS Hindi official-test recordings, pinned audio/references, and primary scorer. These are declared system configurations, not a controlled architecture comparison.', '', '| System | n | Word errors / reference words | WER | CER |', '|---|---:|---:|---:|---:|']
    for label,m in summary.items():
        lines.append(f"| {label} | {m['utterances']} | {m['word_errors']} / {m['ref_words']} | {m['micro_wer']:.2%} | {m['micro_cer']:.2%} |")
    lines += ['', 'Qwen uses the pinned native API, bfloat16, greedy generation, batch size up to eight, explicit Hindi, and full waveforms. Its official output postprocessing is included; raw pre-parser output/tokens remain saved. Whisper uses float32 HF beam-5 short-form inference and native long-form inference for the single >30-second Hindi recording. Precision, decoding and API processing differ.', '', f"Qwen integrity: {audit['records']} records, {audit['observed_eos']} observed EOS terminations, {audit['observed_caps']} observed token caps, {audit['normalized_empty']} empty outputs. Runtime failures are not excluded from a partial table: complete coverage is required.", '', 'Paired WER differences (Qwen minus comparator), sentence-cluster percentile bootstrap, 2,000 draws, seed 12:']
    for label,d in differences.items():
        lines.append(f"- {label}: {100*d['difference']:.2f} percentage points; 95% interval [{100*d['ci_95'][0]:.2f}, {100*d['ci_95'][1]:.2f}], {d['cluster_count']} sentence clusters.")
    lines += ['', 'Intervals describe sentence resampling on this fixed test set. No speaker-population, demographic-fairness, Indian-English, noisy-conversation, mobile-device, or BRIDGE-equivalent conclusion follows. These post-result engineering checks are not an untouched method-development evaluation. No human semantic transcript review was completed.', '', 'Measured billing is unavailable. The shared HF guard records conservative reservations, not an invoice.']
    lines += ['', f"Raw pre-parser decoded text exactly matches the scored API transcript on {audit['raw_primary_text_match_count']}/{audit['records']} recordings (ignoring outer whitespace)."]
    (out/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    (out/'analysis_receipt.json').write_text(json.dumps(dict(analysis_sha256=sha(__file__),scorer_sha256=sha(ROOT/'research/bridge_open/scoring.py'),prediction_sha256=sha(Path(run)/'predictions.jsonl')),indent=2)+'\n')
    return summary


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('run');p.add_argument('output');a=p.parse_args()
    print(json.dumps(compare(a.run,a.output),indent=2))
