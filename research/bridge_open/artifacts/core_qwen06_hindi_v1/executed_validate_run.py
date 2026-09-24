"""Validate a completed bridge-open inference artifact directory.

The validator is intentionally independent of the inference runtime.  It uses
only the frozen manifest, JSONL evidence, hashlib, and the pilot scorer, so a
successful report is useful even when the model environment is gone.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import soundfile as sf
from pathlib import Path
from typing import Any

try:  # Works both as ``python -m research.bridge_open.validate_run`` and a file.
    from .scoring import normalize_v1, score
except ImportError:  # pragma: no cover - exercised by direct CLI invocation
    from scoring import normalize_v1, score


EOS = 50257
TERMINAL = {"success", "empty_output", "generation_limit"}
SCORED = TERMINAL
FLOAT_TOL = 1e-5


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def resolve_input_path(value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return (Path(__file__).resolve().parents[2] / path).resolve()


def _same(a: float, b: float) -> bool:
    return math.isclose(float(a), float(b), rel_tol=1e-9, abs_tol=FLOAT_TOL)


def _load(path: Path) -> Any:
    return json.loads(path.read_text())


def validate_native_trace(chunk, duration, cfg=None):
    """Check the native decoder's input windows and seek advancement independently."""
    errors = []
    trace = chunk.get('native_trace', [])
    total = chunk.get('total_feature_frames')
    hop = chunk.get('feature_hop_samples')
    if type(total) is not int or total <= 0 or hop != 160 or abs(total*hop/16000-duration) >= .01001:
        return ['invalid native feature duration']
    if not isinstance(trace, list) or not trace:
        return ['missing native input trace']
    expected_start = 0
    for step in trace:
        if any(type(step.get(k)) is not int for k in ['start_frame','input_end_frame','next_seek_frame','encoder_frames']):
            return ['malformed native trace frame counts']
        start, end, next_seek = (step[k] for k in ['start_frame','input_end_frame','next_seek_frame'])
        if start != expected_start or end != min(start+3000,total) or step['encoder_frames'] != 3000:
            errors.append('native input window mismatch')
        if not start < next_seek <= start+3000 or start >= total:
            errors.append('invalid native seek advancement')
        expected_start = next_seek
        if cfg and cfg.get('compression_ratio_threshold') is not None:
            import zlib
            attempts = step.get('attempts', [])
            temperatures = cfg['temperature']
            if not attempts or len(attempts)>len(temperatures):
                errors.append('missing or excess fallback attempts')
                continue
            for i, attempt in enumerate(attempts):
                tokens=attempt.get('check_token_ids',[])
                vocab_size=attempt.get('vocab_size',0)
                if not tokens or any(type(t) is not int or not 0<=t<vocab_size for t in tokens):
                    errors.append('invalid fallback token evidence')
                    continue
                width=int(math.log2(vocab_size)/8)+1
                encoded=b''.join(t.to_bytes(width,'little') for t in tokens)
                ratio=len(encoded)/len(zlib.compress(encoded))
                needed=ratio>cfg['compression_ratio_threshold']
                if not _same(ratio,attempt.get('compression_ratio',-1)) or needed!=attempt.get('needs_fallback'):
                    errors.append('fallback decision mismatch')
                if attempt.get('temperature')!=temperatures[i] or attempt.get('should_skip') is not False:
                    errors.append('fallback schedule or silence-skip mismatch')
                if i<len(attempts)-1 and not needed:
                    errors.append('unjustified fallback retry')
            if attempts[-1].get('generation_sequence')!=step.get('generation_sequence'):
                errors.append('fallback selected output mismatch')
            if attempts[-1].get('needs_fallback') and len(attempts)!=len(temperatures):
                errors.append('fallback schedule terminated early')
    if expected_start < total:
        errors.append('native decoding did not reach end of audio')
    if [s.get('generation_sequence') for s in trace] != chunk.get('generation_sequences'):
        errors.append('native trace generation sequences mismatch')
    segments = chunk.get('native_segments')
    if not isinstance(segments, list) or not segments:
        errors.append('missing native timestamp segments')
    elif [t for s in segments for t in s.get('token_ids',[])] != chunk.get('token_ids'):
        errors.append('native display tokens differ from segment concatenation')
    return errors


def validate(output_dir: str | Path) -> dict[str, Any]:
    """Return a JSON-serializable validation report for ``output_dir``."""

    out = Path(output_dir)
    errors: list[str] = []

    def fail(message: str) -> None:
        errors.append(message)

    required = ["config.json", "environment.json", "predictions.jsonl", "summary.json"]
    for name in required:
        if not (out / name).is_file():
            fail(f"missing artifact: {name}")
    if errors:
        return {"valid": False, "output_dir": str(out), "errors": errors}

    try:
        cfg = _load(out / "config.json")
        environment = _load(out / "environment.json")
        summary = _load(out / "summary.json")
        predictions = [json.loads(line) for line in (out / "predictions.jsonl").read_text().splitlines()]
    except Exception as exc:
        return {"valid": False, "output_dir": str(out), "errors": [f"invalid JSON artifact: {exc}"]}

    manifest_path = Path(cfg.get("manifest_path", ""))
    if not manifest_path.is_file():
        fail(f"missing manifest: {manifest_path}")
        records: list[dict[str, Any]] = []
    else:
        if cfg.get("manifest_sha256") != sha(manifest_path):
            fail("manifest SHA-256 does not match config")
        try:
            records = [json.loads(line) for line in manifest_path.read_text().splitlines()]
        except Exception as exc:
            records = []
            fail(f"invalid manifest JSONL: {exc}")
    by_id = {r.get("conversation_id"): r for r in records}
    if len(by_id) != len(records):
        fail("manifest contains duplicate or missing conversation IDs")
    expected_ids = cfg.get("conversation_ids", [])
    if len(expected_ids) != len(set(expected_ids)):
        fail("config contains duplicate conversation IDs")
    missing = [cid for cid in expected_ids if cid not in by_id]
    if missing:
        fail(f"config IDs missing from manifest: {missing}")
    if len(predictions) != len(expected_ids):
        fail(f"prediction count {len(predictions)} != expected count {len(expected_ids)}")
    pred_by_id = {p.get("conversation_id"): p for p in predictions}
    if len(pred_by_id) != len(predictions) or set(pred_by_id) != set(expected_ids):
        fail("predictions do not contain exactly the expected unique IDs")

    config_hash = sha(out / "config.json")
    if environment.get("config_sha256") != config_hash:
        fail("environment config SHA-256 does not match config.json")
    if environment.get("manifest_sha256") != cfg.get("manifest_sha256"):
        fail("environment manifest SHA-256 does not match config")

    eos_ids = cfg.get("eos_token_ids", [EOS])
    if not isinstance(eos_ids, list) or not eos_ids or any(not isinstance(e, int) for e in eos_ids):
        fail("invalid EOS token configuration")
        eos_ids = [EOS]
    chunk_seconds = cfg.get("chunk_seconds")
    if not isinstance(chunk_seconds, (int, float)) or chunk_seconds <= 0:
        fail("invalid chunk_seconds in config")

    ctc = cfg.get("termination_semantics") == "ctc_no_eos"
    recomputed: dict[str, dict[str, Any]] = {}
    for cid in expected_ids:
        if cid not in by_id or cid not in pred_by_id:
            continue
        row, prediction = by_id[cid], pred_by_id[cid]
        if prediction.get("language") != row.get("language"):
            fail(f"{cid}: prediction language differs from manifest")
        if row.get("reference_qc", row.get("dataset_reference_qc")) != "pass":
            fail(f"{cid}: manifest dataset reference QC is not pass")
        if prediction.get("model_id") != cfg.get("model_id"):
            fail(f"{cid}: model_id mismatch")
        if prediction.get("model_revision") != cfg.get("model_revision"):
            fail(f"{cid}: model_revision mismatch")
        if prediction.get("audio_sha256") != row.get("audio_sha256"):
            fail(f"{cid}: prediction audio hash differs from manifest")
        if prediction.get("reference_sha256") != row.get("reference_sha256"):
            fail(f"{cid}: prediction reference hash differs from manifest")
        audio_path, reference_path = resolve_input_path(row.get("audio_path", "")), resolve_input_path(row.get("reference_path", ""))
        if not audio_path.is_file() or sha(audio_path) != row.get("audio_sha256"):
            fail(f"{cid}: audio hash verification failed")
        if not reference_path.is_file() or sha(reference_path) != row.get("reference_sha256"):
            fail(f"{cid}: reference hash verification failed")

        if ctc and prediction.get("termination_semantics") != "ctc_no_eos":
            fail(f"{cid}: missing CTC termination declaration")
        if not ctc and prediction.get("termination_semantics") == "ctc_no_eos":
            fail(f"{cid}: CTC declaration disagrees with configured decoder")
        chunks = prediction.get("chunks")
        if not isinstance(chunks, list) or not chunks:
            fail(f"{cid}: missing chunks")
            continue
        # The manifest duration is measured before resampling. The verified WAV
        # can be up to one target sample longer; chunk boundaries use its PCM.
        duration = float(row.get("duration_seconds", -1))
        try:
            audio_info = sf.info(audio_path)
            if audio_info.samplerate != 16000 or audio_info.channels != 1:
                fail(f"{cid}: expected mono 16 kHz audio")
            pcm_duration = audio_info.frames / audio_info.samplerate
            if abs(pcm_duration - duration) > 1 / audio_info.samplerate + 1e-9:
                fail(f"{cid}: manifest duration differs from WAV by over one sample")
            duration = pcm_duration
        except Exception as exc:
            fail(f"{cid}: cannot inspect WAV duration: {exc}")
        native = cfg.get('decoding_mode') == 'native_longform'
        qwen_api = cfg.get('termination_semantics') == 'qwen_transcribe_api'
        expected_count = 1 if native or qwen_api else math.ceil(duration / float(chunk_seconds)) if duration >= 0 else -1
        if len(chunks) != expected_count:
            fail(f"{cid}: chunk count {len(chunks)} != expected {expected_count}")
        for index, chunk in enumerate(chunks):
            expected_start = 0 if native or qwen_api else index * float(chunk_seconds)
            expected_end = duration if native or qwen_api else min(expected_start + float(chunk_seconds), duration)
            if native:
                for error in validate_native_trace(chunk, duration, cfg):
                    fail(f'{cid}: {error}')
            try:
                start, end = float(chunk["start_seconds"]), float(chunk["end_seconds"])
            except (KeyError, TypeError, ValueError):
                fail(f"{cid}: malformed chunk {index} range")
                continue
            if not _same(start, expected_start) or not _same(end, expected_end):
                fail(f"{cid}: non-contiguous chunk range at index {index}")
            if index and not _same(start, float(chunks[index - 1].get("end_seconds", -1))):
                fail(f"{cid}: chunk ranges are not contiguous")
            if end < start:
                fail(f"{cid}: negative chunk duration at index {index}")
            if ctc:
                raw_ids = chunk.get("ctc_raw_token_ids")
                collapsed = chunk.get("ctc_collapsed_token_ids")
                if not isinstance(raw_ids, list) or not raw_ids or any(type(t) is not int or t < 0 for t in raw_ids):
                    fail(f"{cid}: malformed raw CTC IDs at index {index}")
                elif not isinstance(collapsed, list) or any(type(t) is not int for t in collapsed) or collapsed != [t for i,t in enumerate(raw_ids) if i == 0 or raw_ids[i-1] != t]:
                    fail(f"{cid}: CTC collapse mismatch at index {index}")
                if chunk.get("eos_present", "missing") is not None:
                    fail(f"{cid}: CTC cannot claim EOS termination at index {index}")
            elif qwen_api:
                if chunk.get('token_evidence') not in {'unavailable_from_transcribe_api', 'captured_from_generate'}:
                    fail(f"{cid}: Qwen API artifact has invalid token evidence declaration")
                if chunk.get('eos_present') is not None:
                    fail(f"{cid}: Qwen API artifact must not promote EOS to primary termination")
                if chunk.get('token_evidence') == 'captured_from_generate':
                    from research.bridge_open.qwen_trace import validate_trace
                    for error in validate_trace(chunk.get('generation_trace'), cfg):
                        fail(f'{cid}: {error}')
            else:
                tokens = chunk.get("token_ids", [])
                if not isinstance(tokens, list) or any(not isinstance(token, int) for token in tokens):
                    fail(f"{cid}: malformed token IDs at index {index}")
                sequences = chunk.get('generation_sequences', [tokens])
                if not isinstance(sequences, list) or not sequences or any(not isinstance(seq, list) or not seq or any(type(t) is not int for t in seq) for seq in sequences):
                    fail(f"{cid}: malformed raw generation evidence at index {index}")
                    eos = False
                else:
                    eos = all(any(eos_id in sequence for eos_id in eos_ids) for sequence in sequences)
                if chunk.get("eos_present") != eos:
                    fail(f"{cid}: eos_present disagrees with token IDs at index {index}")

        raw = " ".join(str(chunk.get("text_raw", "")).strip() for chunk in chunks)
        if prediction.get("text_raw") != raw:
            fail(f"{cid}: text_raw is not the exact chunk concatenation")
        if prediction.get("text_for_scoring") != normalize_v1(raw):
            fail(f"{cid}: text_for_scoring does not match normalize_v1(text_raw)")
        capped = False if ctc or qwen_api else any(chunk.get('eos_present') is not True for chunk in chunks)
        status = prediction.get("status")
        if status == "generation_limit" and not capped:
            fail(f"{cid}: generation_limit without a capped chunk")
        if status != "generation_limit" and capped:
            fail(f"{cid}: capped chunk requires generation_limit status")
        if status not in TERMINAL:
            fail(f"{cid}: partial/nonterminal status {status!r}; final acceptance refused")
        if status == "empty_output" and normalize_v1(raw):
            fail(f"{cid}: empty_output has nonempty normalized text")
        if status in {"success", "generation_limit"} and not normalize_v1(raw):
            fail(f"{cid}: {status} has empty normalized text")

        try:
            reference = _load(reference_path)["reference_text"]
            expected_metrics = score(reference, raw)
        except Exception as exc:
            fail(f"{cid}: cannot recompute metrics: {exc}")
            continue
        actual_metrics = prediction.get("metrics")
        if not isinstance(actual_metrics, dict):
            fail(f"{cid}: missing metrics")
            continue
        for key, value in expected_metrics.items():
            actual = actual_metrics.get(key)
            if isinstance(value, float):
                if actual is None or not _same(actual, value):
                    fail(f"{cid}: metric {key} differs from recomputed value")
            elif actual != value:
                fail(f"{cid}: metric {key} differs from recomputed value")
        recomputed[cid] = expected_metrics

    # Require every expected conversation to have a complete, scored record.
    if len(recomputed) != len(expected_ids):
        fail("run is partial or contains unscorable conversations; final acceptance refused")
    scored_predictions = [pred_by_id[cid] for cid in expected_ids if cid in pred_by_id and pred_by_id[cid].get("status") in SCORED]
    if summary.get("attempted") != len(expected_ids) or summary.get("scored") != len(scored_predictions):
        fail("summary attempted/scored totals are inconsistent")
    if summary.get("successful") != sum(pred.get("status") == "success" for pred in scored_predictions):
        fail("summary successful total is inconsistent")

    per_language = summary.get("per_language")
    if not isinstance(per_language, dict):
        fail("summary is missing per_language aggregates")
    else:
        for language in sorted({by_id[cid].get("language") for cid in expected_ids if cid in by_id}):
            group_ids = [cid for cid in expected_ids if cid in pred_by_id and by_id[cid].get("language") == language and pred_by_id[cid].get("status") in SCORED]
            aggregate = per_language.get(language)
            if not isinstance(aggregate, dict):
                fail(f"missing aggregate for language {language}")
                continue
            metrics = [recomputed[cid] for cid in group_ids if cid in recomputed]
            word_errors = sum(m["substitutions"] + m["deletions"] + m["insertions"] for m in metrics)
            ref_words = sum(m["ref_words"] for m in metrics)
            ref_chars = sum(m["ref_chars"] for m in metrics)
            char_errors = sum(m["character_errors"] for m in metrics)
            if not metrics or ref_words <= 0 or ref_chars <= 0:
                fail(f"{language}: aggregate has no valid scored references")
                continue
            checks = {"conversation_ids": group_ids, "scored_count": len(metrics), "attempted_count": sum(cid in by_id and by_id[cid].get("language") == language for cid in expected_ids),
                      "word_errors": word_errors, "ref_words": ref_words, "character_errors": char_errors, "ref_chars": ref_chars,
                      "micro_wer": word_errors / ref_words, "micro_cer": char_errors / ref_chars,
                      "mean_conversation_wer": sum(m["wer"] for m in metrics) / len(metrics),
                      "mean_conversation_wil": sum(m["wil"] for m in metrics) / len(metrics)}
            for key, expected in checks.items():
                actual = aggregate.get(key)
                if isinstance(expected, float):
                    if actual is None or not _same(actual, expected):
                        fail(f"{language}: aggregate {key} is inconsistent")
                elif actual != expected:
                    fail(f"{language}: aggregate {key} is inconsistent")

    expected_summary_status = "completed" if all(cid in pred_by_id and pred_by_id[cid].get("status") == "success" for cid in expected_ids) else "completed_with_quality_flags"
    if summary.get("status") != expected_summary_status:
        fail(f"summary status {summary.get('status')!r} is inconsistent with prediction statuses")
    return {"valid": not errors, "output_dir": str(out), "conversation_count": len(expected_ids), "errors": errors}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir")
    args = parser.parse_args(argv)
    report = validate(args.output_dir)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if report["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
