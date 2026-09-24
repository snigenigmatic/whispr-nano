"""Bounded, same-language batched Whisper inference for short FLEURS clips.

This runner intentionally keeps the append-only artifact format used by
``validate_run.py``.  It never opens reference text until an utterance has a
completed batch prediction, so references cannot affect decoding.
"""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import time
import traceback
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import torch
import transformers
from huggingface_hub import snapshot_download

from research.bridge_open.generation_output import generation_rows
from research.bridge_open.scoring import normalize_v1, score


SAMPLE_RATE = 16_000
MAX_SECONDS = 30
DEFAULT_BATCH_SIZE = 8
DEFAULT_BEAMS = 5
TERMINAL = {"success", "empty_output", "generation_limit"}


def sha(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def verify(path: str | Path, expected: str) -> None:
    if sha(path) != expected:
        raise ValueError(f"Content hash mismatch: {path}")


def effective_config(config: dict[str, Any]) -> dict[str, Any]:
    """Return the explicitly frozen settings for this fixed batch condition."""
    cfg = dict(config)
    cfg.setdefault("batch_size", DEFAULT_BATCH_SIZE)
    cfg.setdefault("num_beams", DEFAULT_BEAMS)
    cfg.setdefault("return_timestamps", False)
    cfg.setdefault("eos_token_ids", [50257])
    return cfg


def group_by_language(rows: list[dict[str, Any]]) -> list[tuple[str, list[dict[str, Any]]]]:
    """Stable language groups; a generation call never receives mixed hints."""
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[row["language"]].append(row)
    return [(language, groups[language]) for language in sorted(groups)]


def checked_audio(row: dict[str, Any], cfg: dict[str, Any]) -> tuple[np.ndarray, int]:
    """Verify all model input properties before any model is loaded or called."""
    verify(row["audio_path"], row["audio_sha256"])
    # This hashes the reference bytes only; it deliberately does not parse or
    # expose reference text before a prediction exists.
    verify(row["reference_path"], row["reference_sha256"])
    audio, sample_rate = sf.read(row["audio_path"], dtype="float32")
    if sample_rate != SAMPLE_RATE or audio.ndim != 1:
        raise ValueError("Expected mono 16 kHz WAV")
    if not len(audio) or len(audio) > int(cfg["chunk_seconds"] * SAMPLE_RATE):
        raise ValueError("FLEURS batch runner accepts nonempty clips of at most 30 seconds")
    if not np.isfinite(audio).all():
        raise ValueError("Audio has non-finite samples")
    if not np.any(audio):
        raise ValueError("Audio is all zero")
    return audio, sample_rate


def _base_record(row: dict[str, Any], cfg: dict[str, Any]) -> dict[str, Any]:
    return dict(
        record_type="conversation_prediction", schema_version="1.0",
        conversation_id=row["conversation_id"], language=row["language"],
        model_id=cfg["model_id"], model_revision=cfg["model_revision"],
        condition_id=cfg.get("condition_id", "audio_only_batched_fixed_30s_no_overlap"),
        attempt=1, audio_sha256=row["audio_sha256"], reference_sha256=row["reference_sha256"],
        started_at=now(), status="inference_error", chunks=[], failure=None,
    )


def _failure(record: dict[str, Any], exc: Exception, started: float) -> None:
    record["status"] = "inference_error"
    record["failure"] = {"type": type(exc).__name__, "message": str(exc)}
    record["ended_at"] = now()
    record["wall_seconds"] = None
    record["real_time_factor"] = None


def _check_config(cfg: dict[str, Any]) -> None:
    if transformers.__version__ != "4.57.6" or importlib.metadata.version("transformers") != "4.57.6":
        raise RuntimeError("Pilot requires matching imported and installed Transformers 4.57.6")
    if cfg.get("chunk_seconds") != MAX_SECONDS:
        raise ValueError("This fixed condition requires 30-second inputs")
    if not isinstance(cfg["batch_size"], int) or cfg["batch_size"] < 1:
        raise ValueError("batch_size must be a positive integer")
    if cfg["num_beams"] != DEFAULT_BEAMS or cfg["return_timestamps"] is not False:
        raise ValueError("This fixed condition requires deterministic beam-5 decoding without timestamps")
    if cfg.get("decoding_mode") == "native_longform":
        raise ValueError("native_longform is incompatible with short cross-utterance batching")


def _configure_determinism() -> None:
    """Make the declared float32 condition explicit on CUDA-capable hosts."""
    # Match the established float32 reference condition. Deterministic beam
    # search does not imply cross-device bitwise deterministic CUDA kernels.
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = False
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False


def _reset_peak(device: str) -> None:
    if device == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()


def _peak_memory(device: str) -> int | None:
    if device == "cuda":
        torch.cuda.synchronize()
        return int(torch.cuda.max_memory_allocated())
    if device == "mps":
        torch.mps.synchronize()
        return int(torch.mps.driver_allocated_memory())
    return None


def _finish_prediction(record: dict[str, Any], row: dict[str, Any], started: float) -> None:
    """Score only after this record has complete raw generation evidence."""
    record["text_raw"] = " ".join(chunk["text_raw"].strip() for chunk in record["chunks"])
    record["text_for_scoring"] = normalize_v1(record["text_raw"])
    record["status"] = "success" if record["text_for_scoring"] else "empty_output"
    if any(not chunk["eos_present"] for chunk in record["chunks"]):
        record["status"] = "generation_limit"
        record["failure"] = {"type": "generation_truncated", "message": "At least one bounded generation lacked EOS."}
    # The only text read from a reference occurs after the batch result was
    # installed above.  Its hash was verified during preflight.
    reference = json.loads(Path(row["reference_path"]).read_text())["reference_text"]
    record["metrics"] = score(reference, record["text_raw"])
    record["ended_at"] = now()
    record["wall_seconds"] = None
    record["real_time_factor"] = None
    record["timing_semantics"] = "See batch_inference; no per-request latency measured."


def _aggregates(rows: list[dict[str, Any]], results: list[dict[str, Any]]) -> dict[str, Any]:
    aggregates = {}
    for language in sorted({row["language"] for row in rows}):
        group = [r for r in results if r["language"] == language and r["status"] in TERMINAL and "metrics" in r]
        if not group:
            continue
        word_errors = sum(sum(r["metrics"][k] for k in ("substitutions", "deletions", "insertions")) for r in group)
        ref_words = sum(r["metrics"]["ref_words"] for r in group)
        char_errors = sum(r["metrics"]["character_errors"] for r in group)
        ref_chars = sum(r["metrics"]["ref_chars"] for r in group)
        aggregates[language] = dict(
            estimand="all fully processed conversations including empty/capped text; runtime failures reported separately",
            conversation_ids=[r["conversation_id"] for r in group], scored_count=len(group),
            attempted_count=sum(row["language"] == language for row in rows),
            clean_completion_count=sum(r["status"] == "success" for r in group),
            generation_limit_count=sum(r["status"] == "generation_limit" for r in group),
            word_errors=word_errors, ref_words=ref_words, character_errors=char_errors, ref_chars=ref_chars,
            micro_wer=word_errors / ref_words, micro_cer=char_errors / ref_chars,
            mean_conversation_wer=sum(r["metrics"]["wer"] for r in group) / len(group),
            mean_conversation_wil=sum(r["metrics"]["wil"] for r in group) / len(group),
        )
    return aggregates


def run(config_path: str | Path) -> int:
    config_path = Path(config_path)
    cfg = effective_config(json.loads(config_path.read_text()))
    _check_config(cfg)
    manifest_path = Path(cfg["manifest_path"])
    verify(manifest_path, cfg["manifest_sha256"])
    manifest = [json.loads(line) for line in manifest_path.read_text().splitlines()]
    by_id = {row["conversation_id"]: row for row in manifest}
    expected_ids = cfg["conversation_ids"]
    if len(by_id) != len(manifest) or len(expected_ids) != len(set(expected_ids)) or set(expected_ids) - set(by_id):
        raise ValueError("Configured IDs must be unique and present in the manifest")
    rows = [by_id[cid] for cid in expected_ids]
    for row in rows:
        if row.get("reference_qc") != "pass" or row.get("language") not in cfg["supported_languages"]:
            raise ValueError("QC failure or unsupported language in configured cohort")

    # Preflight every source, including finite/nonzero checks, before model cost.
    audio_by_id: dict[str, np.ndarray] = {}
    for row in rows:
        audio, _ = checked_audio(row, cfg)
        audio_by_id[row["conversation_id"]] = audio

    out = Path(cfg["output_dir"])
    out.mkdir(parents=True, exist_ok=False)
    (out / "config.json").write_text(json.dumps(cfg, indent=2) + "\n")
    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    torch.set_num_threads(4)
    _configure_determinism()
    device = cfg["device"]
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Configured CUDA device unavailable; refuse silent fallback")
    if device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("Configured MPS device unavailable; refuse silent fallback")
    started = time.monotonic()
    snapshot = Path(snapshot_download(cfg["model_id"], revision=cfg["model_revision"], allow_patterns=["*.json", "*.safetensors", "*.txt"], token=False))
    weight_hashes = {path.name: sha(path) for path in sorted(snapshot.glob("*.safetensors"))}
    if not weight_hashes:
        raise RuntimeError("No safetensors weights in pinned snapshot")
    auxiliary_hashes = {path.name: sha(path) for path in sorted(snapshot.iterdir()) if path.is_file() and path.suffix in {".json", ".txt"}}
    from transformers import WhisperForConditionalGeneration, WhisperProcessor
    processor = WhisperProcessor.from_pretrained(snapshot, local_files_only=True)
    model = WhisperForConditionalGeneration.from_pretrained(snapshot, local_files_only=True, torch_dtype=torch.float32, attn_implementation="sdpa").to(device).eval()
    model.generation_config.forced_decoder_ids = None
    if model.config.eos_token_id not in cfg["eos_token_ids"]:
        raise RuntimeError("Configured EOS evidence does not include model EOS token")
    evidence = dict(started_at=now(), config_sha256=sha(out / "config.json"), manifest_sha256=sha(manifest_path),
        weight_sha256=weight_hashes, auxiliary_file_sha256=auxiliary_hashes,
        code_sha256={path.name: sha(path) for path in (Path(__file__), Path(__file__).with_name("scoring.py"), Path(__file__).with_name("generation_output.py"))},
        packages={name: importlib.metadata.version(name) for name in ("torch", "transformers", "numpy", "scipy", "soundfile", "jiwer", "huggingface-hub")},
        platform=platform.platform(), device=device, dtype="float32", imported_transformers=transformers.__version__, imported_torch=torch.__version__,
        model_load_seconds=time.monotonic() - started, cloud_cost_usd=None if cfg.get("executor") in {"modal", "hf"} else 0)
    (out / "environment.json").write_text(json.dumps(evidence, indent=2) + "\n")

    records = {row["conversation_id"]: _base_record(row, cfg) for row in rows}
    record_started = {cid: time.monotonic() for cid in records}
    batch_events: list[dict[str, Any]] = []
    predictions_path = out / "predictions.jsonl"
    predictions_path.touch(exist_ok=False)
    def persist(group):
        with predictions_path.open("a") as stream:
            for row in group:
                stream.write(json.dumps(records[row["conversation_id"]], ensure_ascii=False) + "\n")
            stream.flush()
    for language, language_rows in group_by_language(rows):
        for index in range(0, len(language_rows), cfg["batch_size"]):
            group = language_rows[index:index + cfg["batch_size"]]
            if time.monotonic() - started > cfg.get("max_wall_seconds", 3600):
                exc = TimeoutError("Configured run wall-time limit reached between batches")
                for row in group:
                    _failure(records[row["conversation_id"]], exc, record_started[row["conversation_id"]])
                persist(group)
                continue
            _reset_peak(device)
            batch_started = time.monotonic()
            for row in group:
                records[row["conversation_id"]]["started_at"] = now()
            try:
                inputs = processor([audio_by_id[row["conversation_id"]] for row in group], sampling_rate=SAMPLE_RATE,
                    padding="max_length", max_length=SAMPLE_RATE * MAX_SECONDS, truncation=False,
                    return_tensors="pt", return_attention_mask=True)
                with torch.inference_mode():
                    generated = model.generate(input_features=inputs.input_features.to(device), attention_mask=inputs.attention_mask.to(device),
                        language=language, task="transcribe", do_sample=False, num_beams=DEFAULT_BEAMS, return_timestamps=False,
                        return_dict_in_generate=True, max_new_tokens=cfg["max_new_tokens"])
                decoded = generation_rows(generated, model.config.eos_token_id)
                texts = processor.batch_decode([item["token_ids"] for item in decoded], skip_special_tokens=True)
                if len(decoded) != len(group) or len(texts) != len(group):
                    raise RuntimeError("Batch decoder output count differs from input count")
                peak = _peak_memory(device)
                batch_seconds = time.monotonic() - batch_started
                total_audio_seconds = sum(len(audio_by_id[row["conversation_id"]]) / SAMPLE_RATE for row in group)
                event = dict(batch_id=f"{language}-{index // cfg['batch_size']}", language=language, batch_size=len(group),
                    conversation_ids=[row["conversation_id"] for row in group],
                    inference_wall_seconds=batch_seconds, audio_seconds=total_audio_seconds,
                    audio_seconds_per_wall_second=total_audio_seconds / batch_seconds if batch_seconds else None,
                    note="Batch wall time and throughput; not an amortized per-request latency.", peak_cuda_allocated_bytes=peak if device == "cuda" else None,
                    sampled_max_mps_driver_bytes=peak if device == "mps" else None)
                batch_events.append(event)
                print(json.dumps(dict(event="utterance_batch_done", **event)), flush=True)
                for row, text, raw in zip(group, texts, decoded):
                    record = records[row["conversation_id"]]
                    audio = audio_by_id[row["conversation_id"]]
                    record["chunks"] = [dict(start_seconds=0.0, end_seconds=len(audio) / SAMPLE_RATE, text_raw=text, **raw)]
                    record["duration_seconds"] = len(audio) / SAMPLE_RATE
                    record["batch_inference"] = event
                    _finish_prediction(record, row, record_started[row["conversation_id"]])
            except Exception as exc:
                traceback.print_exc()
                for row in group:
                    _failure(records[row["conversation_id"]], exc, record_started[row["conversation_id"]])
            persist(group)

    results = [records[row["conversation_id"]] for row in rows]
    (out / "batch_inference.jsonl").write_text("".join(json.dumps(event) + "\n" for event in batch_events))
    success = [record for record in results if record["status"] == "success"]
    scored = [record for record in results if record["status"] in TERMINAL and "metrics" in record]
    summary = dict(status="completed" if len(success) == len(rows) else "completed_with_quality_flags" if len(scored) == len(rows) else "completed_with_runtime_failures",
        scope="engineering/public-data pilot; not BRIDGE-matched or population evidence", attempted=len(rows), successful=len(success), scored=len(scored),
        conversation_ids=expected_ids, per_language=_aggregates(rows, results),
        results=[{key: record.get(key) for key in ("conversation_id", "language", "status", "metrics", "wall_seconds", "real_time_factor")} for record in results],
        total_wall_seconds=time.monotonic() - started, cloud_cost_usd=None if cfg.get("executor") in {"modal", "hf"} else 0)
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(dict(event="final_summary", **summary)), flush=True)
    return 0 if len(scored) == len(rows) else 1


if __name__ == "__main__":
    raise SystemExit(run(Path(__file__).with_name("run_config.json")))
