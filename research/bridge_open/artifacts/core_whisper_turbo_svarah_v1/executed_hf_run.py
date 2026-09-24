"""Runnable, evidence-first Whisper core-condition worker.

This worker stages a frozen condition, verifies HF/native model identity, and
emits a complete artifact bundle on either success or failure. Short clips use
the reviewed HF batch route; longer clips go to native Whisper intact.
"""
from __future__ import annotations

import copy
import hashlib
import importlib.metadata
import inspect
import json
import platform
import shutil
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.diagnostic_v1 import hf_key
from research.bridge_open.artifact_transport import emit
from research.bridge_open.scoring import normalize_v1, score

SAMPLE_RATE = 16_000
SHORT_SECONDS = 30
MODEL_CONTRACTS = {
    ("openai/whisper-small", "973afd24965f72e36ca33b3055d56a652f456b4d"): {"native": "small", "n_mels": 80},
    ("openai/whisper-large-v3-turbo", "41f01f3fe87f28c78e2fbf8b568835947dd65ed9"): {"native": "large-v3-turbo", "n_mels": 128},
    ("openai/whisper-large-v3", "06f233fe06e710322aca913c1bc4249a0d71fce1"): {"native": "large-v3", "n_mels": 128},
}


def sha(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def json_dump(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def source_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def reference_text(path: Path) -> str:
    """Read a staged reference only after generation evidence was installed."""
    raw = path.read_text()
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return raw
    if isinstance(parsed, dict):
        for key in ("reference_text", "text", "transcript"):
            if isinstance(parsed.get(key), str):
                return parsed[key]
    if isinstance(parsed, str):
        return parsed
    raise ValueError(f"Unsupported staged reference format: {path}")


def load_selected() -> dict[str, Any]:
    """Resolve the chosen condition against the frozen suite without editing it."""
    selected = json.loads((ROOT / "experiments/selected_condition.json").read_text())
    suite = json.loads((ROOT / "experiments/core_suite.json").read_text())
    legacy = json.loads((ROOT / "experiments/config.json").read_text())
    matching = [item for item in suite["conditions"] if item["condition_id"] == selected.get("condition_id")]
    require(len(matching) == 1, "Selected condition must occur exactly once in core_suite.json")
    condition = matching[0]
    require(condition["expected_count"] == selected.get("expected_count"),
            "Selected condition differs from frozen suite expected_count")
    source = suite["source_manifests"].get(condition["source"])
    require(source is not None and source["path"] == selected.get("source_manifest") and
            source["sha256"] == selected.get("source_manifest_sha256"), "Selected source manifest differs from frozen suite")
    return dict(selected, language=condition["language"], model_id=suite["model"]["model_id"], model_revision=suite["model"]["model_revision"],
                native_model=suite["model"]["native_model"], short_route=suite["fixed_short_route"],
                long_route=suite["fixed_long_route"], max_wall_seconds=condition["max_wall_seconds"],
                native_transcribe=legacy["native_transcribe"],
                core_suite_sha256=sha(ROOT / "experiments/core_suite.json"))


def model_contract(cfg: dict[str, Any]) -> dict[str, Any]:
    contract = MODEL_CONTRACTS.get((cfg.get("model_id"), cfg.get("model_revision")))
    require(contract is not None, "Model/revision is not an accepted Whisper core pin")
    require(cfg.get("native_model") == contract["native"], "Native model differs from accepted HF model pin")
    return contract


def _check_selected(cfg: dict[str, Any], source: Path) -> None:
    required = ("condition_id", "source_manifest", "source_manifest_sha256", "conversation_ids",
                "language", "expected_count", "model_id", "model_revision", "native_model",
                "short_route", "long_route", "native_transcribe")
    missing = [key for key in required if key not in cfg]
    require(not missing, f"Missing selected-condition fields: {missing}")
    contract = model_contract(cfg)
    require(sha(source) == cfg["source_manifest_sha256"], "Frozen source manifest hash changed")
    ids = cfg["conversation_ids"]
    require(isinstance(ids, list) and len(ids) == cfg["expected_count"] == len(set(ids)),
            "Selected IDs must be unique and match expected_count")
    short, long = cfg["short_route"], cfg["long_route"]
    require(short == {"decoding_mode": "batched_shortform", "chunk_seconds": 30, "batch_size": 8,
                      "num_beams": 5, "max_new_tokens": 440, "return_timestamps": False,
                      "dtype": "float32"}, "Short route differs from the accepted float32 beam-5 route")
    expected_long = {"decoding_mode": "native_longform", "native_model": contract["native"], "fp16": False,
                     "temperature": [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
                     "compression_ratio_threshold": 2.4, "logprob_threshold": -1.0,
                     "no_speech_threshold": 0.6, "condition_on_previous_text": True,
                     "word_timestamps": False, "task": "transcribe"}
    require(long == expected_long, "Long route differs from accepted native Whisper settings")
    expected_native = {"verbose": None, "temperature": [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
                       "compression_ratio_threshold": 2.4, "logprob_threshold": -1.0,
                       "no_speech_threshold": 0.6, "condition_on_previous_text": True,
                       "initial_prompt": None, "carry_initial_prompt": False, "word_timestamps": False,
                       "clip_timestamps": "0", "hallucination_silence_threshold": None, "fp16": False,
                       "task": "transcribe"}
    require(cfg["native_transcribe"] == expected_native, "Native defaults differ from the accepted Whisper settings")


def _selected_rows(cfg: dict[str, Any], source: Path) -> list[dict[str, Any]]:
    all_rows = [json.loads(line) for line in source.read_text().splitlines() if line.strip()]
    by_id = {row["conversation_id"]: row for row in all_rows}
    require(len(by_id) == len(all_rows), "Source manifest has duplicate conversation IDs")
    missing = [cid for cid in cfg["conversation_ids"] if cid not in by_id]
    require(not missing, f"Selected IDs absent from source manifest: {missing[:3]}")
    rows = [copy.deepcopy(by_id[cid]) for cid in cfg["conversation_ids"]]
    require(all(row.get("language") == cfg["language"] for row in rows),
            "Selected condition contains an unexpected language")
    return rows


def _runtime_paths(runtime: Path, row: dict[str, Any]) -> tuple[Path, Path]:
    cid = row["conversation_id"]
    return runtime / "wav" / f"{cid}.wav", runtime / "references" / f"{cid}.reference"


def _stage_rows(cfg: dict[str, Any], rows: list[dict[str, Any]], runtime: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Stage portable bytes and preserve an explicit original-to-runtime map."""
    runtime.mkdir(parents=True, exist_ok=False)
    mapping: list[dict[str, Any]] = []
    staged = copy.deepcopy(rows)
    is_fleurs = cfg["source_manifest"].endswith("fleurs_manifest.jsonl")
    for original, row in zip(rows, staged):
        wav, ref = _runtime_paths(runtime, row)
        row["audio_path"], row["reference_path"] = str(wav), str(ref)
        mapping.append({"conversation_id": row["conversation_id"],
                        "original_audio_path": original["audio_path"], "runtime_audio_path": str(wav),
                        "audio_sha256": row["audio_sha256"], "original_reference_path": original["reference_path"],
                        "runtime_reference_path": str(ref), "reference_sha256": row["reference_sha256"],
                        "staging_mode": "fleurs_archive_member" if is_fleurs else "local_byte_copy",
                        "reference_materialization": "after_prediction" if is_fleurs else "before_inference"})
    if is_fleurs:
        from research.bridge_open.stage_fleurs import stage
        # Audio bytes are enough for decoding.  Do not open reference text
        # until a prediction has been installed for that conversation.
        stage(staged, ROOT / "research/bridge_open/frozen_references.json", stage_references=False)
    else:
        for original, row in zip(rows, staged):
            origin_audio, origin_ref = source_path(original["audio_path"]), source_path(original["reference_path"])
            require(origin_audio.is_file() and origin_ref.is_file(), f"Missing source input for {row['conversation_id']}")
            require(sha(origin_audio) == row["audio_sha256"], f"Source audio hash changed for {row['conversation_id']}")
            require(sha(origin_ref) == row["reference_sha256"], f"Source reference hash changed for {row['conversation_id']}")
            Path(row["audio_path"]).parent.mkdir(parents=True, exist_ok=True)
            Path(row["reference_path"]).parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(origin_audio, row["audio_path"])
            shutil.copyfile(origin_ref, row["reference_path"])
    for row in staged:
        require(sha(row["audio_path"]) == row["audio_sha256"], f"Staged audio hash mismatch for {row['conversation_id']}")
        if not is_fleurs:
            require(sha(row["reference_path"]) == row["reference_sha256"], f"Staged reference hash mismatch for {row['conversation_id']}")
    return staged, mapping


def _preflight_audio(rows: list[dict[str, Any]]) -> dict[str, Any]:
    import numpy as np
    import soundfile as sf
    audio: dict[str, Any] = {}
    for row in rows:
        samples, sample_rate = sf.read(row["audio_path"], dtype="float32")
        require(sample_rate == SAMPLE_RATE and samples.ndim == 1 and len(samples) > 0,
                f"Expected nonempty mono 16 kHz audio for {row['conversation_id']}")
        require(np.isfinite(samples).all() and np.any(samples), f"Invalid waveform for {row['conversation_id']}")
        duration = len(samples) / SAMPLE_RATE
        require(abs(duration - float(row["duration_seconds"])) < 0.02,
                f"Duration mismatch for {row['conversation_id']}")
        audio[row["conversation_id"]] = samples
    return audio


def _base_record(row: dict[str, Any], cfg: dict[str, Any], config_hash: str, script_hash: str, audio: Any) -> dict[str, Any]:
    return {"record_type": "conversation_prediction", "schema_version": "2.0",
            "conversation_id": row["conversation_id"], "language": row["language"], "condition_id": cfg["condition_id"],
            "status": "inference_error", "failure": None,
            "model": {"hf_model_id": cfg["model_id"], "hf_model_revision": cfg["model_revision"], "native_model": cfg["native_model"]},
            "evidence": {"input_audio_sha256": row["audio_sha256"], "reference_sha256": row["reference_sha256"],
                         "config_sha256": config_hash, "worker_sha256": script_hash},
            "full_wav_length_samples": int(len(audio)),
            "full_waveform": {"sample_rate": SAMPLE_RATE, "sample_count": int(len(audio)), "duration_seconds": len(audio) / SAMPLE_RATE},
            "started_at": now(), "text_raw": None, "text_for_scoring": None, "metrics": None,
            "eos": {"status": "unknown", "reason": "not decoded"}}


def _score_after_prediction(record: dict[str, Any], row: dict[str, Any]) -> None:
    require(record["text_raw"] is not None, "Cannot score without prediction evidence")
    record["text_for_scoring"] = normalize_v1(record["text_raw"])
    if not record["text_for_scoring"]:
        record["status"] = "empty_output"
    staged_reference = Path(row["reference_path"])
    if not staged_reference.exists() and row.get("source_repo") == "google/fleurs":
        from research.bridge_open.stage_fleurs import stage_reference
        stage_reference(row, ROOT / "research/bridge_open/frozen_references.json")
    require(staged_reference.is_file() and sha(staged_reference) == row["reference_sha256"],
            f"Staged reference hash mismatch for {row['conversation_id']}")
    reference = reference_text(staged_reference)
    record["metrics"] = score(reference, record["text_raw"])
    record["ended_at"] = now()


def _native_short(native: Any, whisper: Any, audio: Any, language: str, max_new_tokens: int, n_mels: int) -> dict[str, Any]:
    import torch
    options = whisper.DecodingOptions(language=language, task="transcribe", fp16=False, beam_size=5,
                                      temperature=0.0, without_timestamps=True, sample_len=max_new_tokens)
    mel = whisper.log_mel_spectrogram(whisper.pad_or_trim(audio), n_mels=n_mels).to(next(native.parameters()).device)
    with torch.inference_mode():
        result = native.decode(mel, options)
    tokens = list(result.tokens)
    return {"text_raw": result.text, "token_ids": tokens, "options": vars(options),
            "eos": {"status": "unknown", "reason": "native decode strips EOS from returned tokens",
                    "bounded_output_cap": max_new_tokens, "cap_observed": "unknown"},
            "avg_logprob": result.avg_logprob, "no_speech_prob": result.no_speech_prob, "compression_ratio": result.compression_ratio}


def _native_long(native: Any, whisper: Any, audio: Any, row: dict[str, Any], native_options: dict[str, Any]) -> dict[str, Any]:
    import torch
    attempts: list[dict[str, Any]] = []
    original_decode = native.decode
    def traced_decode(mel: Any, options: Any) -> Any:
        result = original_decode(mel, options)
        tokens = list(result.tokens)
        attempts.append({"options": vars(options), "token_ids": tokens, "text_raw": result.text,
                         "avg_logprob": result.avg_logprob, "no_speech_prob": result.no_speech_prob,
                         "compression_ratio": result.compression_ratio,
                         "at_or_over_default_cap": len(tokens) >= native.dims.n_text_ctx // 2})
        return result
    native.decode = traced_decode
    options = copy.deepcopy(native_options)
    options["temperature"] = tuple(options["temperature"])
    try:
        with torch.inference_mode():
            result = native.transcribe(audio.copy(), language=row["language"], **options)
    finally:
        native.decode = original_decode
    require(result.get("language") == row["language"], f"Native language mismatch for {row['conversation_id']}")
    return {"text_raw": result["text"], "segments": result.get("segments", []), "decode_attempts": attempts,
            "effective_options": {"language": row["language"], **native_options},
            "eos": {"status": "unknown", "reason": "native transcribe has no stable utterance EOS contract", "attempt_count": len(attempts)},
            "segment_bounds_valid": all(0 <= segment["start"] <= segment["end"] <= len(audio) / SAMPLE_RATE + .03
                                        for segment in result.get("segments", []))}


def _package_versions() -> dict[str, str]:
    names = ("torch", "transformers", "openai-whisper", "numpy", "scipy", "soundfile", "jiwer", "huggingface-hub")
    versions = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "not-installed"
    return versions


def _aggregate(rows: list[dict[str, Any]], records: list[dict[str, Any]]) -> dict[str, Any]:
    scored = [record for record in records if record.get("metrics") is not None]
    word_errors = sum(sum(record["metrics"][key] for key in ("substitutions", "deletions", "insertions")) for record in scored)
    ref_words = sum(record["metrics"]["ref_words"] for record in scored)
    return {"attempted": len(rows), "prediction_records": len(records), "scored": len(scored),
            "runtime_failures": len(records) - len(scored), "micro_wer": word_errors / ref_words if ref_words else None,
            "status_counts": {status: sum(record["status"] == status for record in records)
                              for status in sorted({record["status"] for record in records})}}


def run() -> int:
    import numpy as np
    import torch
    import whisper
    from huggingface_hub import snapshot_download
    from transformers import WhisperForConditionalGeneration, WhisperProcessor
    cfg = load_selected()
    source = ROOT / cfg["source_manifest"]
    _check_selected(cfg, source)
    contract = model_contract(cfg)
    n_mels = contract["n_mels"]
    out = ROOT / "artifacts" / cfg["condition_id"]
    require(not out.exists(), f"Refusing stale artifact directory: {out}")
    out.mkdir(parents=True)
    json_dump(out / "config.json", cfg)
    config_hash, script_hash = sha(out / "config.json"), sha(__file__)
    rows = _selected_rows(cfg, source)
    runtime = ROOT / "runtime_inputs" / cfg["condition_id"]
    require(not runtime.exists(), f"Refusing stale runtime directory: {runtime}")
    staged, mapping = _stage_rows(cfg, rows, runtime)
    runtime_manifest = runtime / "manifest.jsonl"
    runtime_manifest.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in staged))
    json_dump(out / "source_mapping.json", {"source_manifest": str(source), "source_manifest_sha256": sha(source),
                                             "runtime_manifest": str(runtime_manifest), "runtime_manifest_sha256": sha(runtime_manifest), "rows": mapping})
    audio = _preflight_audio(staged)
    short_rows = [row for row in staged if len(audio[row["conversation_id"]]) <= SHORT_SECONDS * SAMPLE_RATE]
    long_rows = [row for row in staged if len(audio[row["conversation_id"]]) > SHORT_SECONDS * SAMPLE_RATE]
    require(len(short_rows) + len(long_rows) == cfg["expected_count"], "Every staged row must have exactly one route")
    require(torch.cuda.is_available(), "Configured CUDA execution unavailable; refuse silent fallback")
    require(torch.__version__.split("+")[0] == "2.8.0", "Worker requires torch 2.8.0")
    torch.set_num_threads(4); torch.manual_seed(12); np.random.seed(12)
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False; torch.backends.cudnn.benchmark = False
    started = time.monotonic()
    def time_guard() -> None:
        if time.monotonic() - started > cfg["max_wall_seconds"]:
            raise TimeoutError(f"Configured wall-time limit of {cfg['max_wall_seconds']} seconds reached")
    snapshot = Path(snapshot_download(cfg["model_id"], revision=cfg["model_revision"], allow_patterns=["*.json", "*.safetensors", "*.txt"], token=False))
    processor = WhisperProcessor.from_pretrained(snapshot, local_files_only=True)
    hf = WhisperForConditionalGeneration.from_pretrained(snapshot, local_files_only=True, torch_dtype=torch.float32, attn_implementation="sdpa").to("cuda").eval()
    hf.generation_config.forced_decoder_ids = None
    baseline = copy.deepcopy(hf.generation_config)
    native = whisper.load_model(cfg["native_model"], device="cuda", download_root=str(ROOT / "model_cache")).float().eval()
    native_checkpoint = ROOT / "model_cache" / f"{cfg['native_model']}.pt"; native_url = whisper._MODELS[cfg["native_model"]]
    expected_checkpoint_sha = native_url.split("/")[-2]
    require(native_checkpoint.is_file() and sha(native_checkpoint) == expected_checkpoint_sha, "Native checkpoint does not match official Whisper URL")
    json_dump(out / "hf_receipt.json", {"model_id": cfg["model_id"], "requested_revision": cfg["model_revision"],
                                         "snapshot_path": str(snapshot),
                                         "files": {path.name: sha(path) for path in sorted(snapshot.iterdir()) if path.is_file()}})
    json_dump(out / "native_receipt.json", {"native_model": cfg["native_model"], "official_download_url": native_url,
                                             "official_url_sha256": expected_checkpoint_sha,
                                             "checkpoint_path": str(native_checkpoint), "checkpoint_sha256": sha(native_checkpoint)})
    require(processor.feature_extractor.feature_size == native.dims.n_mels == n_mels,
            f"Whisper feature width must be {n_mels} mel bins")
    require(processor.feature_extractor.sampling_rate == SAMPLE_RATE, "Unexpected HF feature sample rate")
    hf_state, native_state = hf.model.state_dict(), native.state_dict(); learned = set(dict(native.named_parameters()))
    weight_rows = []; mapped_keys = set()
    for key, tensor in native_state.items():
        if tensor.is_sparse:
            continue
        mapped = hf_key(key); require(mapped in hf_state, f"Native tensor has no HF counterpart: {key} -> {mapped}")
        other = hf_state[mapped]; require(tuple(tensor.shape) == tuple(other.shape), f"Tensor shape mismatch: {key}")
        delta = float((tensor - other).abs().max())
        require(torch.equal(tensor, other) if key in learned else delta <= 1e-5,
                f"Tensor parity failure: {key}, max_abs_delta={delta}")
        mapped_keys.add(mapped); weight_rows.append({"native_key": key, "hf_key": mapped, "learned": key in learned, "max_abs_delta": delta})
    require(mapped_keys == set(hf_state), "HF state contains unmapped tensors")
    feature_rows = []
    for row in staged:
        time_guard()
        samples = audio[row["conversation_id"]]
        hf_features = processor(samples, sampling_rate=SAMPLE_RATE, padding="max_length", max_length=SHORT_SECONDS * SAMPLE_RATE,
                                truncation=True, return_tensors="pt").input_features[0]
        native_features = whisper.log_mel_spectrogram(whisper.pad_or_trim(samples), n_mels=n_mels).cpu()
        require(tuple(hf_features.shape) == tuple(native_features.shape) == (n_mels, 3000), f"Feature shape mismatch for {row['conversation_id']}")
        delta = float((hf_features - native_features).abs().max()); require(delta < 1e-4, f"Feature parity failure for {row['conversation_id']}: {delta}")
        feature_rows.append({"conversation_id": row["conversation_id"], "n_mels": n_mels, "max_abs_delta": delta})
    environment = {"started_at": now(), "platform": platform.platform(), "device": torch.cuda.get_device_name(0), "dtype": "float32",
                   "packages": _package_versions(), "model": {"id": cfg["model_id"], "revision": cfg["model_revision"],
                   "hf_safetensors_sha256": {path.name: sha(path) for path in sorted(snapshot.glob("*.safetensors"))},
                   "native_model": cfg["native_model"], "native_checkpoint_sha256": sha(native_checkpoint), "native_official_download_url": native_url,
                   "native_official_url_sha256": expected_checkpoint_sha}, "code": {"worker_sha256": script_hash,
                   "hf_key_source_sha256": sha(ROOT / "experiments/diagnostic_v1.py"), "native_transcribe_signature": str(inspect.signature(native.transcribe))},
                   "validation": {"weight_tensor_count": len(weight_rows), "feature_count": len(feature_rows), "n_mels": n_mels}}
    json_dump(out / "environment.json", environment); json_dump(out / "weight_validation.json", weight_rows); json_dump(out / "feature_validation.json", feature_rows)
    records = {row["conversation_id"]: _base_record(row, cfg, config_hash, script_hash, audio[row["conversation_id"]]) for row in staged}
    predictions_path = out / "predictions.jsonl"
    predictions_path.touch(exist_ok=False)
    persisted: set[str] = set()
    def persist(row: dict[str, Any]) -> None:
        conversation_id = row["conversation_id"]
        require(conversation_id not in persisted, f"Duplicate persisted prediction: {conversation_id}")
        with predictions_path.open("a") as stream:
            stream.write(json.dumps(records[conversation_id], ensure_ascii=False) + "\n")
        persisted.add(conversation_id)
    pilot_ids = {row["conversation_id"] for row in short_rows[:3]}
    pilot_native: dict[str, dict[str, Any]] = {}
    for row in short_rows[:3]:
        try:
            time_guard()
            pilot_native[row["conversation_id"]] = _native_short(native, whisper, audio[row["conversation_id"]], row["language"], 440, n_mels)
            pilot_native[row["conversation_id"]]["parity_phase"] = "fixed_three_short_files"
        except Exception as exc:
            pilot_native[row["conversation_id"]] = {"status": "error", "parity_phase": "fixed_three_short_files",
                                                       "failure": {"type": type(exc).__name__, "message": str(exc)}}
    for row in short_rows:
        if row["conversation_id"] not in pilot_native:
            records[row["conversation_id"]]["native_short_parity"] = {
                "status": "not_requested", "reason": "parity scope is the fixed three-file pilot"}
    for index in range(0, len(short_rows), 8):
        group = short_rows[index:index + 8]
        try:
            time_guard()
            hf.generation_config = copy.deepcopy(baseline)
            inputs = processor([audio[row["conversation_id"]] for row in group], sampling_rate=SAMPLE_RATE, padding="max_length",
                               max_length=SHORT_SECONDS * SAMPLE_RATE, truncation=False, return_tensors="pt", return_attention_mask=True).to("cuda")
            with torch.inference_mode():
                generated = hf.generate(**inputs, language=cfg["language"], task="transcribe", num_beams=5, do_sample=False,
                                        return_timestamps=False, max_new_tokens=440, return_dict_in_generate=True)
            sequences = generated.sequences.cpu().tolist(); require(len(sequences) == len(group), "HF batch count differs from inputs")
            for row, text, tokens in zip(group, processor.batch_decode(sequences, skip_special_tokens=True), sequences):
                record = records[row["conversation_id"]]; eos = hf.config.eos_token_id in tokens
                record["route"] = "hf_batched_shortform"; record["hf_generation"] = {"batch_index": index // 8, "token_ids": tokens,
                    "options": {"num_beams": 5, "do_sample": False, "return_timestamps": False, "max_new_tokens": 440, "language": cfg["language"], "task": "transcribe"}}
                record["text_raw"] = text; record["eos"] = {"status": "present" if eos else "absent", "token_id": hf.config.eos_token_id,
                                                               "total_sequence_token_count": len(tokens), "bounded_output_cap": 440,
                                                               "cap_observed": "unknown", "missing_eos": not eos}
                record["status"] = "success" if eos else "generation_limit"
                if not eos: record["failure"] = {"type": "generation_truncated", "message": "HF sequence lacked EOS"}
                _score_after_prediction(record, row)
                if row["conversation_id"] in pilot_native:
                    native_result = pilot_native[row["conversation_id"]]
                    if native_result.get("status") != "error":
                        native_result["text_matches_hf"] = native_result["text_raw"] == record["text_raw"]
                    record["native_short_parity"] = native_result
                persist(row)
        except Exception as exc:
            traceback.print_exc()
            for row in group:
                if row["conversation_id"] in persisted:
                    continue
                record = records[row["conversation_id"]]
                record["failure"] = {"type": type(exc).__name__, "message": str(exc)}; record["ended_at"] = now()
                if row["conversation_id"] in pilot_native:
                    record["native_short_parity"] = pilot_native[row["conversation_id"]]
                persist(row)
    for row in long_rows:
        record = records[row["conversation_id"]]; record["route"] = "native_longform_full_waveform"
        try:
            time_guard()
            native_result = _native_long(native, whisper, audio[row["conversation_id"]], row, cfg["native_transcribe"])
            record["native_longform"] = native_result; record["text_raw"] = native_result["text_raw"]; record["eos"] = native_result["eos"]; record["status"] = "success"
            _score_after_prediction(record, row)
            persist(row)
        except Exception as exc:
            traceback.print_exc(); record["failure"] = {"type": type(exc).__name__, "message": str(exc)}; record["ended_at"] = now(); persist(row)
    ordered = [records[row["conversation_id"]] for row in staged]
    require(len(persisted) == len(ordered), "Prediction stream does not cover every configured input")
    validation = {"expected_count": cfg["expected_count"], "prediction_count": len(persisted), "unique_prediction_ids": len(persisted),
                  "missing_prediction_ids": [row["conversation_id"] for row in staged if row["conversation_id"] not in persisted], "weight_gate": "passed", "feature_gate": "passed",
                  "native_short_parity_pilot_count": len(pilot_ids), "native_short_parity_not_requested_count": len(short_rows) - len(pilot_ids),
                  "short_route_count": len(short_rows), "long_route_count": len(long_rows)}
    require(validation["prediction_count"] == validation["expected_count"] == validation["unique_prediction_ids"], "Prediction output does not cover every configured input exactly once")
    json_dump(out / "validation.json", validation)
    summary = {"status": "completed" if all(record["metrics"] is not None for record in ordered) else "completed_with_runtime_failures",
               "purpose": "fixed public-data Whisper core condition", "total_wall_seconds": time.monotonic() - started, **_aggregate(staged, ordered)}
    json_dump(out / "summary.json", summary)
    return 0 if summary["status"] == "completed" else 1


def main() -> int:
    cfg: dict[str, Any] = {"condition_id": "unknown-condition"}
    artifact = ROOT / "artifacts" / f"unknown-condition.launch-failure-{int(time.time())}"
    code = 1
    try:
        cfg = load_selected()
        primary = ROOT / "artifacts" / cfg["condition_id"]
        artifact = primary if not primary.exists() else ROOT / "artifacts" / f"{cfg['condition_id']}.launch-failure-{int(time.time())}"
        code = run()
    except Exception as exc:
        traceback.print_exc(); artifact.mkdir(parents=True, exist_ok=True)
        json_dump(artifact / "failure.json", {"status": "failed", "type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc(), "occurred_at": now()})
        if not (artifact / "config.json").exists(): json_dump(artifact / "config.json", cfg)
    finally:
        if artifact.exists(): emit(artifact)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
