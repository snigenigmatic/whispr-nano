"""Batch-1 GPU resource profiling for the fixed 12-item Svarah panel."""
from __future__ import annotations

import hashlib
import json
import math
import platform
import re
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol, Sequence

ROOT = Path(__file__).resolve().parents[3]
STANDARD_ARTIFACT_ROOT = ROOT / "research/bridge_open/artifacts"
FIXED_PANEL_MANIFEST = ROOT / "data/bridge_open/svarah_native_shard02_v2/bundle/manifest.jsonl"
FIXED_PANEL_MANIFEST_SHA256 = "14c08d00b5e585de77ee3af6b486b05980e0cc9e8a31139654437b2ed8082dac"
FIXED_PANEL_IDS = ("svarah_06413", "svarah_06648", "svarah_06257", "svarah_06387", "svarah_06420", "svarah_06341", "svarah_06269", "svarah_06271", "svarah_06332", "svarah_06284", "svarah_06191", "svarah_06522")
SCHEMA_VERSION = "core-resource-profile/v1"


@dataclass(frozen=True)
class PreloadedAudio:
    """One already-decoded, mono 16 kHz input for the fixed resource panel."""
    conversation_id: str
    array: Any
    sample_rate: int
    audio_sha256: str | None = None
    manifest_duration_seconds: float | None = None


class CudaMetrics(Protocol):
    def synchronize(self) -> None: ...
    def memory_allocated(self) -> int: ...
    def memory_reserved(self) -> int: ...
    def reset_peak_memory_stats(self) -> None: ...
    def max_memory_allocated(self) -> int: ...
    def max_memory_reserved(self) -> int: ...
    def environment(self) -> Mapping[str, Any]: ...


class TorchCudaMetrics:
    """Metrics backed by a pre-existing PyTorch CUDA process; never a CPU fallback."""
    def __init__(self, device_index: int = 0) -> None:
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable; resource profiling refuses a CPU fallback")
        self._torch, self._device_index = torch, device_index

    def synchronize(self) -> None: self._torch.cuda.synchronize(self._device_index)
    def memory_allocated(self) -> int: return int(self._torch.cuda.memory_allocated(self._device_index))
    def memory_reserved(self) -> int: return int(self._torch.cuda.memory_reserved(self._device_index))
    def reset_peak_memory_stats(self) -> None: self._torch.cuda.reset_peak_memory_stats(self._device_index)
    def max_memory_allocated(self) -> int: return int(self._torch.cuda.max_memory_allocated(self._device_index))
    def max_memory_reserved(self) -> int: return int(self._torch.cuda.max_memory_reserved(self._device_index))

    def environment(self) -> Mapping[str, Any]:
        p = self._torch.cuda.get_device_properties(self._device_index)
        return {"device_index": self._device_index, "gpu_name": self._torch.cuda.get_device_name(self._device_index),
                "device_properties": {"name": p.name, "total_memory_bytes": int(p.total_memory),
                                      "multi_processor_count": int(p.multi_processor_count),
                                      "compute_capability": f"{p.major}.{p.minor}"},
                "torch_version": self._torch.__version__, "cuda_version": self._torch.version.cuda}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_copy(value: Mapping[str, Any], label: str) -> dict[str, Any]:
    try:
        return json.loads(json.dumps(dict(value), ensure_ascii=False, sort_keys=True))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be JSON serializable") from exc


def read_fixed_panel(manifest_path: str | Path = FIXED_PANEL_MANIFEST) -> tuple[dict[str, Any], ...]:
    """Read and strictly validate the immutable 12-row acceptance manifest."""
    path = Path(manifest_path)
    _require(path.is_file(), f"Fixed panel manifest is missing: {path}")
    _require(_sha256_file(path) == FIXED_PANEL_MANIFEST_SHA256,
             "Fixed panel manifest bytes differ from the accepted 12-row manifest")
    try:
        rows = tuple(json.loads(line) for line in path.read_text().splitlines() if line.strip())
    except json.JSONDecodeError as exc:
        raise ValueError("Fixed panel manifest is not valid JSONL") from exc
    _require(tuple(row.get("conversation_id") for row in rows) == FIXED_PANEL_IDS,
             "Fixed panel IDs or order differ from the accepted 12-row panel")
    for row in rows:
        cid = row["conversation_id"]
        _require(row.get("sample_rate") == 16000 and row.get("source_sample_rate") == 16000,
                 f"{cid}: expected a 16 kHz source")
        _require(row.get("source_channels") == 1 and row.get("channel_policy") == "original_mono",
                 f"{cid}: expected an original mono source")
        _require(row.get("resampling") == "none" and row.get("canonicalization") == "none_original_wav_preserved",
                 f"{cid}: resource panel must preserve the original WAV")
        duration = row.get("duration_seconds")
        _require(isinstance(duration, (int, float)) and 0 < duration < 28,
                 f"{cid}: duration must be positive and below 28 seconds")
    return rows


def preload_fixed_panel(manifest_path: str | Path = FIXED_PANEL_MANIFEST) -> tuple[PreloadedAudio, ...]:
    """Decode WAVs before measurement; file I/O in this function is never timed."""
    import soundfile as sf
    bundle = Path(manifest_path).resolve().parent
    loaded = []
    for row in read_fixed_panel(manifest_path):
        # The manifest retains the original source-relative name while the
        # verified acceptance bundle stores the byte-identical WAV under wav/.
        path = bundle / "wav" / Path(row["audio_path"]).name
        _require(path.is_file() and _sha256_file(path) == row["audio_sha256"],
                 f"{row['conversation_id']}: WAV is missing or hash differs from manifest")
        array, sr = sf.read(path, dtype="float32")
        loaded.append(PreloadedAudio(row["conversation_id"], array, sr, row["audio_sha256"], float(row["duration_seconds"])))
    _validate_inputs(loaded)
    return tuple(loaded)


def _validate_inputs(inputs: Sequence[PreloadedAudio]) -> None:
    _require(tuple(item.conversation_id for item in inputs) == FIXED_PANEL_IDS,
             "Inputs must contain every fixed-panel row once in the accepted manifest order")
    for item in inputs:
        _require(item.sample_rate == 16000, f"{item.conversation_id}: expected sample rate 16000, got {item.sample_rate}")
        array = item.array
        _require(getattr(array, "ndim", None) == 1 and isinstance(getattr(array, "size", None), int) and array.size > 0,
                 f"{item.conversation_id}: expected a nonempty one-dimensional mono array")
        try:
            import numpy as np
            finite = bool(array.dtype.kind in "fiu" and np.isfinite(array).all())
        except (AttributeError, TypeError, ValueError):
            finite = False
        _require(finite, f"{item.conversation_id}: audio contains non-finite or nonnumeric values")
        duration = float(array.size) / item.sample_rate
        if item.manifest_duration_seconds is not None:
            _require(math.isclose(duration, item.manifest_duration_seconds, rel_tol=0.0, abs_tol=1.0 / item.sample_rate),
                     f"{item.conversation_id}: preloaded duration differs from manifest")


def _validate_metadata(model_residency: Mapping[str, Any], pipeline_configuration: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    model, pipeline = _json_copy(model_residency, "model_residency"), _json_copy(pipeline_configuration, "pipeline_configuration")
    for key in ("model_id", "model_hashes", "parameter_count", "declared_precision", "declared_search"):
        _require(key in model, f"model_residency is missing required field {key!r}")
    _require(isinstance(model["model_id"], str) and model["model_id"], "model_residency.model_id must be nonempty")
    _require(isinstance(model["model_hashes"], dict) and model["model_hashes"], "model_residency.model_hashes must be nonempty")
    _require(isinstance(model["parameter_count"], int) and model["parameter_count"] > 0, "model_residency.parameter_count must be positive")
    _require(isinstance(model["declared_precision"], str) and model["declared_precision"], "model_residency.declared_precision must be nonempty")
    _require(isinstance(model["declared_search"], dict), "model_residency.declared_search must be a mapping")
    _require(model.get("resident_before_measurement") is True, "model_residency.resident_before_measurement must be true")
    _require(pipeline.get("batch_size") == 1, "Resource profile condition requires batch_size=1")
    return model, pipeline


def _failure(exc: BaseException) -> dict[str, str]: return {"type": type(exc).__name__, "message": str(exc)}


def _text(response: Any) -> tuple[str | None, str | None, dict[str, str] | None]:
    if not isinstance(response, str):
        return None, None, {"type": "TypeError", "message": "transcribe_one must return str"}
    return response, hashlib.sha256(response.encode("utf-8")).hexdigest(), None


def _warmup(inputs: Sequence[PreloadedAudio], transcribe_one: Callable[[Any, int], str], cuda: CudaMetrics) -> list[dict[str, Any]]:
    records = []
    for ordinal, item in enumerate(inputs):
        text = digest = None
        failure = None
        try:
            text, digest, failure = _text(transcribe_one(item.array, item.sample_rate))
        except Exception as exc:
            failure = _failure(exc)
        try:
            cuda.synchronize()
        except Exception as exc:
            failure = failure or _failure(exc)
        records.append({"phase": "warmup", "ordinal": ordinal, "conversation_id": item.conversation_id,
                        "duration_seconds": float(item.array.size) / item.sample_rate,
                        "status": "success" if failure is None else "failed", "text": text,
                        "text_sha256": digest, "failure": failure})
    return records


def _timed_call(item: PreloadedAudio, pass_index: int, ordinal: int, transcribe_one: Callable[[Any, int], str], cuda: CudaMetrics, clock: Callable[[], float]) -> dict[str, Any]:
    """Exactly one callable invocation; synchronization brackets its timed scope."""
    cuda.synchronize()
    resident = {"allocated_bytes": cuda.memory_allocated(), "reserved_bytes": cuda.memory_reserved()}
    cuda.reset_peak_memory_stats()
    started = clock()
    text = digest = None
    failure = None
    try:
        text, digest, failure = _text(transcribe_one(item.array, item.sample_rate))
    except Exception as exc:
        failure = _failure(exc)
    finally:
        cuda.synchronize()
        elapsed = clock() - started
    _require(elapsed >= 0, "Clock moved backwards during timed inference")
    duration = float(item.array.size) / item.sample_rate
    return {"phase": "timed", "pass_index": pass_index, "ordinal": ordinal, "conversation_id": item.conversation_id,
            "audio_sha256": item.audio_sha256, "duration_seconds": duration, "status": "success" if failure is None else "failed",
            "text": text, "text_sha256": digest, "failure": failure, "end_to_end_inference_seconds": elapsed,
            "real_time_factor": elapsed / duration, "resident_baseline_memory_bytes": resident,
            "peak_cuda_allocated_bytes": cuda.max_memory_allocated(), "peak_cuda_reserved_bytes": cuda.max_memory_reserved()}


def measure_resource_profile(inputs: Sequence[PreloadedAudio], transcribe_one: Callable[[Any, int], str], model_residency: Mapping[str, Any], pipeline_configuration: Mapping[str, Any], *, cuda: CudaMetrics | None = None, clock: Callable[[], float] = time.perf_counter) -> dict[str, Any]:
    """Run one full warmup panel and three fixed-order timed batch-1 passes."""
    _validate_inputs(inputs)
    model, pipeline = _validate_metadata(model_residency, pipeline_configuration)
    _require(callable(transcribe_one) and callable(clock), "transcribe_one and clock must be callable")
    metrics = cuda if cuda is not None else TorchCudaMetrics()
    environment = _json_copy(metrics.environment(), "cuda.environment")
    for key in ("gpu_name", "device_properties", "torch_version", "cuda_version"):
        _require(key in environment, f"cuda.environment is missing required field {key!r}")
    warmup = _warmup(inputs, transcribe_one, metrics)
    timed = [_timed_call(item, pass_index, ordinal, transcribe_one, metrics, clock)
             for pass_index in range(1, 4) for ordinal, item in enumerate(inputs)]
    all_calls = warmup + timed
    return {"schema_version": SCHEMA_VERSION, "created_at": datetime.now(timezone.utc).isoformat(),
            "study": {"panel_name": "svarah_native_shard02_v2_fixed_12", "panel_manifest_sha256": FIXED_PANEL_MANIFEST_SHA256,
                      "conversation_ids": list(FIXED_PANEL_IDS), "warmup_passes": 1, "timed_complete_passes": 3,
                      "condition": {"batch_size": 1, "fixed_order": True, "audio_loaded_before_timing": True},
                      "clock_scope": "per-call end-to-end inference: callable feature preparation, transfer, decoding, and text postprocessing; CUDA synchronized immediately before and after the callable; excludes model load, setup, audio file I/O, reference loading, and scoring",
                      "comparability_note": "This batch-1 resource condition is separate from the primary full-quality Qwen batch-8 condition. It is not a direct full-WER/latency Pareto comparison without matched decoding and parity qualification."},
            "model_residency": model, "pipeline_configuration": pipeline,
            "environment": {**environment, "platform": platform.platform(), "python_version": platform.python_version()},
            "warmup_calls": warmup, "timed_calls": timed,
            "summary": {"expected_warmup_calls": 12, "observed_warmup_calls": len(warmup), "expected_timed_calls": 36,
                        "observed_timed_calls": len(timed), "failed_calls_including_warmup": sum(x["status"] == "failed" for x in all_calls),
                        "timed_failed_calls": sum(x["status"] == "failed" for x in timed)}}


def export_standard_artifact(result: Mapping[str, Any], run_name: str, artifact_root: str | Path = STANDARD_ARTIFACT_ROOT) -> Path:
    """Create ``<artifact_root>/<run_name>/resource_profile.json`` without overwrite."""
    _require(re.fullmatch(r"[a-z0-9][a-z0-9_-]{0,79}", run_name) is not None, "run_name must use lowercase letters, digits, underscores, or hyphens")
    payload = _json_copy(result, "result")
    _require(payload.get("schema_version") == SCHEMA_VERSION, "result has an unsupported schema version")
    out = Path(artifact_root) / run_name
    _require(not out.exists(), f"Artifact directory already exists: {out}")
    out.mkdir(parents=True)
    path = out / "resource_profile.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    return path
