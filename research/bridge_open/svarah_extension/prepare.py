"""Prepare a fully audited, exposed Svarah native-audio smoke bundle.

This script is intentionally CPU-only. It refuses incomplete cache downloads,
never invents speaker identities, and writes every source row to the audit even
when that row is not eligible for the 12-item smoke bundle.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import shutil
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np
import pyarrow as pa
import pyarrow.ipc as ipc
import scipy
import soundfile as sf
from scipy import signal
from scipy.io import wavfile


ROOT = Path(__file__).resolve().parents[3]
SOURCE_REVISION = "ebbf7777fe771490696a3f7b007097606fa8c924"
EXPECTED_CACHE_FILES = (
    "svarah-test-00000-of-00003.arrow",
    "svarah-test-00001-of-00003.arrow",
    "svarah-test-00002-of-00003.arrow",
    "dataset_info.json",
)
EXPECTED_SHARDS = tuple(name for name in EXPECTED_CACHE_FILES if name.endswith(".arrow"))
SOURCE_DATASET = "ai4bharat/Svarah"
SMOKE_SEED_PREFIX = "svarah-native-12:"
TARGET_SAMPLE_RATE = 16_000


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def require_complete_cache(cache_dir: Path) -> list[Path]:
    missing = [name for name in EXPECTED_CACHE_FILES if not (cache_dir / name).is_file()]
    if missing:
        raise RuntimeError(f"Svarah cache download is incomplete; missing: {', '.join(missing)}")
    shards = [cache_dir / name for name in EXPECTED_CACHE_FILES if name.endswith(".arrow")]
    if any(path.stat().st_size == 0 for path in shards):
        raise RuntimeError("Svarah cache has an empty Arrow shard")
    return shards


def require_available_shard(cache_dir: Path, shard: str) -> list[Path]:
    if shard not in EXPECTED_SHARDS:
        raise ValueError("--available-shard must name one expected Svarah Arrow shard")
    path = cache_dir / shard
    if not path.is_file() or path.stat().st_size == 0:
        raise RuntimeError(f"Requested available shard is missing or empty: {path}")
    return [path]


def load_prior_metadata(path: Path) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if len(rows) != 6656:
        raise ValueError(f"Expected 6656 prior metadata rows, found {len(rows)}")
    for index, row in enumerate(rows):
        if row.get("dataset_index") != index or row.get("utt_id") != f"svarah_{index:05d}":
            raise ValueError(f"Prior metadata index mismatch at {index}")
        if not isinstance(row.get("audio_path"), str) or not isinstance(row.get("text"), str):
            raise ValueError(f"Prior metadata is missing audio path or reference at {index}")
    return rows


def metadata_pair_index(rows: list[dict]) -> dict[tuple[str, str], dict]:
    indexed: dict[tuple[str, str], dict] = {}
    for row in rows:
        key = (Path(row["audio_path"]).name, row["text"])
        if key in indexed:
            raise ValueError("Prior metadata has a non-unique audio-basename/reference pair")
        indexed[key] = row
    return indexed


def open_arrow(path: Path):
    source = pa.memory_map(str(path), "r")
    try:
        return ipc.open_file(source)
    except pa.ArrowInvalid:
        source.seek(0)
        return ipc.open_stream(source)


def iter_arrow_rows(shards: list[Path]) -> Iterator[tuple[str, int, dict, list[str]]]:
    for shard in shards:
        reader = open_arrow(shard)
        schema = reader.schema
        names = list(schema.names)
        if hasattr(reader, "num_record_batches"):
            batches = (reader.get_batch(batch_number) for batch_number in range(reader.num_record_batches))
        else:
            batches = iter(reader)
        shard_offset = 0
        for batch_number, batch in enumerate(batches):
            for row_number, row in enumerate(batch.to_pylist()):
                yield shard.name, shard_offset + row_number, row, names
            shard_offset += batch.num_rows


def audio_parts(value: object) -> tuple[bytes, str | None]:
    if isinstance(value, dict):
        raw = value.get("bytes")
        path = value.get("path")
    elif isinstance(value, (bytes, bytearray)):
        raw, path = value, None
    else:
        raise ValueError(f"Unsupported Arrow audio value type: {type(value).__name__}")
    if not isinstance(raw, (bytes, bytearray)) or not raw:
        raise ValueError("Arrow audio payload is missing or empty")
    return bytes(raw), str(path) if path is not None else None


@dataclass
class DecodedAudio:
    samples: np.ndarray
    sample_rate: int
    channels: int
    frames: int
    format: str
    subtype: str
    peak: float
    finite: bool
    silent: bool
    clipping: bool
    scipy: dict


def decode_audio(raw: bytes) -> DecodedAudio:
    with sf.SoundFile(io.BytesIO(raw)) as handle:
        sample_rate, channels, frames = handle.samplerate, handle.channels, handle.frames
        audio_format, subtype = handle.format, handle.subtype
        samples = handle.read(dtype="float32", always_2d=True)
    if samples.ndim != 2 or samples.shape[0] != frames or samples.shape[1] != channels:
        raise ValueError("SoundFile frame/channel count disagrees with decoded samples")
    finite = bool(np.isfinite(samples).all())
    peak = float(np.max(np.abs(samples))) if samples.size else float("nan")
    scipy_report: dict = {"applicable": audio_format == "WAV"}
    if audio_format == "WAV":
        try:
            scipy_rate, scipy_samples = wavfile.read(io.BytesIO(raw))
            scipy_frames = int(scipy_samples.shape[0]) if scipy_samples.ndim else int(scipy_samples.size)
            scipy_channels = int(scipy_samples.shape[1]) if scipy_samples.ndim == 2 else 1
            scipy_float = scipy_samples.astype(np.float64)
            if scipy_samples.dtype == np.uint8:
                scipy_float = (scipy_float - 128.0) / 128.0
            elif scipy_samples.dtype == np.int16:
                scipy_float /= 2**15
            elif scipy_samples.dtype == np.int32:
                # scipy exposes 24-bit PCM left-justified in int32 as well.
                scipy_float /= 2**31
            elif scipy_samples.dtype.kind != "f":
                raise ValueError(f"Unsupported SciPy WAV dtype {scipy_samples.dtype}")
            if scipy_float.ndim == 1:
                scipy_float = scipy_float[:, None]
            max_abs_sample_difference = float(np.max(np.abs(scipy_float - samples))) if samples.size else 0.0
            samples_match = bool(np.allclose(scipy_float, samples, rtol=0.0, atol=2.0 / 2**15))
            scipy_report.update({"sample_rate": int(scipy_rate), "frames": scipy_frames,
                                 "channels": scipy_channels, "dtype": str(scipy_samples.dtype),
                                 "matches_soundfile": int(scipy_rate) == sample_rate and
                                 scipy_frames == frames and scipy_channels == channels and samples_match,
                                 "max_abs_sample_difference": max_abs_sample_difference})
            if not scipy_report["matches_soundfile"]:
                raise ValueError("SciPy WAV decode disagrees with SoundFile")
        except Exception as exc:
            scipy_report.update({"error": f"{type(exc).__name__}: {exc}", "matches_soundfile": False})
            raise ValueError(f"SciPy WAV validation failed: {exc}") from exc
    return DecodedAudio(samples=samples, sample_rate=sample_rate, channels=channels,
                        frames=frames, format=audio_format, subtype=subtype, peak=peak,
                        finite=finite, silent=bool(peak == 0.0), clipping=bool(peak >= 0.9999),
                        scipy=scipy_report)


def canonical_wav(raw: bytes, decoded: DecodedAudio) -> tuple[bytes, str]:
    if decoded.format == "WAV" and decoded.sample_rate == TARGET_SAMPLE_RATE and decoded.channels == 1:
        return raw, "none_original_wav_preserved"
    mono = decoded.samples.mean(axis=1, dtype=np.float64).astype(np.float32)
    transforms = ["downmix_mean"] if decoded.channels != 1 else []
    if decoded.sample_rate != TARGET_SAMPLE_RATE:
        divisor = math.gcd(decoded.sample_rate, TARGET_SAMPLE_RATE)
        mono = signal.resample_poly(mono, TARGET_SAMPLE_RATE // divisor, decoded.sample_rate // divisor).astype(np.float32)
        transforms.append("scipy.signal.resample_poly")
    if not np.isfinite(mono).all() or len(mono) == 0:
        raise ValueError("Canonical audio is non-finite or empty")
    transformed_peak = float(np.max(np.abs(mono)))
    if transformed_peak > 1.0:
        raise ValueError(f"Canonical transform produced an out-of-range peak: {transformed_peak}")
    sink = io.BytesIO()
    sf.write(sink, mono, TARGET_SAMPLE_RATE, format="WAV", subtype="FLOAT")
    return sink.getvalue(), "+".join(transforms) or "float_wav_reencode"


def exposure_ids(results_dir: Path) -> tuple[set[str], dict]:
    paths = sorted(results_dir.glob("results_*_svarah_clean.csv"))
    if not paths:
        raise ValueError("No prior Svarah result CSVs found for exposure audit")
    sets: dict[str, set[str]] = {}
    for path in paths:
        with path.open(newline="") as stream:
            rows = list(csv.DictReader(stream))
        values = {str(row.get("utt_id", "")) for row in rows}
        if not values or "" in values or len(values) != len(rows):
            raise ValueError(f"Invalid or duplicate exposure IDs in {path.name}")
        sets[path.name] = values
    union = set().union(*sets.values())
    return union, {"files": {name: len(values) for name, values in sets.items()},
                   "union_count": len(union), "union_sha256": canonical_json(sorted(union))}


def structural_reference(text: object) -> tuple[bool, str | None]:
    if not isinstance(text, str) or not text.strip():
        return False, "empty_reference"
    return True, None


def assert_exact_mapping(record: dict, metadata: dict) -> None:
    """Bind a record to its global metadata row before writing a reference."""
    if record["conversation_id"] != metadata["utt_id"]:
        raise ValueError("conversation ID differs from bound global metadata row")
    if record["dataset_index"] != metadata["dataset_index"]:
        raise ValueError("global metadata index differs from bound record")
    if record["source_audio_basename"] != Path(metadata["audio_path"]).name:
        raise ValueError("source audio basename differs from bound global metadata row")
    if record["source_reference_sha256"] != sha256_bytes(metadata["text"].encode()):
        raise ValueError("reference SHA differs from bound global metadata row")


def assert_partial_mapping_regression(records: list[dict], prior: list[dict], partial: bool) -> dict:
    """Prove a shard-local offset cannot silently select a global reference."""
    if not partial:
        return {"applicable": False}
    first = min(records, key=lambda record: record["source_scan_index"])
    if first["source_scan_index"] != 0 or first["dataset_index"] == 0:
        raise ValueError("partial mapping regression precondition failed")
    wrong = dict(first, dataset_index=0)
    try:
        assert_exact_mapping(wrong, prior[0])
    except ValueError:
        pass
    else:  # pragma: no cover - defensive evidence failure
        raise AssertionError("shard-local index was accepted as a global reference binding")
    return {"applicable": True, "source_scan_index": 0,
            "global_dataset_index": first["dataset_index"], "wrong_global_index_rejected": True}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, default=ROOT / "data/bridge_open/svarah_cache_v1")
    parser.add_argument("--metadata", type=Path, default=ROOT / "paper/data/svarah_metadata_raw.jsonl")
    parser.add_argument("--results-dir", type=Path, default=ROOT / "audit/svarah_fairness_audit")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data/bridge_open/svarah_native_v1")
    parser.add_argument("--audit-dir", type=Path, default=ROOT / "research/bridge_open/artifacts/svarah_input_audit_v1")
    parser.add_argument("--available-shard", choices=EXPECTED_SHARDS,
                        help="Prepare one confirmed Arrow shard only; never labeled full Svarah.")
    parser.add_argument("--all-available", action="store_true",
                        help="Bundle every structurally readable row in --available-shard, including flagged clipping rows.")
    args = parser.parse_args(argv)
    partial = args.available_shard is not None
    if args.all_available and not partial:
        parser.error("--all-available requires --available-shard; it must not imply a complete Svarah cache")
    if partial and (args.output_dir == ROOT / "data/bridge_open/svarah_native_v1" or
                    args.audit_dir == ROOT / "research/bridge_open/artifacts/svarah_input_audit_v1"):
        if args.all_available:
            args.output_dir = ROOT / "data/bridge_open/svarah_native_shard02_full_v1"
            args.audit_dir = ROOT / "research/bridge_open/artifacts/svarah_input_audit_shard02_full_v1"
        else:
            args.output_dir = ROOT / "data/bridge_open/svarah_native_shard02_v2"
            args.audit_dir = ROOT / "research/bridge_open/artifacts/svarah_input_audit_shard02_v2"
    if args.output_dir.exists() or args.audit_dir.exists():
        raise RuntimeError("Refuse to overwrite prior Svarah preparation evidence")
    shards = require_available_shard(args.cache_dir, args.available_shard) if partial else require_complete_cache(args.cache_dir)
    prior = load_prior_metadata(args.metadata)
    prior_pairs = metadata_pair_index(prior)
    exposed, exposure_receipt = exposure_ids(args.results_dir)
    if exposure_receipt["union_count"] != 2794:
        raise ValueError(f"Expected 2794 exposed Svarah IDs, got {exposure_receipt['union_count']}")

    records: list[dict] = []
    source_schema: dict[str, list[str]] = {}
    index = 0
    for shard_name, shard_row, row, names in iter_arrow_rows(shards):
        source_schema.setdefault(shard_name, names)
        if not partial and index >= len(prior):
            raise ValueError("Arrow cache contains more than 6656 rows")
        path_hint = row.get("audio_filepath", {}).get("path") if isinstance(row.get("audio_filepath"), dict) else None
        expected = prior_pairs.get((Path(path_hint or "").name, row.get("text"))) if partial else prior[index]
        if expected is None:
            raise ValueError("Arrow row does not map uniquely to pinned prior metadata")
        flags: list[str] = []
        source_audio_path = None
        raw_hash = None
        decoded = None
        try:
            raw, source_audio_path = audio_parts(row.get("audio_filepath"))
            source_pair = (Path(source_audio_path or "").name, row.get("text"))
            if expected is None or source_pair != (Path(expected["audio_path"]).name, expected["text"]):
                raise ValueError("audio_basename_mismatch_with_prior_metadata")
            raw_hash = sha256_bytes(raw)
            decoded = decode_audio(raw)
            if not decoded.finite:
                flags.append("nonfinite")
            if decoded.silent:
                flags.append("silent")
            if decoded.clipping:
                flags.append("clipping")
            if decoded.peak > 1.0:
                flags.append("peak_out_of_range")
            duration = decoded.frames / decoded.sample_rate
            expected_duration = float(expected["duration"])
            if abs(duration - expected_duration) > max(1 / decoded.sample_rate, 1e-6):
                flags.append("duration_mismatch_with_prior_metadata")
        except Exception as exc:
            flags.append(f"malformed:{type(exc).__name__}")
            duration = None
            expected_duration = float(expected.get("duration", 0.0))
        ref_ok, ref_reason = structural_reference(expected.get("text"))
        if not ref_ok:
            flags.append(ref_reason or "invalid_reference")
        utt_id = expected["utt_id"]
        record = {
            "conversation_id": utt_id,
            "dataset_index": expected["dataset_index"],
            "source_scan_index": index,
            "language": "en",
            "source_dataset": SOURCE_DATASET,
            "source_revision": SOURCE_REVISION,
            "source_shard": shard_name,
            "source_row": shard_row,
            "source_audio_basename": Path(source_audio_path or "").name or None,
            "source_audio_sha256": raw_hash,
            "source_reference_sha256": sha256_bytes(expected["text"].encode()),
            "source_duration_seconds": expected_duration,
            "decoded_duration_seconds": duration,
            "source_sample_rate": decoded.sample_rate if decoded else None,
            "source_channels": decoded.channels if decoded else None,
            "source_frames": decoded.frames if decoded else None,
            "source_format": decoded.format if decoded else None,
            "source_subtype": decoded.subtype if decoded else None,
            "peak": decoded.peak if decoded and math.isfinite(decoded.peak) else None,
            "finite": decoded.finite if decoded else False,
            "silent": decoded.silent if decoded else None,
            "clipping": decoded.clipping if decoded else None,
            "scipy_wav_check": decoded.scipy if decoded else {"applicable": None},
            "exposed_priorly": utt_id in exposed,
            "reference_qc": "pass" if ref_ok and not any(flag.startswith("malformed") for flag in flags) else "fail",
            "reference_qc_scope": "structural_not_human",
            "flags": flags,
        }
        assert_exact_mapping(record, expected)
        records.append(record)
        index += 1
    if not partial and index != 6656:
        raise ValueError(f"Arrow cache row count mismatch: {index}")
    mapping_regression = assert_partial_mapping_regression(records, prior, partial)

    args.audit_dir.mkdir(parents=True)
    (args.audit_dir / "all_records.jsonl").write_text("".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records))
    (args.audit_dir / "exposure_ids.json").write_text(json.dumps({"ids": sorted(exposed), **exposure_receipt}, indent=2) + "\n")
    malformed = sorted({flag for record in records for flag in record["flags"] if flag.startswith("malformed:")})
    if malformed:
        blocked = {
            "status": "blocked_before_bundle_creation",
            "dataset": SOURCE_DATASET,
            "source_revision": SOURCE_REVISION,
            "dataset_coverage": "all_6656" if not partial else f"limited_{args.available_shard}",
            "records": len(records),
            "malformed_flags": malformed,
            "flag_counts": dict(Counter(flag for record in records for flag in record["flags"])),
        }
        (args.audit_dir / "receipt.json").write_text(json.dumps(blocked, indent=2) + "\n")
        print(json.dumps(blocked), flush=True)
        return

    eligible = [record for record in records if not record["flags"] and record["decoded_duration_seconds"] <= 40]
    if args.all_available:
        # This is an availability-selected cohort, not a quality-selected one.  Its
        # structural block above has already rejected malformed source audio; valid
        # clipped rows remain in the cohort and retain their flags in the audit.
        cohort = list(records)
        if {record["conversation_id"] for record in cohort} != {record["conversation_id"] for record in records}:
            raise AssertionError("Full available-shard cohort omitted a source record")
        cohort_selection = "all verified rows in the one available source shard; no duration, clipping, exposure, text, or model-result exclusion"
        cohort_kind = "all_available_shard_rows"
    else:
        if len(eligible) < 12:
            raise ValueError(f"Only {len(eligible)} structurally eligible <=40s rows; need 12")
        longest = max(eligible, key=lambda record: (record["decoded_duration_seconds"], record["conversation_id"]))
        remainder = sorted((record for record in eligible if record["conversation_id"] != longest["conversation_id"]),
                           key=lambda record: hashlib.sha256((SMOKE_SEED_PREFIX + record["conversation_id"]).encode()).hexdigest())
        cohort = [longest, *remainder[:11]]
        if len({record["conversation_id"] for record in cohort}) != 12:
            raise AssertionError("Smoke selection is not unique")
        cohort_selection = "longest structurally eligible <=40s, then 11 ascending SHA256('svarah-native-12:'+utt_id); no transcript, ASR, or exposure selection"
        cohort_kind = "smoke_12"

    args.output_dir.mkdir(parents=True)
    bundle_dir = args.output_dir / "bundle"
    audio_dir, ref_dir = (bundle_dir / name for name in ("wav", "references"))
    for directory in (audio_dir, ref_dir):
        directory.mkdir(parents=True)
    raw_lookup: dict[str, bytes] = {}
    # Re-read only selected rows to avoid retaining all embedded audio in RAM.
    selected_ids = {record["conversation_id"] for record in cohort}
    selected_by_scan_index = {record["source_scan_index"]: record for record in cohort}
    source_index = 0
    for _, _, row, _ in iter_arrow_rows(shards):
        selected = selected_by_scan_index.get(source_index)
        if selected is not None:
            raw, _ = audio_parts(row.get("audio_filepath"))
            raw_lookup[selected["conversation_id"]] = raw
        source_index += 1
    if set(raw_lookup) != selected_ids:
        raise ValueError("Could not recover exactly the selected source audio payloads")

    cohort_rows = []
    for record in cohort:
        uid = record["conversation_id"]
        raw = raw_lookup[uid]
        decoded = decode_audio(raw)
        wav, canonicalization = canonical_wav(raw, decoded)
        audio_path = audio_dir / f"{uid}.wav"
        audio_path.write_bytes(wav)
        bound_metadata = prior[record["dataset_index"]]
        assert_exact_mapping(record, bound_metadata)
        reference_text = bound_metadata["text"]
        if sha256_bytes(reference_text.encode()) != record["source_reference_sha256"]:
            raise ValueError(f"Reference hash changed for {uid}")
        reference_path = ref_dir / f"{uid}.json"
        reference_path.write_text(json.dumps({"reference_text": reference_text}, ensure_ascii=False) + "\n")
        if sha256_bytes(raw) != record["source_audio_sha256"]:
            raise ValueError(f"Raw source hash changed for {uid}")
        audio_info = sf.info(audio_path)
        if audio_info.samplerate != TARGET_SAMPLE_RATE or audio_info.channels != 1:
            raise ValueError(f"Canonical WAV does not meet 16 kHz mono contract for {uid}")
        row = {
            "conversation_id": uid,
            "recording_unit": "utterance",
            "source_repo": SOURCE_DATASET,
            "source_revision": SOURCE_REVISION,
            "source_shard": record["source_shard"],
            "source_row": record["source_row"],
            "source_sentence_id": record["source_reference_sha256"],
            "audio_path": f"research/bridge_open/svarah_inputs/wav/{uid}.wav",
            "audio_sha256": sha256_file(audio_path),
            "raw_audio_sha256": record["source_audio_sha256"],
            "reference_path": f"research/bridge_open/svarah_inputs/references/{uid}.json",
            "reference_sha256": sha256_file(reference_path),
            "reference_text_sha256": record["source_reference_sha256"],
            "language": "en",
            "duration_seconds": audio_info.frames / audio_info.samplerate,
            "source_duration_seconds": record["decoded_duration_seconds"],
            "sample_rate": TARGET_SAMPLE_RATE,
            "source_sample_rate": record["source_sample_rate"],
            "source_channels": record["source_channels"],
            "resampling": "scipy.signal.resample_poly" if "resample_poly" in canonicalization else "none",
            "channel_policy": "mean_to_mono" if "downmix_mean" in canonicalization else "original_mono",
            "canonicalization": canonicalization,
            "reference_qc": "pass",
            "reference_qc_scope": "structural_not_human",
            "exposure_scope": "preexposed_exploratory_corpus",
        }
        cohort_rows.append(row)
    manifest = bundle_dir / "manifest.jsonl"
    manifest.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in cohort_rows))
    manifest_sha = sha256_file(manifest)
    config = {
        "model_id": "facebook/omniASR-CTC-1B",
        "model_revision": "8c22e3ffdaa4aab6431b128b84b991a7d9c2515c",
        "manifest_path": "research/bridge_open/svarah_inputs/manifest.jsonl",
        "manifest_sha256": manifest_sha,
        "conversation_ids": [row["conversation_id"] for row in cohort_rows],
        "cohort_kind": cohort_kind,
        "cohort_selection": cohort_selection,
        "cohort_count": len(cohort_rows),
        "supported_languages": ["en"],
        "full_waveform": True,
        "decoding_mode": "full_waveform_ctc",
        "batch_size": 1,
        "termination_semantics": "ctc_no_eos",
        "input_scope": ("preexposed_exploratory_svarah_native_smoke" if not partial else
                        "preexposed_exploratory_limited_svarah_shard_smoke") if not args.all_available else
                       "preexposed_exploratory_limited_svarah_available_shard_all_rows",
        "source_revision": SOURCE_REVISION,
    }
    config_name = "cohort_config_inputs.json" if args.all_available else "smoke_config_inputs.json"
    (args.output_dir / config_name).write_text(json.dumps(config, indent=2) + "\n")
    receipt = {
        "dataset": SOURCE_DATASET,
        "source_revision": SOURCE_REVISION,
        "cache_files": {path.name: {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
                        for path in ([*shards, args.cache_dir / "dataset_info.json"] if not partial else shards)},
        "prior_metadata": {"path": str(args.metadata.relative_to(ROOT)), "sha256": sha256_file(args.metadata), "rows": len(prior)},
        "arrow_schema": source_schema,
        "records": len(records),
        "dataset_coverage": "all_6656" if not partial else f"limited_{args.available_shard}",
        "structurally_eligible_le40": len(eligible),
        "flag_counts": dict(Counter(flag for record in records for flag in record["flags"])),
        "source_rate_counts": dict(Counter(str(record["source_sample_rate"]) for record in records)),
        "scipy_version": scipy.__version__,
        "partial_mapping_regression": mapping_regression,
        "exposure": exposure_receipt,
        "cohort_kind": cohort_kind,
        "cohort_count": len(cohort_rows),
        "cohort_ids": [row["conversation_id"] for row in cohort_rows],
        "cohort_manifest_sha256": manifest_sha,
        "cohort_flag_counts": dict(Counter(flag for record in cohort for flag in record["flags"])),
        "selection": cohort_selection,
        "scope": "preexposed_exploratory; no speaker identities inferred" if not partial else
                 "preexposed_exploratory_limited_shard; not full Svarah; no speaker identities inferred",
    }
    (args.audit_dir / "receipt.json").write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"status": "prepared", "records": len(records), "cohort": len(cohort_rows),
                      "cohort_kind": cohort_kind, "audit": str(args.audit_dir)}))


if __name__ == "__main__":
    main()
