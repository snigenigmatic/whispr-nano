"""CPU-only integrity audit for the frozen FLEURS v2 input corpus.

This checks encoded bytes and decoder agreement.  It does not inspect model
outputs or establish that a human transcript is semantically correct.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import subprocess
import tarfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
from scipy.io import wavfile


SAMPLE_RATE = 16_000
HF_CACHE = Path.home() / ".cache/huggingface/hub/datasets--google--fleurs/snapshots/70bb2e84b976b7e960aa89f1c648e09c59f894dd"
WHISPER_PROCESSOR = Path.home() / ".cache/huggingface/hub/models--openai--whisper-large-v3-turbo/snapshots/41f01f3fe87f28c78e2fbf8b568835947dd65ed9"


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def pcm_float(data: np.ndarray) -> np.ndarray:
    """Convert scipy's WAV representation to the soundfile float convention."""
    if np.issubdtype(data.dtype, np.floating):
        return data.astype(np.float32)
    if np.issubdtype(data.dtype, np.signedinteger):
        return data.astype(np.float32) / float(np.iinfo(data.dtype).max + 1)
    if np.issubdtype(data.dtype, np.unsignedinteger):
        limit = np.iinfo(data.dtype)
        return (data.astype(np.float32) - (limit.max + 1) / 2) / ((limit.max + 1) / 2)
    raise TypeError(f"unsupported scipy WAV dtype {data.dtype}")


def validate_waveform(audio: np.ndarray, sample_rate: int) -> None:
    """Red-capable amplitude/rate properties used by this audit."""
    if sample_rate != SAMPLE_RATE:
        raise ValueError("wrong sample rate")
    if audio.ndim != 1 or not len(audio):
        raise ValueError("expected nonempty mono audio")
    if not np.isfinite(audio).all():
        raise ValueError("non-finite samples")
    if not np.any(audio):
        raise ValueError("silent all-zero waveform")
    if np.abs(audio).max() > 1.00001:
        raise ValueError("out-of-range waveform amplitude")


def _reference_text(row: dict[str, Any]) -> str:
    return json.loads(Path(row["reference_path"]).read_text())["reference_text"]


def _archive_and_tsv(rows: list[dict[str, Any]]) -> list[str]:
    issues = []
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(row["source_config"], row["source_shard"])].append(row)
    for (config, shard), selected in groups.items():
        archive = HF_CACHE / shard
        tsv = HF_CACHE / f"data/{config}/test.tsv"
        if not archive.is_file() or not tsv.is_file():
            issues.append(f"{config}: missing pinned cache archive or TSV")
            continue
        expected_archive = {row["source_archive_sha256"] for row in selected}
        if len(expected_archive) != 1 or sha(archive) != next(iter(expected_archive)):
            issues.append(f"{config}: source archive SHA mismatch")
        records = list(csv.reader(io.StringIO(tsv.read_text()), delimiter="\t", quoting=csv.QUOTE_NONE))
        tsv_by_name = {record[1]: record for record in records if len(record) == 7}
        desired = {row["source_member"]: row for row in selected}
        found = set()
        with tarfile.open(archive, "r:gz") as tar:
            for member in tar:
                if member.name not in desired:
                    continue
                row = desired[member.name]
                found.add(member.name)
                data = tar.extractfile(member).read()
                if hashlib.sha256(data).hexdigest() != row["audio_sha256"]:
                    issues.append(f"{row['conversation_id']}: archive member hash mismatch")
                source = tsv_by_name.get(Path(member.name).name)
                if source is None:
                    issues.append(f"{row['conversation_id']}: source member missing TSV row")
                elif source[0] != str(row["source_sentence_id"]) or source[3] != _reference_text(row):
                    issues.append(f"{row['conversation_id']}: TSV/reference join mismatch")
        if found != set(desired):
            issues.append(f"{config}: archive missing {len(set(desired) - found)} selected members")
    return issues


def _ffmpeg_decode(path: Path) -> np.ndarray:
    process = subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-i", str(path), "-ac", "1", "-ar", "16000", "-f", "f32le", "pipe:1"], check=True, capture_output=True)
    return np.frombuffer(process.stdout, dtype="<f4")


def _processor_parity(rows: list[dict[str, Any]]) -> dict[str, Any]:
    from transformers import WhisperProcessor
    processor = WhisperProcessor.from_pretrained(WHISPER_PROCESSOR, local_files_only=True)
    report = {"model": "openai/whisper-large-v3-turbo", "processor_snapshot": str(WHISPER_PROCESSOR), "local_files_only": True, "languages": {}}
    for language in sorted({row["language"] for row in rows}):
        chosen = sorted((row for row in rows if row["language"] == language), key=lambda row: row["conversation_id"])[:8]
        audio = [sf.read(row["audio_path"], dtype="float32")[0] for row in chosen]
        batch = processor(audio, sampling_rate=SAMPLE_RATE, padding="max_length", return_attention_mask=True, return_tensors="np")
        features = batch.input_features
        masks = batch.attention_mask
        if features.shape != (len(chosen), 128, 3000):
            raise ValueError(f"{language}: expected 128x3000 features, got {features.shape}")
        parity = []
        for index, samples in enumerate(audio):
            single = processor(samples, sampling_rate=SAMPLE_RATE, padding="max_length", return_attention_mask=True, return_tensors="np")
            parity.append(bool(np.array_equal(features[index], single.input_features[0]) and np.array_equal(masks[index], single.attention_mask[0])))
        report["languages"][language] = {"conversation_ids": [row["conversation_id"] for row in chosen], "feature_shape": list(features.shape),
            "attention_mask_lengths": [int(mask.sum()) for mask in masks], "batch_singleton_mel_exact": parity}
        if not all(parity):
            raise ValueError(f"{language}: batch/singleton mel parity mismatch")
    return report


def audit(manifest_path: str | Path, output_dir: str | Path) -> dict[str, Any]:
    manifest_path = Path(manifest_path)
    rows = [json.loads(line) for line in manifest_path.read_text().splitlines()]
    if len(rows) != 2322 or len({row["conversation_id"] for row in rows}) != 2322:
        raise ValueError("expected exactly 2322 unique v2 rows")
    details, issues = [], []
    ffmpeg = defaultdict(list)
    for row in rows:
        path = Path(row["audio_path"])
        info = sf.info(path)
        audio, rate = sf.read(path, dtype="float32")
        try:
            validate_waveform(audio, rate)
        except ValueError as exc:
            issues.append(f"{row['conversation_id']}: {exc}")
        scipy_rate, scipy_audio = wavfile.read(path)
        scipy_float = pcm_float(np.asarray(scipy_audio))
        delta = float(np.max(np.abs(audio - scipy_float))) if audio.shape == scipy_float.shape else math.inf
        if rate != scipy_rate or audio.shape != scipy_float.shape or delta > 1.1 / 32768:
            issues.append(f"{row['conversation_id']}: soundfile/scipy decode mismatch")
        if sha(path) != row["audio_sha256"] or sha(Path(row["reference_path"])) != row["reference_sha256"]:
            issues.append(f"{row['conversation_id']}: local source hash mismatch")
        detail = dict(conversation_id=row["conversation_id"], language=row["language"], sample_rate=rate, channels=info.channels,
            frames=info.frames, subtype=info.subtype, finite=bool(np.isfinite(audio).all()), peak=float(np.abs(audio).max()),
            rms=float(np.sqrt(np.mean(audio.astype("float64") ** 2))), dc=float(np.mean(audio)), clipped_fraction=float(np.mean(np.abs(audio) >= .999)),
            exact_zero_fraction=float(np.mean(audio == 0)), scipy_rate=int(scipy_rate), scipy_dtype=str(scipy_audio.dtype), scipy_max_abs_delta=delta)
        details.append(detail)
        if len(ffmpeg[row["language"]]) < 3:
            ffmpeg[row["language"]].append((row, audio))
    archive_issues = _archive_and_tsv(rows)
    issues.extend(archive_issues)
    ffmpeg_report = []
    for language, samples in sorted(ffmpeg.items()):
        for row, source in samples:
            canonical = _ffmpeg_decode(Path(row["audio_path"]))
            delta = float(np.max(np.abs(source - canonical))) if source.shape == canonical.shape else math.inf
            if source.shape != canonical.shape or delta > 2e-6:
                issues.append(f"{row['conversation_id']}: ffmpeg/soundfile decode mismatch")
            ffmpeg_report.append({"conversation_id": row["conversation_id"], "language": language, "sample_count": len(source), "max_abs_delta": delta})
    synthetic = {}
    for name, waveform, rate in (("wrong_rate", np.ones(2, dtype=np.float32), 8000), ("out_of_range", np.array([1.1], dtype=np.float32), 16000), ("silent", np.zeros(2, dtype=np.float32), 16000)):
        try:
            validate_waveform(waveform, rate)
        except ValueError as exc:
            synthetic[name] = str(exc)
        else:
            raise AssertionError(f"red-capable check accepted {name}")
    processor = _processor_parity(rows)
    language_summary = {language: dict(count=sum(row["language"] == language for row in details),
        max_peak=max(row["peak"] for row in details if row["language"] == language), max_rms=max(row["rms"] for row in details if row["language"] == language),
        max_abs_dc=max(abs(row["dc"]) for row in details if row["language"] == language), max_clipped_fraction=max(row["clipped_fraction"] for row in details if row["language"] == language)) for language in sorted({row["language"] for row in details})}
    report = {"manifest": str(manifest_path), "manifest_sha256": sha(manifest_path), "rows": len(rows), "passed": not issues,
        "decoder_checks": {"soundfile_and_scipy_all_rows": len(details), "ffmpeg_canonical_sample": ffmpeg_report}, "archive_and_tsv_issues": archive_issues,
        "property_red_checks": synthetic, "processor": processor, "per_language": language_summary, "issues": issues, "records": details,
        "scope": "Automated byte, decoder, waveform, feature-input, and metadata checks only; no human transcript-content verification."}
    output = Path(output_dir); output.mkdir(parents=True, exist_ok=False)
    (output / "machine_report.json").write_text(json.dumps(report, indent=2) + "\n")
    conclusion = ["# FLEURS input audit", "", f"Rows audited: {len(rows)}. Overall pass: {report['passed']}.", "", "The audit checks exact local/archive bytes, TSV joins, three independent audio decode paths, waveform properties, and processor batch/singleton mel parity. It does not establish transcript semantic correctness or diagnose model WER."]
    if issues: conclusion += ["", "## Issues", "", *[f"- {issue}" for issue in issues]]
    else: conclusion += ["", "No input-integrity discrepancy was found by these bounded checks."]
    (output / "CONCLUSIONS.md").write_text("\n".join(conclusion) + "\n")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default="data/bridge_open/fleurs_test_v1/corpus-v2.jsonl")
    parser.add_argument("--output-dir", default="research/bridge_open/artifacts/fleurs_input_audit")
    args = parser.parse_args(argv)
    report = audit(args.manifest, args.output_dir)
    print(json.dumps({"event": "fleurs_input_audit", "passed": report["passed"], "rows": report["rows"], "issues": len(report["issues"])}))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
