"""Audit the exact MUCS and HiACC releases and build an ignored local manifest."""

from __future__ import annotations

from collections import Counter, defaultdict
import csv
import json
from pathlib import Path
import random
import statistics

import soundfile as sf

from ast_asr.termeval.normalize import normalize
from ast_asr.termeval.tokens import token_class


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "termeval"
OUT = ROOT / "results" / "night1"
SEED = 20261011


def _kv(path: Path) -> dict[str, str]:
    return dict(line.rstrip("\n").split(maxsplit=1) for line in path.open())


def _script_summary(rows: list[dict]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for row in rows:
        counts.update(token_class(token) for token in normalize(row["reference"]).split())
    return dict(sorted(counts.items()))


def _samples(rows: list[dict], n: int = 20) -> list[str]:
    return [r["reference"] for r in random.Random(SEED).sample(rows, min(n, len(rows)))]


def _counts(values) -> dict[str, int]:
    return dict(sorted(Counter(str(v) for v in values).items()))


def _probe_mucs() -> tuple[list[dict], dict]:
    base = ROOT / "data" / "mucs_hien" / "test"
    tx = base / "transcripts"
    if not tx.is_dir():
        raise FileNotFoundError(tx)
    texts = _kv(tx / "text")
    speakers = _kv(tx / "utt2spk")
    recordings = _kv(tx / "wav.scp")
    segments = [line.strip().split() for line in (tx / "segments").open()]
    assert len(texts) == len(speakers) == len(segments)
    clip_dir = ROOT / "data" / "mucs_hien" / "clips"
    clip_dir.mkdir(exist_ok=True)
    rows: list[dict] = []
    durations = []
    for utt, recording, start_s, end_s in segments:
        start, end = float(start_s), float(end_s)
        source = base / recordings[recording]
        info = sf.info(source)
        if info.samplerate != 16000:
            raise ValueError(f"MUCS recording not 16 kHz: {source}")
        if end > info.duration + 0.02 or end <= start:
            raise ValueError(f"bad segment {utt}: {start}-{end}, source {info.duration}")
        clip = clip_dir / f"{utt}.wav"
        if not clip.exists():
            with sf.SoundFile(source) as inp:
                inp.seek(round(start * 16000))
                audio = inp.read(round((end - start) * 16000), dtype="float32", always_2d=True)
            sf.write(clip, audio.mean(axis=1), 16000, subtype="PCM_16")
        durations.append(end - start)
        rows.append({"dataset": "mucs_hien", "utt_id": utt, "speaker": speakers[utt],
                     "recording": recording, "split": "test", "reference": texts[utt],
                     "audio_path": str(clip), "duration_s": end - start})
    return rows, {
        "recordings": len(recordings), "utterances": len(rows), "speakers": len(set(speakers.values())),
        "source_audio_rate_hz": 16000, "segments_present": True,
        "duration_s_min": min(durations), "duration_s_median": statistics.median(durations),
        "duration_s_p95": sorted(durations)[int(0.95 * (len(durations) - 1))],
        "duration_s_max": max(durations), "duration_hours": sum(durations) / 3600,
        "token_classes": _script_summary(rows), "samples": _samples(rows),
        "speaker_example": sorted(set(speakers.values()))[:5],
    }


def _probe_hiacc() -> tuple[list[dict], dict]:
    base = ROOT / "data" / "hiacc" / "Corpus"
    if not base.is_dir():
        raise FileNotFoundError(base)
    rows: list[dict] = []
    summary: dict = {"groups": {}, "metadata_files": {}}
    for group in ("adult", "children"):
        group_dir = base / group
        speaker_csv = group_dir / "metadata" / "speaker_info.csv"
        stats_csv = group_dir / "metadata" / "sentence_stats.csv"
        with speaker_csv.open(newline="") as handle:
            reader = csv.DictReader(handle)
            speaker_fields = reader.fieldnames
            speaker_meta = {r["PID"].strip(): r for r in reader}
        with stats_csv.open(newline="") as handle:
            stats_reader = csv.DictReader(handle)
            stats_fields = stats_reader.fieldnames
            stats_rows = sum(1 for _ in stats_reader)
        annotations = json.loads((group_dir / "annotations" / "code_switched_labels.json").read_text())
        annotation_fields = sorted({k for item in annotations for k in item})
        summary["metadata_files"][group] = {
            "speaker_info.csv": speaker_fields, "sentence_stats.csv": stats_fields,
            "code_switched_labels.json": annotation_fields,
            "speaker_info_rows": len(speaker_meta), "sentence_stats_rows": stats_rows,
        }
        group_rows = []
        missing_speaker_meta: Counter[str] = Counter()
        for split in ("train", "val", "test"):
            if group == "adult":
                tx_path = group_dir / "transcription" / f"combined_output_changed_{split}_output.txt"
            else:
                tx_path = group_dir / "transcript" / f"{split}_output.txt"
            for line in tx_path.open():
                name, reference = line.rstrip("\n").split(", ", 1)
                audio_path = group_dir / "audio" / f"{split}_split" / name
                if not audio_path.is_file():
                    raise FileNotFoundError(audio_path)
                speaker = name[:4]
                if speaker not in speaker_meta:
                    missing_speaker_meta[speaker] += 1
                info = sf.info(audio_path)
                if info.samplerate != 16000 or info.channels != 1:
                    raise ValueError(f"HiACC audio not 16 kHz mono: {audio_path}, {info}")
                row = {"dataset": "hiacc", "utt_id": f"{group}_{Path(name).stem}",
                       "speaker": f"{group}_{speaker}", "group": group, "split": split,
                       "reference": reference, "audio_path": str(audio_path),
                       "duration_s": info.duration,
                       "gender": speaker_meta.get(speaker, {}).get("Gender"),
                       "age": speaker_meta.get(speaker, {}).get("Age"),
                       "l1": speaker_meta.get(speaker, {}).get("L1")}
                group_rows.append(row)
        # Check whether the paper's claimed speaker-disjoint splits hold in this archive.
        by_split = {s: {r["speaker"] for r in group_rows if r["split"] == s} for s in ("train", "val", "test")}
        overlaps = {f"{a}_{b}": len(by_split[a] & by_split[b]) for a, b in
                    (("train", "val"), ("train", "test"), ("val", "test"))}
        summary["groups"][group] = {
            "utterances": len(group_rows), "speakers": len({r["speaker"] for r in group_rows}),
            "speaker_metadata_rows": len(speaker_meta),
            "missing_speaker_metadata_utterances": dict(missing_speaker_meta),
            "split_counts": _counts(r["split"] for r in group_rows),
            "split_speaker_overlap": overlaps,
            "hours": sum(r["duration_s"] for r in group_rows) / 3600,
            "token_classes": _script_summary(group_rows),
            "speaker_metadata": {field: _counts(v[field] for v in speaker_meta.values())
                                 for field in ("Gender", "Age", "L1")},
            "annotation_rows": len(annotations),
            "annotation_labels": _counts(x.get("label") for x in annotations),
            "samples": _samples(group_rows),
        }
        rows.extend(group_rows)
    summary["total_utterances"] = len(rows)
    summary["total_speakers"] = len({r["speaker"] for r in rows})
    summary["total_hours"] = sum(r["duration_s"] for r in rows) / 3600
    summary["token_classes"] = _script_summary(rows)
    return rows, summary


def main() -> None:
    DATA.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    mucs_rows, mucs = _probe_mucs()
    hiacc_rows, hiacc = _probe_hiacc()
    manifest = mucs_rows + hiacc_rows
    with (DATA / "manifest.jsonl").open("w") as handle:
        for row in manifest:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    (OUT / "data_summary.json").write_text(json.dumps({"mucs_hien": mucs, "hiacc": hiacc}, ensure_ascii=False, indent=2))
    print(json.dumps({"mucs": len(mucs_rows), "hiacc": len(hiacc_rows),
                      "mucs_hours": mucs["duration_hours"], "hiacc_hours": hiacc["total_hours"]}))


if __name__ == "__main__":
    main()
