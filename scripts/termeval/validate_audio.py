"""Check every local evaluation clip before accepting WER denominators."""

from __future__ import annotations

from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
import statistics

import numpy as np
import soundfile as sf

from ast_asr.termeval.normalize import normalize


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/night1"


def main() -> None:
    manifest = [json.loads(line) for line in (ROOT / "data/termeval/manifest.jsonl").open()]
    flags = []
    by_dataset: dict[str, Counter[str]] = defaultdict(Counter)
    duration_errors: dict[str, list[float]] = defaultdict(list)
    audio_hashes: dict[str, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))
    for row in manifest:
        dataset = row["dataset"]
        by_dataset[dataset]["utterances"] += 1
        path = Path(row["audio_path"])
        with sf.SoundFile(path) as handle:
            rate, channels, frames = handle.samplerate, handle.channels, handle.frames
            audio = handle.read(dtype="float32", always_2d=False)
        duration = frames / rate
        duration_errors[dataset].append(abs(duration - row["duration_s"]))
        reason = []
        if rate != 16000 or channels != 1:
            reason.append("format_not_16k_mono")
        if not frames:
            reason.append("zero_samples")
        if abs(duration - row["duration_s"]) > 0.03:
            reason.append("duration_mismatch_gt_30ms")
        rms = float(np.sqrt(np.mean(np.square(audio, dtype=np.float64)))) if len(audio) else 0
        peak = float(np.max(np.abs(audio))) if len(audio) else 0
        saturation_fraction = float(np.mean(np.abs(audio) >= 0.999)) if len(audio) else 0
        if rms < 0.001:
            reason.append("near_silent_rms_lt_0.001")
        if peak >= 0.999:
            reason.append("near_clipped_peak_ge_0.999")
        if saturation_fraction >= 0.001:
            reason.append("saturated_samples_ge_0.1_percent")
        n_ref = len(normalize(row["reference"]).split())
        if n_ref == 0:
            reason.append("empty_reference")
        if duration and n_ref / duration > 8:
            reason.append("reference_gt_8_tokens_per_sec")
        if duration and n_ref / duration < 0.2:
            reason.append("reference_lt_0.2_tokens_per_sec")
        # Hash the decoded sample stream to detect exact duplicate audio even
        # when WAV headers or filenames differ.
        audio_hashes[dataset][hashlib.sha256(audio.tobytes()).hexdigest()].append(row["utt_id"])
        if reason:
            by_dataset[dataset].update(reason)
            flags.append({"dataset": dataset, "utt_id": row["utt_id"], "speaker": row["speaker"],
                          "duration_s": round(duration, 5), "reference_tokens": n_ref,
                          "rms": round(rms, 7), "peak": round(peak, 5),
                          "saturation_fraction": round(saturation_fraction, 7),
                          "flags": ";".join(reason)})
    with (OUT / "audio_qc_flags.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["dataset", "utt_id", "speaker", "duration_s",
                                                       "reference_tokens", "rms", "peak",
                                                       "saturation_fraction", "flags"])
        writer.writeheader()
        writer.writerows(flags)
    summary = {}
    for dataset, counts in by_dataset.items():
        duplicate_groups = [ids for ids in audio_hashes[dataset].values() if len(ids) > 1]
        summary[dataset] = {"counts": dict(counts),
                            "max_manifest_duration_error_s": max(duration_errors[dataset]),
                            "median_manifest_duration_error_s": statistics.median(duration_errors[dataset]),
                            "exact_duplicate_audio_groups": len(duplicate_groups),
                            "exact_duplicate_audio_utterances": sum(len(ids) for ids in duplicate_groups),
                            "duplicate_examples": duplicate_groups[:10]}
    (OUT / "audio_qc.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
