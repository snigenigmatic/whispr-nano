"""Speaker-clustered paired script and age-group gaps for the frozen HiACC pilot.

These are descriptive contrasts. Script is not a verified language label, and
adult/child recordings differ in content and collection conditions.
"""

from __future__ import annotations

from collections import defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "results/night1/tokens_hiacc_whisper-large-v3_A_raw.parquet"
OUT = ROOT / "results/night1/speaker_group_gaps.json"
SEED = 20261011
RESAMPLES = 4000


def _rate(counts: dict, group: str, kind: str) -> float:
    err, denom = counts[group][kind]
    return err / denom


def main() -> None:
    tokens = pd.read_parquet(SOURCE)
    groups: dict[str, dict[str, dict[str, tuple[int, int]]]] = defaultdict(dict)
    for speaker, frame in tokens.groupby("speaker"):
        group = "adult" if speaker.startswith("adult_") else "children"
        if group == "children" and not speaker.startswith("children_"):
            raise ValueError(f"unexpected speaker prefix: {speaker}")
        speaker_counts = {}
        for kind in ("all", "LAT", "DEV"):
            part = frame if kind == "all" else frame[frame.tok_class == kind]
            denom = int(part.ref_idx.notna().sum())
            err = int(part.op.isin(("SUB", "DEL", "INS")).sum()) if kind == "all" else int(
                part.op.isin(("SUB", "DEL")).sum()
            )
            speaker_counts[kind] = (err, denom)
        groups[group][speaker] = speaker_counts

    def aggregate(samples: dict[str, list[str]]) -> dict[str, dict[str, tuple[int, int]]]:
        return {g: {kind: (
            sum(groups[g][s][kind][0] for s in ids),
            sum(groups[g][s][kind][1] for s in ids),
        ) for kind in ("all", "LAT", "DEV")} for g, ids in samples.items()}

    speaker_ids = {g: sorted(d) for g, d in groups.items()}
    if {g: len(v) for g, v in speaker_ids.items()} != {"adult": 24, "children": 20}:
        raise ValueError("unexpected frozen HiACC speaker counts")

    def statistics(counts: dict) -> dict[str, float]:
        a, c = counts["adult"], counts["children"]
        adult_script_gap = _rate(counts, "adult", "LAT") - _rate(counts, "adult", "DEV")
        child_script_gap = _rate(counts, "children", "LAT") - _rate(counts, "children", "DEV")
        return {
            "adult_wer": a["all"][0] / a["all"][1],
            "children_wer": c["all"][0] / c["all"][1],
            "children_minus_adult_wer": c["all"][0] / c["all"][1] - a["all"][0] / a["all"][1],
            "adult_lat_minus_dev": adult_script_gap,
            "children_lat_minus_dev": child_script_gap,
            "children_minus_adult_script_gap": child_script_gap - adult_script_gap,
        }

    observed_counts = aggregate(speaker_ids)
    point = statistics(observed_counts)
    if (observed_counts["adult"]["all"], observed_counts["children"]["all"]) != (
        (12267, 32593), (6498, 18081)
    ):
        raise ValueError("pooled word edit counts differ from the frozen report")
    if (observed_counts["adult"]["LAT"], observed_counts["children"]["LAT"]) != (
        (4398, 16395), (2354, 3582)
    ):
        raise ValueError("Latin token counts differ from the frozen report")
    rng = np.random.default_rng(SEED)
    draws = {key: [] for key in point}
    for _ in range(RESAMPLES):
        sampled = {g: list(rng.choice(ids, size=len(ids), replace=True))
                   for g, ids in speaker_ids.items()}
        values = statistics(aggregate(sampled))
        for key, value in values.items():
            draws[key].append(value)
    summary = {
        "source": str(SOURCE.relative_to(ROOT)),
        "source_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        "model": "openai/whisper-large-v3",
        "condition": "HiACC automatic language detection, raw reference scoring",
        "bootstrap_unit": "verified HiACC speaker ID, resampled within adult/children groups",
        "resamples": RESAMPLES,
        "seed": SEED,
        "counts": observed_counts,
        "estimates": {key: {
            "point_percentage_points": 100 * value,
            "ci95_percentage_points": [100 * v for v in np.percentile(draws[key], [2.5, 97.5])],
        } for key, value in point.items()},
        "interpretation": "Descriptive only: script does not uniquely identify language, and age groups differ in content/collection context.",
    }
    OUT.write_text(json.dumps(summary, indent=2))
    for key, value in summary["estimates"].items():
        print(key, round(value["point_percentage_points"], 2),
              [round(float(x), 2) for x in value["ci95_percentage_points"]])
    print("wrote", OUT)


if __name__ == "__main__":
    main()
