"""Freeze speaker-disjoint lexicon-development and evaluation partitions."""

from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path
import random


ROOT = Path(__file__).resolve().parents[2]
SEED = 20261011


def main() -> None:
    rows = [json.loads(line) for line in (ROOT / "data/termeval/manifest.jsonl").open()]
    by_stratum: dict[tuple[str, str], set[str]] = defaultdict(set)
    for row in rows:
        by_stratum[(row["dataset"], row.get("group", "all"))].add(row["speaker"])
    rng = random.Random(SEED)
    assignments = {}
    for (dataset, stratum), speakers in sorted(by_stratum.items()):
        ordered = sorted(speakers)
        rng.shuffle(ordered)
        n_dev = len(ordered) // 2
        for speaker in ordered[:n_dev]:
            assignments[speaker] = "lexicon_dev"
        for speaker in ordered[n_dev:]:
            assignments[speaker] = "heldout_eval"
    output = {"seed": SEED, "unit": "speaker ID", "stratification": "dataset and HiACC adult/child",
              "assignments": dict(sorted(assignments.items())),
              "counts": {dataset: {partition: sum(r["dataset"] == dataset and assignments[r["speaker"]] == partition
                                                   for r in rows)
                                   for partition in ("lexicon_dev", "heldout_eval")}
                         for dataset in ("mucs_hien", "hiacc")}}
    dest = ROOT / "results/night1/speaker_holdout_split.json"
    dest.write_text(json.dumps(output, indent=2))
    print(json.dumps({"counts": output["counts"],
                      "speaker_counts": {p: sum(v == p for v in assignments.values())
                                         for p in ("lexicon_dev", "heldout_eval")} }))


if __name__ == "__main__":
    main()
