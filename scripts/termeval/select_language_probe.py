"""Freeze an output-blind matched MUCS language-setting QC subset.

The probe uses only IDs and speakers from the first cached auto-language batch,
never the hypothesis text or reference, and is an engineering diagnostic.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import random


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/night1"
SEED = 20261011


def main() -> None:
    manifest = {r["utt_id"]: r for r in map(json.loads, (ROOT / "data/termeval/manifest.jsonl").open())}
    partial = OUT / "predictions_mucs_hien_auto_partial.jsonl"
    candidates = [json.loads(line)["utt_id"] for line in partial.open()]
    if len(candidates) != len(set(candidates)) or len(candidates) < 640:
        raise ValueError("expected at least 640 distinct cached MUCS auto-language IDs")
    candidates = candidates[:640]
    by_speaker: dict[str, list[str]] = {}
    for utt in candidates:
        row = manifest[utt]
        if row["dataset"] != "mucs_hien":
            raise ValueError("non-MUCS candidate")
        by_speaker.setdefault(row["speaker"], []).append(utt)
    rng = random.Random(SEED)
    selected = []
    for speaker, ids in sorted(by_speaker.items()):
        if len(ids) < 20:
            raise ValueError(f"speaker {speaker} lacks 20 cached candidates")
        selected.extend(rng.sample(sorted(ids), 20))
    output = {"seed": SEED, "method": "20 IDs per speaker among first 640 cached MUCS auto-language clips; no text inspected for selection",
              "n_speakers": len(by_speaker), "n_utterances": len(selected),
              "auto_partial_sha256_at_selection": hashlib.sha256(partial.read_bytes()).hexdigest(),
              "utt_ids": selected}
    (OUT / "language_probe_ids.json").write_text(json.dumps(output, indent=2))
    print(json.dumps({"n_speakers": len(by_speaker), "n_utterances": len(selected)}))


if __name__ == "__main__":
    main()
