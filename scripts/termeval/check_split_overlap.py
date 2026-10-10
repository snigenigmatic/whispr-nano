"""Quantify exact-reference overlap across the frozen speaker split."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from ast_asr.termeval.normalize import normalize


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/night1"
MANIFEST = ROOT / "data/termeval/manifest.jsonl"
SPLIT = OUT / "speaker_holdout_split.json"


def main() -> None:
    rows = [json.loads(line) for line in MANIFEST.open()]
    assignments = json.loads(SPLIT.read_text())["assignments"]
    results = {}
    for dataset in ("hiacc", "mucs_hien"):
        subset = [r for r in rows if r["dataset"] == dataset]
        dev_refs = {normalize(r["reference"]) for r in subset
                    if assignments[r["speaker"]] == "lexicon_dev"}
        heldout = [r for r in subset if assignments[r["speaker"]] == "heldout_eval"]
        overlap = sum(normalize(r["reference"]) in dev_refs for r in heldout)
        results[dataset] = {
            "heldout_utterances": len(heldout),
            "exact_reference_overlap_utterances": overlap,
            "novel_reference_utterances": len(heldout) - overlap,
            "exact_reference_overlap_fraction": overlap / len(heldout),
        }
    (OUT / "speaker_holdout_overlap.json").write_text(json.dumps({
        "definition": "exact match after the frozen C3 text normalizer",
        "manifest_sha256": hashlib.sha256(MANIFEST.read_bytes()).hexdigest(),
        "split_sha256": hashlib.sha256(SPLIT.read_bytes()).hexdigest(),
        "datasets": results,
    }, indent=2))
    print(json.dumps(results))


if __name__ == "__main__":
    main()
