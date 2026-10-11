"""Package only a complete, paired Lahaja Whisper-small development comparison.

All prediction text stays local because Lahaja access is gated. This script
checks row identity, references and raw edit counts before publishing aggregates.
"""

from __future__ import annotations

from collections import defaultdict
import hashlib
import json
from pathlib import Path
import unicodedata

import jiwer
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
RUNS = Path.home() / ".local/share/openresearch"
GREEDY = RUNS / "lahaja-pilot-v1"
BEAM = RUNS / "lahaja-small-beam5-dev-v1"
OUT = ROOT / "results/night1/lahaja_small_decoder_dev_20261011.json"
SEED = 20261011
RESAMPLES = 4000


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _normalize(value: str) -> str:
    value = unicodedata.normalize("NFC", value).lower()
    return " ".join("".join(
        ch if unicodedata.category(ch)[0] in "LMN" else " " for ch in value
    ).split())


def _load(folder: Path) -> tuple[dict, list[dict]]:
    summary = json.loads((folder / "summary.json").read_text())
    rows = [json.loads(line) for line in (folder / "predictions.jsonl").read_text().splitlines()]
    if len(rows) != 104 or len({r["row_index"] for r in rows}) != 104:
        raise ValueError(f"incomplete or duplicate development predictions: {folder}")
    if len({r["sp_id"] for r in rows}) != 26:
        raise ValueError(f"unexpected development speaker count: {folder}")
    for row in rows:
        edits = jiwer.process_words(_normalize(row["reference"]), _normalize(row["prediction"]))
        if (edits.substitutions, edits.deletions, edits.insertions) != (
            row["substitutions"], row["deletions"], row["insertions"]
        ):
            raise ValueError(f"cached edit counts failed validation: {folder}")
        if row["word_errors"] != edits.substitutions + edits.deletions + edits.insertions:
            raise ValueError(f"cached word errors failed validation: {folder}")
        if row["ref_words"] != len(_normalize(row["reference"]).split()):
            raise ValueError(f"cached reference count failed validation: {folder}")
    if summary["overall"]["word_errors"] != sum(r["word_errors"] for r in rows):
        raise ValueError(f"summary disagrees with predictions: {folder}")
    return summary, rows


def _quality(rows: list[dict]) -> dict:
    return {
        "generated_token_cap_256_count": (
            sum(r["generated_tokens"] >= 256 for r in rows)
            if all("generated_tokens" in r for r in rows) else None
        ),
        "hypothesis_at_least_3x_reference_words_count": sum(
            len(_normalize(r["prediction"]).split()) >= 3 * r["ref_words"] for r in rows
        ),
        "utterance_wer_over_100_percent_count": sum(
            r["word_errors"] > r["ref_words"] for r in rows
        ),
    }


def _paired_ci(greedy: list[dict], beam: list[dict]) -> list[float]:
    counts = defaultdict(lambda: [0, 0, 0])
    for before, after in zip(greedy, beam):
        speaker = before["sp_id"]
        counts[speaker][0] += before["word_errors"]
        counts[speaker][1] += after["word_errors"]
        counts[speaker][2] += before["ref_words"]
    ids = sorted(counts)
    rng = np.random.default_rng(SEED)
    differences = []
    for _ in range(RESAMPLES):
        sampled = rng.choice(ids, size=len(ids), replace=True)
        before = sum(counts[s][0] for s in sampled)
        after = sum(counts[s][1] for s in sampled)
        words = sum(counts[s][2] for s in sampled)
        differences.append(100 * (after - before) / words)
    return [float(x) for x in np.percentile(differences, [2.5, 97.5])]


def main() -> None:
    greedy_summary, greedy = _load(GREEDY)
    beam_summary, beam = _load(BEAM)
    greedy.sort(key=lambda r: r["row_index"])
    beam.sort(key=lambda r: r["row_index"])
    for before, after in zip(greedy, beam):
        for key in ("row_index", "fname", "sp_id", "scenario", "reference", "ref_words"):
            if before[key] != after[key]:
                raise ValueError(f"nonpaired {key} on row {before['row_index']}")
    if greedy_summary["model"] != beam_summary["model"] or (
        greedy_summary["model_revision"] != beam_summary["model_revision"]
    ):
        raise ValueError("decoder control changed the model checkpoint")
    if greedy_summary["dataset_revision"] != beam_summary["dataset_revision"]:
        raise ValueError("decoder control changed the dataset revision")
    words = sum(r["ref_words"] for r in greedy)
    before_errors = sum(r["word_errors"] for r in greedy)
    after_errors = sum(r["word_errors"] for r in beam)
    result = {
        "status": "completed_development_decoder_control",
        "dataset": greedy_summary["dataset"],
        "dataset_revision": greedy_summary["dataset_revision"],
        "model": greedy_summary["model"],
        "model_revision": greedy_summary["model_revision"],
        "greedy_run_id": "85e94d4f-b9ea-4227-be40-cefbc9cb705c",
        "beam5_run_id": "916e8365-1d40-4a5f-a2a9-5f327923bafa",
        "greedy_code_commit": "efaa2b0",
        "beam5_code_commit": "1c3254d942e341f4828cb808ecb0d489e1643a83",
        "clips": 104,
        "speakers": 26,
        "reference_words": words,
        "greedy": {"word_errors": before_errors, "wer_percent": 100 * before_errors / words,
                   "quality": _quality(greedy),
                   "predictions_sha256": _sha(GREEDY / "predictions.jsonl")},
        "beam5": {"word_errors": after_errors, "wer_percent": 100 * after_errors / words,
                  "quality": _quality(beam),
                  "predictions_sha256": _sha(BEAM / "predictions.jsonl")},
        "beam5_minus_greedy_wer_points": 100 * (after_errors - before_errors) / words,
        "paired_speaker_bootstrap_95_ci_difference_points": _paired_ci(greedy, beam),
        "bootstrap_resamples": RESAMPLES,
        "bootstrap_seed": SEED,
        "limits": [
            "This is the same first-shard development sample used to diagnose greedy failure; it is not a held-out paper result.",
            "Beam size also changes inference cost; latency and memory were not compared in this receipt.",
            "The scorer preserves written digit versus spoken number-word differences.",
        ],
        "packaging_script_sha256": _sha(Path(__file__)),
    }
    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"output": str(OUT), "beam5_wer_percent": result["beam5"]["wer_percent"],
                      "beam5_minus_greedy_points": result["beam5_minus_greedy_wer_points"],
                      "beam5_quality": result["beam5"]["quality"]}))


if __name__ == "__main__":
    main()
