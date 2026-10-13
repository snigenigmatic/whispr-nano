"""Development-only test of a reference-free, selective five-beam fallback.

The detector uses the greedy hypothesis alone. References enter scoring only.
All source predictions were produced by already answered experiment nodes.
"""

from __future__ import annotations

from collections import defaultdict
import hashlib
import json
from pathlib import Path
import unicodedata

import jiwer
import numpy as np

from publication_study.loop_repair import suppress_adjacent_word_loops


ROOT = Path.home() / ".local/share/openresearch"
GREEDY = ROOT / "lahaja-pilot-v1" / "predictions.jsonl"
BEAM = ROOT / "lahaja-small-beam5-dev-v1" / "predictions.jsonl"
OUTPUT = ROOT / "lahaja-selective-beam-dev-v1"
EXPECTED_SHA256 = {
    GREEDY: "ccb5bd8fc1621fd90597876fa58f8c04078cabba6a48e812f26177fd951d303d",
    BEAM: "98885b5fc6288b949cbfc026160e70fae9ee4f91af7dd4b2f9e0a9bd2eb32134",
}
SEED = 20261011


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _normalize(text: str) -> str:
    text = unicodedata.normalize("NFC", text).lower()
    return " ".join("".join(
        c if unicodedata.category(c)[0] in "LMN" else " " for c in text
    ).split())


def _errors(reference: str, prediction: str) -> tuple[int, int]:
    ref = _normalize(reference)
    hyp = _normalize(prediction)
    counts = jiwer.process_words(ref, hyp)
    return counts.substitutions + counts.deletions + counts.insertions, len(ref.split())


def _load(path: Path) -> dict[int, dict]:
    actual = _sha(path)
    if actual != EXPECTED_SHA256[path]:
        raise ValueError(f"frozen prediction hash changed: {path.name} {actual}")
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    if len(rows) != 104 or len({r["row_index"] for r in rows}) != 104:
        raise ValueError("development cohort incomplete or duplicate row index")
    if len({r["sp_id"] for r in rows}) != 26:
        raise ValueError("development speaker count changed")
    for row in rows:
        errors, words = _errors(row["reference"], row["prediction"])
        if (errors, words) != (row["word_errors"], row["ref_words"]):
            raise ValueError("frozen row score disagrees with recomputation")
    return {r["row_index"]: r for r in rows}


def _ci(rows: list[dict], a: str, b: str) -> list[float]:
    grouped = defaultdict(lambda: [0, 0, 0])
    for row in rows:
        cell = grouped[row["sp_id"]]
        cell[0] += row[a]
        cell[1] += row[b]
        cell[2] += row["ref_words"]
    speakers = sorted(grouped)
    rng = np.random.default_rng(SEED)
    values = []
    for _ in range(4000):
        drawn = rng.choice(speakers, len(speakers), replace=True)
        errors_a = sum(grouped[s][0] for s in drawn)
        errors_b = sum(grouped[s][1] for s in drawn)
        words = sum(grouped[s][2] for s in drawn)
        values.append(100 * (errors_b - errors_a) / words)
    return [float(x) for x in np.percentile(values, [2.5, 97.5])]


def main() -> dict:
    greedy = _load(GREEDY)
    beam = _load(BEAM)
    if set(greedy) != set(beam):
        raise ValueError("greedy and beam development IDs differ")
    rows = []
    for idx in sorted(greedy):
        g, b = greedy[idx], beam[idx]
        for key in ("row_index", "sp_id", "fname", "reference", "ref_words", "scenario"):
            if g[key] != b[key]:
                raise ValueError(f"development row mismatch: {idx} {key}")
        # The same >=5-word loop detector was selected on this development set
        # for the earlier text-repair analysis; this comparison is exploratory.
        repaired, removed = suppress_adjacent_word_loops(g["prediction"])
        triggered = removed > 0
        selected = b["prediction"] if triggered else g["prediction"]
        fallback_errors, words = _errors(g["reference"], selected)
        repaired_errors, repaired_words = _errors(g["reference"], repaired)
        if words != g["ref_words"] or repaired_words != words:
            raise ValueError("reference changed during fallback scoring")
        rows.append({
            "row_index": idx, "sp_id": g["sp_id"], "reference": g["reference"],
            "greedy_prediction": g["prediction"], "beam_prediction": b["prediction"],
            "fallback_prediction": selected, "triggered": triggered,
            "greedy_errors": g["word_errors"], "beam_errors": b["word_errors"],
            "repair_errors": repaired_errors, "fallback_errors": fallback_errors,
            "ref_words": words,
        })
    OUTPUT.mkdir(parents=True, exist_ok=True)
    predictions = OUTPUT / "predictions.jsonl"
    with predictions.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    words = sum(r["ref_words"] for r in rows)
    errors = {name: sum(r[f"{name}_errors"] for r in rows)
              for name in ("greedy", "beam", "repair", "fallback")}
    result = {
        "status": "development_selective_beam_complete",
        "dataset": "ai4bharat/Lahaja",
        "dataset_revision": "d4ffd2ecbdd933e37c917ddcf620eef159ceb3a7",
        "cohort": "104 clips, 26 first-shard development speakers; not a held-out result",
        "rule": "if greedy hypothesis has >=5 identical adjacent normalized words, use five-beam transcript; else use greedy transcript",
        "source_sha256": {"greedy": EXPECTED_SHA256[GREEDY], "beam": EXPECTED_SHA256[BEAM]},
        "predictions_sha256": _sha(predictions),
        "clips": len(rows), "speakers": len({r["sp_id"] for r in rows}),
        "reference_words": words,
        "errors": errors,
        "wer": {name: count / words for name, count in errors.items()},
        "triggered_clips": sum(r["triggered"] for r in rows),
        "trigger_fraction": sum(r["triggered"] for r in rows) / len(rows),
        "triggered_helped_vs_greedy": sum(r["triggered"] and r["fallback_errors"] < r["greedy_errors"] for r in rows),
        "triggered_harmed_vs_greedy": sum(r["triggered"] and r["fallback_errors"] > r["greedy_errors"] for r in rows),
        "fallback_minus_greedy_wer_points_ci": _ci(rows, "greedy_errors", "fallback_errors"),
        "fallback_minus_beam_wer_points_ci": _ci(rows, "beam_errors", "fallback_errors"),
        "caveat": "The loop threshold was selected on this development sample. No independent-genuine-repeat control is included here.",
        "predictions_path": str(predictions),
    }
    (OUTPUT / "summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    return result
