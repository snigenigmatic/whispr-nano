"""Evaluate the frozen adjacent-word-loop repair on unseen Lahaja speakers."""

from __future__ import annotations

from collections import defaultdict
import hashlib
import json
from pathlib import Path
import time
import unicodedata

import jiwer
import numpy as np

from publication_study.loop_repair import MIN_REPETITIONS, suppress_adjacent_word_loops


RUNS = Path.home() / ".local/share/openresearch"
SOURCE = RUNS / "lahaja-small-heldout-405-v1"
OUTPUT = RUNS / "lahaja-loop-repair-heldout-v1"
POLICY = Path(__file__).resolve().parents[2] / "src/publication_study/loop_repair.py"
EXPECTED_CLIPS = 405
EXPECTED_SPEAKERS = 106
EXPECTED_SOURCE_SHA256 = "359d322efcfceb671869042418bb8d07684816e528c7afa785dd15fa5e773b09"
EXPECTED_POLICY_SHA256 = "4a89775ecae07364dd69a661e1ac612a3ffd10c7991f76305b45ae48c8873867"
EXPECTED_SELECTION_SHA256 = "6134e2a9eabd65f590bf4f7f7829626e520379a66e058d9dd51ce4acf36c4060"
SEED = 20261011
RESAMPLES = 4000


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _normalize(value: str) -> str:
    value = unicodedata.normalize("NFC", value).lower()
    return " ".join("".join(
        ch if unicodedata.category(ch)[0] in "LMN" else " " for ch in value
    ).split())


def _errors(reference: str, hypothesis: str) -> tuple[int, int]:
    ref = _normalize(reference)
    hyp = _normalize(hypothesis)
    counts = jiwer.process_words(ref, hyp)
    return counts.substitutions + counts.deletions + counts.insertions, len(ref.split())


def _speaker_ci(rows: list[dict]) -> list[float]:
    counts = defaultdict(lambda: [0, 0, 0])
    for row in rows:
        cell = counts[row["sp_id"]]
        cell[0] += row["raw_errors"]
        cell[1] += row["repaired_errors"]
        cell[2] += row["ref_words"]
    ids = sorted(counts)
    rng = np.random.default_rng(SEED)
    values = []
    for _ in range(RESAMPLES):
        sample = rng.choice(ids, size=len(ids), replace=True)
        before = sum(counts[s][0] for s in sample)
        after = sum(counts[s][1] for s in sample)
        words = sum(counts[s][2] for s in sample)
        values.append(100 * (after - before) / words)
    return [float(x) for x in np.percentile(values, [2.5, 97.5])]


def main() -> dict:
    started = time.perf_counter()
    source_summary = json.loads((SOURCE / "summary.json").read_text())
    source_predictions = SOURCE / "predictions.jsonl"
    if _sha(source_predictions) != EXPECTED_SOURCE_SHA256:
        raise ValueError("frozen source prediction hash changed")
    rows = [json.loads(line) for line in source_predictions.read_text().splitlines()]
    if len(rows) != EXPECTED_CLIPS or len({
        (r["data_file"], r["row_index"]) for r in rows
    }) != EXPECTED_CLIPS:
        raise ValueError("held-out source is incomplete or contains duplicate rows")
    if len({r["sp_id"] for r in rows}) != EXPECTED_SPEAKERS:
        raise ValueError("held-out speaker count changed")
    if source_summary["model"] != "openai/whisper-small":
        raise ValueError("unexpected source model")
    if source_summary["dataset_revision"] != "d4ffd2ecbdd933e37c917ddcf620eef159ceb3a7":
        raise ValueError("unexpected Lahaja dataset revision")
    if source_summary["selection_sha256"] != EXPECTED_SELECTION_SHA256:
        raise ValueError("held-out clip selection changed")
    if _sha(POLICY) != EXPECTED_POLICY_SHA256:
        raise ValueError("development-frozen policy changed")
    dev_rows = [json.loads(line) for line in (
        RUNS / "lahaja-pilot-v1/predictions.jsonl"
    ).read_text().splitlines()]
    if {r["sp_id"] for r in rows} & {r["sp_id"] for r in dev_rows}:
        raise ValueError("held-out speakers overlap development speakers")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    output_predictions = OUTPUT / "predictions.jsonl"
    scored = []
    with output_predictions.open("w") as handle:
        for row in rows:
            raw_errors, ref_words = _errors(row["reference"], row["prediction"])
            if (raw_errors, ref_words) != (row["word_errors"], row["ref_words"]):
                raise ValueError("source edit arithmetic changed")
            repaired, removed = suppress_adjacent_word_loops(row["prediction"])
            repaired_errors, repaired_words = _errors(row["reference"], repaired)
            if repaired_words != ref_words:
                raise ValueError("reference token count changed")
            record = {
                "data_file": row["data_file"], "row_index": row["row_index"],
                "sp_id": row["sp_id"],
                "scenario": row["scenario"], "reference": row["reference"],
                "raw_prediction": row["prediction"], "repaired_prediction": repaired,
                "raw_errors": raw_errors, "repaired_errors": repaired_errors,
                "ref_words": ref_words, "removed_words": removed,
            }
            scored.append(record)
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    before = sum(r["raw_errors"] for r in scored)
    after = sum(r["repaired_errors"] for r in scored)
    words = sum(r["ref_words"] for r in scored)
    if (before, words) != (source_summary["overall"]["word_errors"],
                           source_summary["overall"]["ref_words"]):
        raise ValueError("source summary disagrees with recomputed pooled edits")
    result = {
        "status": "speaker_disjoint_loop_repair_complete",
        "selection_sha256": EXPECTED_SELECTION_SHA256,
        "speaker_overlap_with_development": 0,
        "dataset": source_summary["dataset"],
        "dataset_revision": source_summary["dataset_revision"],
        "model": source_summary["model"],
        "model_revision": source_summary["model_revision"],
        "source_predictions_sha256": _sha(source_predictions),
        "policy_sha256": _sha(POLICY),
        "rule": f"collapse >= {MIN_REPETITIONS} adjacent identical normalized hypothesis words to one",
        "clips": len(scored),
        "speakers": len({r["sp_id"] for r in scored}),
        "reference_words": words,
        "raw_word_errors": before,
        "raw_wer": before / words,
        "repaired_word_errors": after,
        "repaired_wer": after / words,
        "repaired_minus_raw_wer_points": 100 * (after - before) / words,
        "paired_speaker_bootstrap_95_ci_difference_points": _speaker_ci(scored),
        "bootstrap_resamples": RESAMPLES,
        "bootstrap_seed": SEED,
        "clips_changed": sum(r["removed_words"] > 0 for r in scored),
        "words_removed": sum(r["removed_words"] for r in scored),
        "clips_helped": sum(r["repaired_errors"] < r["raw_errors"] for r in scored),
        "clips_harmed": sum(r["repaired_errors"] > r["raw_errors"] for r in scored),
        "raw_utterance_wer_over_100_percent_count": sum(
            r["raw_errors"] > r["ref_words"] for r in scored),
        "repaired_utterance_wer_over_100_percent_count": sum(
            r["repaired_errors"] > r["ref_words"] for r in scored),
        "repair_elapsed_s": time.perf_counter() - started,
        "predictions_sha256": _sha(output_predictions),
        "predictions_path": str(output_predictions),
    }
    (OUTPUT / "summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    return result
