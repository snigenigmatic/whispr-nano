"""Independently validate and publish aggregate Lahaja loop-repair evidence."""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import re
import sys
import unicodedata

import jiwer
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from publication_study.loop_repair import suppress_adjacent_word_loops  # noqa: E402


RUNS = Path.home() / ".local/share/openresearch"
DATASET_REVISION = "d4ffd2ecbdd933e37c917ddcf620eef159ceb3a7"
MODEL_REVISION = "973afd24965f72e36ca33b3055d56a652f456b4d"
SELECTION_SHA256 = "6134e2a9eabd65f590bf4f7f7829626e520379a66e058d9dd51ce4acf36c4060"
SEED = 20261011
RESAMPLES = 4000
SPLITS = {
    "dev": ("lahaja-pilot-v1", "lahaja-loop-repair-dev-v1", 104, 26),
    "heldout": ("lahaja-small-heldout-405-v1", "lahaja-loop-repair-heldout-v1", 405, 106),
}
SOURCE_PROVENANCE = {
    "dev": ("85e94d4f-b9ea-4227-be40-cefbc9cb705c", "efaa2b000442cf804e5c4bfb8fed28e0352407e7"),
    "heldout": ("b01c023b-d98f-4b28-9338-6655f95a82b4", "662adfbe145f614a836266360e0eef3ad5d53e6d"),
}


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _words(value: str) -> list[str]:
    value = unicodedata.normalize("NFC", value).lower()
    return "".join(ch if unicodedata.category(ch)[0] in "LMN" else " " for ch in value).split()


def _errors(reference: str, hypothesis: str) -> int:
    counts = jiwer.process_words(" ".join(_words(reference)), " ".join(_words(hypothesis)))
    return counts.substitutions + counts.deletions + counts.insertions


def _key(row: dict, split: str) -> tuple:
    return (row["data_file"], row["row_index"]) if split == "heldout" else (row["row_index"],)


def _paired_ci(rows: list[dict]) -> list[float]:
    counts = defaultdict(lambda: [0, 0, 0])
    for row in rows:
        bucket = counts[row["sp_id"]]
        bucket[0] += row["raw_errors"]
        bucket[1] += row["repaired_errors"]
        bucket[2] += row["ref_words"]
    ids = sorted(counts)
    rng = np.random.default_rng(SEED)
    values = []
    for _ in range(RESAMPLES):
        sample = rng.choice(ids, size=len(ids), replace=True)
        values.append(100 * (sum(counts[s][1] for s in sample) -
                             sum(counts[s][0] for s in sample)) /
                      sum(counts[s][2] for s in sample))
    return [float(x) for x in np.percentile(values, [2.5, 97.5])]


def package(split: str, run_id: str, code_commit: str) -> dict:
    source_name, repair_name, expected_clips, expected_speakers = SPLITS[split]
    source_folder, repair_folder = RUNS / source_name, RUNS / repair_name
    source_path = source_folder / "predictions.jsonl"
    repair_path = repair_folder / "predictions.jsonl"
    source_summary = json.loads((source_folder / "summary.json").read_text())
    summary = json.loads((repair_folder / "summary.json").read_text())
    source_rows = [json.loads(line) for line in source_path.read_text().splitlines()]
    repair_rows = [json.loads(line) for line in repair_path.read_text().splitlines()]
    if len(source_rows) != expected_clips or len(repair_rows) != expected_clips:
        raise ValueError("incomplete source or repaired prediction set")
    if len({_key(r, split) for r in source_rows}) != expected_clips:
        raise ValueError("duplicate source IDs")
    if len({_key(r, split) for r in repair_rows}) != expected_clips:
        raise ValueError("duplicate repaired IDs")
    if len({r["sp_id"] for r in source_rows}) != expected_speakers:
        raise ValueError("speaker count changed")
    if source_summary["model_revision"] != MODEL_REVISION or (
        source_summary["dataset_revision"] != DATASET_REVISION
    ):
        raise ValueError("source model or dataset revision changed")
    if split == "heldout":
        if source_summary["selection_sha256"] != SELECTION_SHA256:
            raise ValueError("held-out IDs differ from the frozen Turbo sample")
        dev_rows = [json.loads(line) for line in (RUNS / "lahaja-pilot-v1/predictions.jsonl").read_text().splitlines()]
        if {r["sp_id"] for r in dev_rows} & {r["sp_id"] for r in source_rows}:
            raise ValueError("development speakers leaked into held-out sample")
    if summary["source_predictions_sha256"] != _sha(source_path) or (
        summary["predictions_sha256"] != _sha(repair_path)
    ):
        raise ValueError("prediction hash mismatch")
    policy_path = ROOT / "src/publication_study/loop_repair.py"
    if summary["policy_sha256"] != _sha(policy_path):
        raise ValueError("frozen repair policy changed")
    source_by_id = {_key(r, split): r for r in source_rows}
    checked = []
    for repaired in repair_rows:
        original = source_by_id[_key(repaired, split)]
        if repaired["sp_id"] != original["sp_id"] or repaired["reference"] != original["reference"]:
            raise ValueError("nonpaired speaker or reference")
        if repaired["raw_prediction"] != original["prediction"]:
            raise ValueError("source prediction mismatch")
        expected_text, expected_removed = suppress_adjacent_word_loops(original["prediction"])
        if (repaired["repaired_prediction"], repaired["removed_words"]) != (
            expected_text, expected_removed
        ):
            raise ValueError("repaired prediction differs from frozen policy")
        if repaired["ref_words"] != len(_words(original["reference"])):
            raise ValueError("reference count changed")
        if (repaired["raw_errors"], repaired["repaired_errors"]) != (
            _errors(original["reference"], original["prediction"]),
            _errors(original["reference"], expected_text),
        ):
            raise ValueError("edit arithmetic mismatch")
        checked.append(repaired)
    words = sum(r["ref_words"] for r in checked)
    raw_errors = sum(r["raw_errors"] for r in checked)
    repaired_errors = sum(r["repaired_errors"] for r in checked)
    ci = _paired_ci(checked)
    if (raw_errors, words) != (source_summary["overall"]["word_errors"],
                               source_summary["overall"]["ref_words"]):
        raise ValueError("source summary disagrees with independent count")
    if (raw_errors, repaired_errors, words) != (summary["raw_word_errors"],
                                                summary["repaired_word_errors"],
                                                summary["reference_words"]):
        raise ValueError("repair summary disagrees with independent count")
    if not np.allclose(ci, summary["paired_speaker_bootstrap_95_ci_difference_points"]):
        raise ValueError("speaker-bootstrap difference changed")
    bracket_rows = bracket_tokens = clean_words = clean_raw_errors = clean_repaired_errors = 0
    for row in checked:
        tags = re.findall(r"\[[^\]]+\]", row["reference"])
        bracket_rows += bool(tags)
        bracket_tokens += len(tags)
        clean_reference = re.sub(r"\[[^\]]+\]", " ", row["reference"])
        clean_words += len(_words(clean_reference))
        clean_raw_errors += _errors(clean_reference, row["raw_prediction"])
        clean_repaired_errors += _errors(clean_reference, row["repaired_prediction"])
    scenario_slices = {}
    for name in sorted({r["scenario"] for r in checked}):
        group = [r for r in checked if r["scenario"] == name]
        group_words = sum(r["ref_words"] for r in group)
        group_raw = sum(r["raw_errors"] for r in group)
        group_repaired = sum(r["repaired_errors"] for r in group)
        scenario_slices[name] = {
            "clips": len(group), "speakers": len({r["sp_id"] for r in group}),
            "raw_wer_percent": 100 * group_raw / group_words,
            "repaired_wer_percent": 100 * group_repaired / group_words,
            "repaired_minus_raw_wer_points": 100 * (group_repaired - group_raw) / group_words,
            "clips_changed": sum(r["removed_words"] > 0 for r in group),
            "clips_harmed": sum(r["repaired_errors"] > r["raw_errors"] for r in group),
        }
    result = {
        "status": "validated_development_result" if split == "dev" else "validated_speaker_disjoint_result",
        "dataset": source_summary["dataset"],
        "dataset_revision": DATASET_REVISION,
        "model": source_summary["model"],
        "model_revision": MODEL_REVISION,
        "split": split,
        "clips": expected_clips,
        "speakers": expected_speakers,
        "speaker_overlap_with_development": 0 if split == "heldout" else None,
        "heldout_selection_sha256": SELECTION_SHA256 if split == "heldout" else None,
        "reference_words": words,
        "raw_word_errors": raw_errors,
        "repaired_word_errors": repaired_errors,
        "raw_wer_percent": 100 * raw_errors / words,
        "repaired_wer_percent": 100 * repaired_errors / words,
        "repaired_minus_raw_wer_points": 100 * (repaired_errors - raw_errors) / words,
        "paired_speaker_bootstrap_95_ci_difference_points": ci,
        "clips_changed": sum(r["removed_words"] > 0 for r in checked),
        "words_removed": sum(r["removed_words"] for r in checked),
        "clips_helped": sum(r["repaired_errors"] < r["raw_errors"] for r in checked),
        "clips_harmed": sum(r["repaired_errors"] > r["raw_errors"] for r in checked),
        "raw_utterances_over_100_percent_wer": sum(r["raw_errors"] > r["ref_words"] for r in checked),
        "repaired_utterances_over_100_percent_wer": sum(r["repaired_errors"] > r["ref_words"] for r in checked),
        "scenarios_descriptive_only": scenario_slices,
        "posthoc_bracket_tag_sensitivity": {
            "bracketed_reference_rows": bracket_rows,
            "bracketed_items": bracket_tokens,
            "reference_words_without_bracketed_items": clean_words,
            "raw_wer_percent": 100 * clean_raw_errors / clean_words,
            "repaired_wer_percent": 100 * clean_repaired_errors / clean_words,
            "repaired_minus_raw_wer_points": 100 * (clean_repaired_errors - clean_raw_errors) / clean_words,
            "interpretation": "Exploratory sensitivity only; the predeclared headline retains the literal verbatim reference.",
        },
        "policy_sha256": _sha(policy_path),
        "source_predictions_sha256": _sha(source_path),
        "repaired_predictions_sha256": _sha(repair_path),
        "source_run_id": SOURCE_PROVENANCE[split][0],
        "source_code_commit": SOURCE_PROVENANCE[split][1],
        "run_id": run_id,
        "code_commit": code_commit,
        "limits": [
            "This is a development sample used to select the repair threshold; it is not an unbiased improvement estimate."
            if split == "dev" else
            "The rule was frozen using only first-shard speakers before this evaluation; the model's own predictions are the only inference-time input.",
            "Raw WER keeps number words distinct from written digits and does not measure semantic number correctness.",
            "The rule suppresses only long adjacent single-word loops; it does not correct recognition or preserve natural repetitions of five or more copies.",
        ],
        "packaging_script_sha256": _sha(Path(__file__)),
    }
    output = ROOT / f"results/night1/lahaja_loop_repair_{split}_20261011.json"
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    return {"output": str(output), "clips": expected_clips,
            "raw_wer_percent": result["raw_wer_percent"],
            "repaired_wer_percent": result["repaired_wer_percent"],
            "paired_ci95_difference_points": ci}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=tuple(SPLITS), required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--code-commit", required=True)
    args = parser.parse_args()
    print(json.dumps(package(args.split, args.run_id, args.code_commit)))


if __name__ == "__main__":
    main()
