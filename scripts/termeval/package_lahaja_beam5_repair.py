"""Independently validate beam5 plus frozen word-loop repair; share aggregates."""

from __future__ import annotations

from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sys
import unicodedata

import jiwer
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from publication_study.loop_repair import suppress_adjacent_word_loops  # noqa: E402


RUNS = Path.home() / ".local/share/openresearch"
SOURCE = RUNS / "lahaja-small-beam5-heldout-405-v1"
REPAIR = RUNS / "lahaja-beam5-loop-repair-heldout-v1"
DEV = RUNS / "lahaja-pilot-v1"
OUT = ROOT / "results/night1/lahaja_beam5_loop_repair_heldout_20261013.json"
DATASET_REVISION = "d4ffd2ecbdd933e37c917ddcf620eef159ceb3a7"
MODEL_REVISION = "973afd24965f72e36ca33b3055d56a652f456b4d"
SELECTION_SHA256 = "6134e2a9eabd65f590bf4f7f7829626e520379a66e058d9dd51ce4acf36c4060"
POLICY_SHA256 = "4a89775ecae07364dd69a661e1ac612a3ffd10c7991f76305b45ae48c8873867"
SEED = 20261011
RESAMPLES = 4000


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _key(row: dict) -> tuple[str, int]:
    return row["data_file"], row["row_index"]


def _words(value: str) -> list[str]:
    value = unicodedata.normalize("NFC", value).lower()
    return "".join(ch if unicodedata.category(ch)[0] in "LMN" else " " for ch in value).split()


def _errors(reference: str, hypothesis: str) -> int:
    counts = jiwer.process_words(" ".join(_words(reference)), " ".join(_words(hypothesis)))
    return counts.substitutions + counts.deletions + counts.insertions


def _paired_ci(rows: list[dict]) -> list[float]:
    counts = defaultdict(lambda: [0, 0, 0])
    for row in rows:
        bucket = counts[row["sp_id"]]
        bucket[0] += row["raw_errors"]
        bucket[1] += row["repaired_errors"]
        bucket[2] += row["ref_words"]
    speakers = sorted(counts)
    rng = np.random.default_rng(SEED)
    values = []
    for _ in range(RESAMPLES):
        drawn = rng.choice(speakers, size=len(speakers), replace=True)
        values.append(100 * sum(counts[s][1] - counts[s][0] for s in drawn) /
                      sum(counts[s][2] for s in drawn))
    return [float(v) for v in np.percentile(values, [2.5, 97.5])]


def main() -> None:
    policy_path = ROOT / "src/publication_study/loop_repair.py"
    if _sha(policy_path) != POLICY_SHA256:
        raise ValueError("frozen policy bytes changed")
    source_path = SOURCE / "predictions.jsonl"
    repaired_path = REPAIR / "predictions.jsonl"
    source_summary = json.loads((SOURCE / "summary.json").read_text())
    repair_summary = json.loads((REPAIR / "summary.json").read_text())
    source = [json.loads(line) for line in source_path.read_text().splitlines()]
    repaired = [json.loads(line) for line in repaired_path.read_text().splitlines()]
    if len(source) != 405 or len(repaired) != 405:
        raise ValueError("incomplete held-out prediction set")
    source_by_id = {_key(r): r for r in source}
    repair_by_id = {_key(r): r for r in repaired}
    if len(source_by_id) != 405 or len(repair_by_id) != 405 or set(source_by_id) != set(repair_by_id):
        raise ValueError("duplicate or non-paired row IDs")
    if len({r["sp_id"] for r in source}) != 106:
        raise ValueError("held-out speaker count changed")
    dev = [json.loads(line) for line in (DEV / "predictions.jsonl").read_text().splitlines()]
    if {r["sp_id"] for r in source} & {r["sp_id"] for r in dev}:
        raise ValueError("development speaker overlap")
    for name, value in {
        "dataset_revision": DATASET_REVISION,
        "model_revision": MODEL_REVISION,
        "selection_sha256": SELECTION_SHA256,
        "num_beams": 5,
    }.items():
        if source_summary[name] != value:
            raise ValueError(f"source {name} changed")
    if repair_summary["source_predictions_sha256"] != _sha(source_path):
        raise ValueError("source prediction hash changed")
    if repair_summary["predictions_sha256"] != _sha(repaired_path):
        raise ValueError("repaired prediction hash changed")
    if repair_summary["policy_sha256"] != POLICY_SHA256:
        raise ValueError("policy hash changed")
    checked = []
    for key in sorted(source_by_id):
        before, after = source_by_id[key], repair_by_id[key]
        expected_text, removed = suppress_adjacent_word_loops(before["prediction"])
        if (after["reference"], after["raw_prediction"], after["sp_id"], after["scenario"]) != (
            before["reference"], before["prediction"], before["sp_id"], before["scenario"]
        ):
            raise ValueError(f"non-paired reference, hypothesis or speaker: {key}")
        if (after["repaired_prediction"], after["removed_words"]) != (expected_text, removed):
            raise ValueError(f"not the frozen rule: {key}")
        if after["ref_words"] != len(_words(before["reference"])):
            raise ValueError(f"reference word count changed: {key}")
        if (after["raw_errors"], after["repaired_errors"]) != (
            _errors(before["reference"], before["prediction"]),
            _errors(before["reference"], expected_text),
        ):
            raise ValueError(f"edit arithmetic changed: {key}")
        if after["raw_errors"] != before["word_errors"]:
            raise ValueError(f"raw prediction score mismatch: {key}")
        checked.append(after)
    words = sum(r["ref_words"] for r in checked)
    raw_errors = sum(r["raw_errors"] for r in checked)
    repaired_errors = sum(r["repaired_errors"] for r in checked)
    if (words, raw_errors, repaired_errors) != (7339, 5752, 4761):
        raise ValueError("unexpected pooled counts")
    if (repair_summary["reference_words"], repair_summary["raw_word_errors"],
        repair_summary["repaired_word_errors"]) != (words, raw_errors, repaired_errors):
        raise ValueError("run summary disagrees with independent score")
    ci = _paired_ci(checked)
    if not np.allclose(ci, repair_summary["paired_speaker_bootstrap_95_ci_difference_points"]):
        raise ValueError("speaker-bootstrap interval changed")
    result = {
        "status": "validated_exploratory_speaker_disjoint_result",
        "dataset": "ai4bharat/Lahaja",
        "dataset_revision": DATASET_REVISION,
        "model": "openai/whisper-small",
        "model_revision": MODEL_REVISION,
        "selection_sha256": SELECTION_SHA256,
        "clips": 405,
        "speakers": 106,
        "speaker_overlap_with_development": 0,
        "reference_words": words,
        "beam5_word_errors": raw_errors,
        "beam5_wer_percent": 100 * raw_errors / words,
        "beam5_plus_repair_word_errors": repaired_errors,
        "beam5_plus_repair_wer_percent": 100 * repaired_errors / words,
        "difference_wer_points": 100 * (repaired_errors - raw_errors) / words,
        "paired_speaker_bootstrap_95_ci_difference_points": ci,
        "bootstrap_resamples": RESAMPLES,
        "bootstrap_seed": SEED,
        "clips_changed": sum(r["removed_words"] > 0 for r in checked),
        "clips_helped": sum(r["repaired_errors"] < r["raw_errors"] for r in checked),
        "clips_harmed": sum(r["repaired_errors"] > r["raw_errors"] for r in checked),
        "words_removed": sum(r["removed_words"] for r in checked),
        "policy_sha256": POLICY_SHA256,
        "source_predictions_sha256": _sha(source_path),
        "repaired_predictions_sha256": _sha(repaired_path),
        "source_run_id": "36217bd3-5caa-48ea-8521-0660ca15d7f0",
        "run_id": "9f3eb87b-7716-44c0-a2fb-3574b963a0ca",
        "code_commit": "fc83ff5d02c24977f218f2e48f6e522379bec27f",
        "limits": [
            "The five-beam-plus-repair combination was proposed after inspecting the held-out five-beam aggregate; treat this as exploratory, not a predeclared confirmatory test.",
            "The exact repair rule was frozen on different first-shard speakers and was not retuned for beam5.",
            "A positive control shows the text-only rule can erase genuinely spoken repetition; it is not a generally safe ASR correction.",
            "Raw WER counts written digits and spoken number words separately and does not measure number-value correctness.",
            "Lahaja is Hindi, so this does not establish an Indian-English or Hinglish improvement.",
            "Gated Lahaja references and raw predictions remain local; this file contains only aggregates and hashes.",
        ],
        "packaging_script_sha256": _sha(Path(__file__)),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"output": str(OUT), "wer_percent": result["beam5_plus_repair_wer_percent"],
                      "difference_points": result["difference_wer_points"]}))


if __name__ == "__main__":
    main()
