"""Validate the complete held-out Lahaja beam control and publish aggregates only."""

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
GREEDY = RUNS / "lahaja-small-heldout-405-v1"
BEAM = RUNS / "lahaja-small-beam5-heldout-405-v1"
REPAIR = RUNS / "lahaja-loop-repair-heldout-v1"
OUT = ROOT / "results/night1/lahaja_small_beam5_heldout_20261013.json"
DATASET_REVISION = "d4ffd2ecbdd933e37c917ddcf620eef159ceb3a7"
MODEL_REVISION = "973afd24965f72e36ca33b3055d56a652f456b4d"
SELECTION_SHA256 = "6134e2a9eabd65f590bf4f7f7829626e520379a66e058d9dd51ce4acf36c4060"
SEED = 20261011
RESAMPLES = 4000


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _words(value: str) -> list[str]:
    value = unicodedata.normalize("NFC", value).lower()
    return "".join(ch if unicodedata.category(ch)[0] in "LMN" else " " for ch in value).split()


def _errors(reference: str, prediction: str) -> tuple[int, int, int]:
    edits = jiwer.process_words(" ".join(_words(reference)), " ".join(_words(prediction)))
    return edits.substitutions, edits.deletions, edits.insertions


def _key(row: dict) -> tuple[str, int]:
    return row["data_file"], row["row_index"]


def _load(folder: Path, beams: int) -> tuple[dict, dict[tuple[str, int], dict]]:
    summary = json.loads((folder / "summary.json").read_text())
    rows = [json.loads(line) for line in (folder / "predictions.jsonl").read_text().splitlines()]
    if len(rows) != 405 or len({_key(r) for r in rows}) != 405:
        raise ValueError(f"incomplete or duplicate predictions: {folder}")
    if len({r["sp_id"] for r in rows}) != 106:
        raise ValueError(f"speaker count changed: {folder}")
    expected = {
        "dataset_revision": DATASET_REVISION,
        "model": "openai/whisper-small",
        "model_revision": MODEL_REVISION,
        "selection_sha256": SELECTION_SHA256,
        "num_beams": beams,
        "device": "mps",
        "language": "hi",
        "task": "transcribe",
    }
    for key, value in expected.items():
        if summary[key] != value:
            raise ValueError(f"{folder}: {key} changed")
    for row in rows:
        s, d, i = _errors(row["reference"], row["prediction"])
        if (s, d, i) != (row["substitutions"], row["deletions"], row["insertions"]):
            raise ValueError(f"edit counts changed: {folder} {_key(row)}")
        if row["word_errors"] != s + d + i or row["ref_words"] != len(_words(row["reference"])):
            raise ValueError(f"word counts changed: {folder} {_key(row)}")
    words = sum(r["ref_words"] for r in rows)
    errors = sum(r["word_errors"] for r in rows)
    if (summary["overall"]["word_errors"], summary["overall"]["ref_words"]) != (errors, words):
        raise ValueError(f"summary arithmetic changed: {folder}")
    return summary, {_key(r): r for r in rows}


def _paired_ci(before: dict, after: dict) -> list[float]:
    counts = defaultdict(lambda: [0, 0, 0])
    for key in sorted(before):
        a, b = before[key], after[key]
        bucket = counts[a["sp_id"]]
        bucket[0] += a["word_errors"]
        bucket[1] += b["word_errors"]
        bucket[2] += a["ref_words"]
    speakers = sorted(counts)
    rng = np.random.default_rng(SEED)
    samples = []
    for _ in range(RESAMPLES):
        draw = rng.choice(speakers, size=len(speakers), replace=True)
        samples.append(100 * (sum(counts[s][1] - counts[s][0] for s in draw) /
                              sum(counts[s][2] for s in draw)))
    return [float(x) for x in np.percentile(samples, [2.5, 97.5])]


def main() -> None:
    greedy_summary, greedy = _load(GREEDY, 1)
    beam_summary, beam = _load(BEAM, 5)
    if set(greedy) != set(beam):
        raise ValueError("beam run changed held-out row identities")
    if greedy_summary["overall"]["ref_words"] != 7339:
        raise ValueError("reference word count changed")
    dev_rows = [json.loads(line) for line in (RUNS / "lahaja-pilot-v1/predictions.jsonl").read_text().splitlines()]
    if {r["sp_id"] for r in dev_rows} & {r["sp_id"] for r in beam.values()}:
        raise ValueError("development speakers leaked into held-out sample")
    for key in greedy:
        a, b = greedy[key], beam[key]
        for field in ("reference", "ref_words", "sp_id", "fname", "scenario", "native_state", "duration"):
            if a[field] != b[field]:
                raise ValueError(f"non-paired {field} at {key}")
    repair_summary = json.loads((REPAIR / "summary.json").read_text())
    repair_rows = [json.loads(line) for line in (REPAIR / "predictions.jsonl").read_text().splitlines()]
    repair = {_key(r): r for r in repair_rows}
    if len(repair) != 405 or set(repair) != set(greedy):
        raise ValueError("repair run changed held-out row identities")
    if repair_summary["source_predictions_sha256"] != _sha(GREEDY / "predictions.jsonl"):
        raise ValueError("repair run uses different greedy source")
    repair_as_rows = {}
    for key in greedy:
        source, row = greedy[key], repair[key]
        if (row["reference"], row["raw_prediction"], row["sp_id"]) != (
            source["reference"], source["prediction"], source["sp_id"]
        ):
            raise ValueError(f"repair is not paired at {key}")
        if row["raw_errors"] != source["word_errors"] or row["repaired_errors"] != sum(
            _errors(row["reference"], row["repaired_prediction"])
        ):
            raise ValueError(f"repair counts changed at {key}")
        repair_as_rows[key] = {**source, "word_errors": row["repaired_errors"]}
    words = 7339
    greedy_errors = sum(r["word_errors"] for r in greedy.values())
    beam_errors = sum(r["word_errors"] for r in beam.values())
    repaired_errors = sum(r["word_errors"] for r in repair_as_rows.values())
    if (greedy_errors, beam_errors, repaired_errors) != (8787, 5752, 5709):
        raise ValueError("unexpected pooled errors")
    result = {
        "status": "validated_speaker_disjoint_decoder_control",
        "dataset": "ai4bharat/Lahaja",
        "dataset_revision": DATASET_REVISION,
        "model": "openai/whisper-small",
        "model_revision": MODEL_REVISION,
        "selection_sha256": SELECTION_SHA256,
        "clips": 405,
        "speakers": 106,
        "speaker_overlap_with_development": 0,
        "reference_words": words,
        "conditions": {
            "greedy": {"word_errors": greedy_errors, "wer_percent": 100 * greedy_errors / words,
                       "generation_cap_count": greedy_summary["decoder_quality"]["generated_token_cap_256_count"],
                       "wall_seconds": greedy_summary["elapsed_s"],
                       "predictions_sha256": _sha(GREEDY / "predictions.jsonl"),
                       "run_id": "b01c023b-d98f-4b28-9338-6655f95a82b4"},
            "beam5": {"word_errors": beam_errors, "wer_percent": 100 * beam_errors / words,
                      "generation_cap_count": beam_summary["decoder_quality"]["generated_token_cap_256_count"],
                      "wall_seconds": beam_summary["elapsed_s"],
                      "predictions_sha256": _sha(BEAM / "predictions.jsonl"),
                      "run_id": "36217bd3-5caa-48ea-8521-0660ca15d7f0",
                      "code_commit": "5f5915046463ae14cd83ec7f6c06f84310b14953"},
            "greedy_plus_frozen_loop_repair": {"word_errors": repaired_errors,
                "wer_percent": 100 * repaired_errors / words,
                "predictions_sha256": _sha(REPAIR / "predictions.jsonl"),
                "run_id": "4c1caf10-3070-4f27-9711-b8ab152fd549"},
        },
        "beam5_minus_greedy_wer_points": 100 * (beam_errors - greedy_errors) / words,
        "beam5_minus_greedy_speaker_bootstrap_95_ci_points": _paired_ci(greedy, beam),
        "beam5_minus_repair_wer_points": 100 * (beam_errors - repaired_errors) / words,
        "beam5_minus_repair_speaker_bootstrap_95_ci_points": _paired_ci(repair_as_rows, beam),
        "bootstrap_seed": SEED,
        "bootstrap_resamples": RESAMPLES,
        "beam5_clips_better_than_repair": sum(beam[k]["word_errors"] < repair_as_rows[k]["word_errors"] for k in beam),
        "beam5_clips_worse_than_repair": sum(beam[k]["word_errors"] > repair_as_rows[k]["word_errors"] for k in beam),
        "beam5_clips_tied_with_repair": sum(beam[k]["word_errors"] == repair_as_rows[k]["word_errors"] for k in beam),
        "limits": [
            "The small-model decoder was selected after first-shard development; this frozen 405-clip speaker-disjoint comparison is a secondary control.",
            "Wall times include setup and run on one Mac MPS GPU and are not a controlled latency benchmark.",
            "The text-only loop rule is unsafe for genuinely repeated speech; a separate positive control documents this failure.",
            "Raw WER keeps written digits distinct from spoken number words.",
            "Only aggregates are shared because Lahaja access is gated; raw predictions and references remain local.",
        ],
        "packaging_script_sha256": _sha(Path(__file__)),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"output": str(OUT), "beam5_wer_percent": result["conditions"]["beam5"]["wer_percent"],
                      "beam5_minus_repair_wer_points": result["beam5_minus_repair_wer_points"]}))


if __name__ == "__main__":
    main()
