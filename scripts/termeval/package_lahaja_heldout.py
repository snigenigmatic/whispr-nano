"""Validate and publish aggregate Lahaja speaker-disjoint evidence only."""

from __future__ import annotations

from collections import defaultdict
import hashlib
import json
from pathlib import Path
import unicodedata

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
RUNS = Path.home() / ".local/share/openresearch"
DEV_PRED = RUNS / "lahaja-turbo-104-v1/predictions.jsonl"
HELDOUT = RUNS / "lahaja-turbo-heldout-405-v1"
OUT = ROOT / "results/night1/lahaja_heldout_20261011.json"
SEED = 20261011
RESAMPLES = 4000


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _words(value: str) -> list[str]:
    value = unicodedata.normalize("NFC", value).lower()
    return "".join(ch if unicodedata.category(ch)[0] in "LMN" else " " for ch in value).split()


def _speaker_ci(rows: list[dict]) -> list[float]:
    counts = defaultdict(lambda: [0, 0])
    for row in rows:
        counts[row["sp_id"]][0] += row["word_errors"]
        counts[row["sp_id"]][1] += row["ref_words"]
    ids = sorted(counts)
    rng = np.random.default_rng(SEED)
    rates = []
    for _ in range(RESAMPLES):
        sampled = rng.choice(ids, size=len(ids), replace=True)
        rates.append(sum(counts[s][0] for s in sampled) /
                     sum(counts[s][1] for s in sampled))
    return [float(x) for x in np.percentile(rates, [2.5, 97.5])]


def main() -> None:
    dev = [json.loads(line) for line in DEV_PRED.read_text().splitlines()]
    summary = json.loads((HELDOUT / "summary.json").read_text())
    predictions = HELDOUT / "predictions.jsonl"
    rows = [json.loads(line) for line in predictions.read_text().splitlines()]
    dev_speakers = {r["sp_id"] for r in dev}
    holdout_speakers = {r["sp_id"] for r in rows}
    if len(dev_speakers) != 26 or len(rows) != 405 or len(holdout_speakers) != 106:
        raise ValueError("development/held-out sample count changed")
    if dev_speakers & holdout_speakers:
        raise ValueError("speaker overlap between first-shard pilot and held-out extension")
    if len({(r["data_file"], r["row_index"]) for r in rows}) != 405:
        raise ValueError("duplicate held-out Parquet row")
    if len({r["native_state"] for r in rows}) != 17:
        raise ValueError("expected 17 released state labels")
    if (sum(r["word_errors"] for r in rows), sum(r["ref_words"] for r in rows)) != (
        summary["overall"]["word_errors"], summary["overall"]["ref_words"]
    ):
        raise ValueError("summary and prediction edit counts disagree")
    ci = _speaker_ci(rows)
    if not np.allclose(ci, summary["overall"]["speaker_bootstrap_95_ci_wer"]):
        raise ValueError("speaker-bootstrap interval mismatch")
    if summary["selection_sha256"] != "6134e2a9eabd65f590bf4f7f7829626e520379a66e058d9dd51ce4acf36c4060":
        raise ValueError("held-out selection hash changed")

    result = {
        "status": "speaker_disjoint_open_model_baseline_complete",
        "dataset": summary["dataset"],
        "dataset_revision": summary["dataset_revision"],
        "selection": summary["sample_policy"],
        "selection_sha256": summary["selection_sha256"],
        "development_speakers": 26,
        "heldout_speakers": 106,
        "speaker_overlap": 0,
        "heldout_clips": 405,
        "heldout_states": 17,
        "heldout_audio_minutes_by_metadata": round(sum(r["duration"] for r in rows) / 60, 3),
        "model": summary["model"],
        "model_revision": summary["model_revision"],
        "run_id": "48514944-d432-4179-82ee-89eb054ee122",
        "code_commit": "fc90d23b5f4425650d881e46db11eea58e02ae01",
        "predictions_sha256": _sha(predictions),
        "word_errors": summary["overall"]["word_errors"],
        "reference_words": summary["overall"]["ref_words"],
        "wer_percent": 100 * summary["overall"]["wer"],
        "speaker_bootstrap_95_ci_percent": [100 * x for x in ci],
        "scenarios": {name: {"clips": value["clips"], "speakers": value["speakers"],
                            "word_errors": value["word_errors"],
                            "reference_words": value["ref_words"],
                            "wer_percent": 100 * value["wer"]}
                      for name, value in summary["scenarios"].items()},
        "observed_decoder_quality": {
            "generated_token_cap_256_count": sum(r["generated_tokens"] >= 256 for r in rows),
            "hypothesis_at_least_3x_reference_words_count": sum(
                len(_words(r["prediction"])) >= 3 * r["ref_words"] for r in rows),
            "utterance_wer_over_100_percent_count": sum(
                r["word_errors"] > r["ref_words"] for r in rows),
        },
        "scoring": summary["scorer"],
        "limits": ["The model/decoder recipe was frozen after a first-shard QC pilot; this score tests new speakers, not new corpus collection conditions.",
                   "No number-word equivalence mapping; written digits versus spoken words inflate raw WER.",
                   "Regional slices require adequate speaker counts and content-matched comparisons.",
                   "The full 6,152 clips were not decoded; this is a speaker-balanced sample."],
        "packaging_script_sha256": _sha(Path(__file__)),
    }
    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"output": str(OUT), "clips": 405,
                      "speaker_overlap": 0, "wer_percent": result["wer_percent"],
                      "ci95_percent": result["speaker_bootstrap_95_ci_percent"],
                      "qc": result["observed_decoder_quality"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
