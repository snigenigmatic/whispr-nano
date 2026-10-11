"""Validate and package the local Lahaja pilot without publishing gated transcripts."""

from __future__ import annotations

from collections import defaultdict
import hashlib
import json
from pathlib import Path
import unicodedata

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
RUNS = Path.home() / ".local/share/openresearch"
OUT = ROOT / "results/night1/lahaja_pilot_20261011.json"
SEED = 20261011
RESAMPLES = 4000


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _words(value: str) -> list[str]:
    value = unicodedata.normalize("NFC", value).lower()
    return "".join(ch if unicodedata.category(ch)[0] in "LMN" else " " for ch in value).split()


def _load(folder: str) -> tuple[dict, list[dict], str]:
    base = RUNS / folder
    summary = json.loads((base / "summary.json").read_text())
    path = base / "predictions.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    if len(rows) != 104 or len({r["row_index"] for r in rows}) != 104:
        raise ValueError(f"incomplete or duplicate predictions in {folder}")
    if sum(r["word_errors"] for r in rows) != summary["overall"]["word_errors"]:
        raise ValueError(f"word edit mismatch in {folder}")
    if sum(r["ref_words"] for r in rows) != summary["overall"]["ref_words"]:
        raise ValueError(f"reference word mismatch in {folder}")
    return summary, rows, _sha(path)


def _speaker_ci(rows: list[dict]) -> list[float]:
    by_speaker = defaultdict(lambda: [0, 0])
    for row in rows:
        count = by_speaker[row["sp_id"]]
        count[0] += row["word_errors"]
        count[1] += row["ref_words"]
    rng = np.random.default_rng(SEED)
    speakers = sorted(by_speaker)
    rates = []
    for _ in range(RESAMPLES):
        sampled = rng.choice(speakers, size=len(speakers), replace=True)
        errors = sum(by_speaker[s][0] for s in sampled)
        words = sum(by_speaker[s][1] for s in sampled)
        rates.append(errors / words)
    return [float(x) for x in np.percentile(rates, [2.5, 97.5])]


def main() -> None:
    small, small_rows, small_sha = _load("lahaja-pilot-v1")
    turbo, turbo_rows, turbo_sha = _load("lahaja-turbo-104-v1")
    paired = list(zip(small_rows, turbo_rows, strict=True))
    if any(a["row_index"] != b["row_index"] or a["reference"] != b["reference"]
           or a["sp_id"] != b["sp_id"] for a, b in paired):
        raise ValueError("pilot systems did not evaluate identical released clips")
    if turbo["dataset_revision"] != "d4ffd2ecbdd933e37c917ddcf620eef159ceb3a7":
        raise ValueError("unexpected Lahaja dataset revision")
    if (len({r["sp_id"] for r in turbo_rows}),
        len({r["native_state"] for r in turbo_rows})) != (26, 11):
        raise ValueError("unexpected first-shard speaker/state coverage")
    if {scenario: sum(r["scenario"] == scenario for r in turbo_rows)
        for scenario in ("Read", "Extempore")} != {"Read": 52, "Extempore": 52}:
        raise ValueError("scenario sample counts changed")
    ci = _speaker_ci(turbo_rows)
    if not np.allclose(ci, turbo["overall"]["speaker_bootstrap_95_ci_wer"]):
        raise ValueError("speaker-bootstrap interval mismatch")

    result = {
        "scope": "First Lahaja Parquet shard, deterministic two clips per speaker per scenario; development pilot, not full-corpus or held-out model comparison",
        "dataset": turbo["dataset"],
        "dataset_revision": turbo["dataset_revision"],
        "sample": {"clips": 104, "speakers": 26, "states": 11,
                   "read_clips": 52, "extempore_clips": 52,
                   "audio_minutes_by_metadata": round(sum(r["duration"] for r in turbo_rows) / 60, 3)},
        "primary": {"model": turbo["model"], "model_revision": turbo["model_revision"],
                    "run_id": "33cdb67d-09b2-4b60-83b0-ad59d1cba90d",
                    "code_commit": "8671b2d9a61de73736abe1149dbf711b1b4f3a3d",
                    "predictions_sha256": turbo_sha,
                    "word_errors": turbo["overall"]["word_errors"],
                    "reference_words": turbo["overall"]["ref_words"],
                    "wer_percent": 100 * turbo["overall"]["wer"],
                    "speaker_bootstrap_95_ci_percent": [100 * x for x in ci],
                    "read_wer_percent": 100 * turbo["scenarios"]["Read"]["wer"],
                    "extempore_wer_percent": 100 * turbo["scenarios"]["Extempore"]["wer"],
                    "observed_quality": {
                        "generated_token_cap_256_count": sum(r["generated_tokens"] >= 256 for r in turbo_rows),
                        "hypothesis_at_least_3x_reference_words_count": sum(
                            len(_words(r["prediction"])) >= 3 * r["ref_words"]
                            for r in turbo_rows),
                        "utterance_wer_over_100_percent_count": sum(
                            r["word_errors"] > r["ref_words"] for r in turbo_rows),
                    },
                    "decoder_note": "The returned Whisper sequence in Transformers 5.16.1 omits EOS during postprocessing; the run summary's eos_missing field is invalid and is excluded here."},
        "diagnostic": {"model": small["model"], "model_revision": small["model_revision"],
                       "run_id": "85e94d4f-b9ea-4227-be40-cefbc9cb705c",
                       "predictions_sha256": small_sha,
                       "raw_wer_percent": 100 * small["overall"]["wer"],
                       "utterance_wer_over_100_percent_count": sum(
                           r["word_errors"] > r["ref_words"] for r in small_rows),
                       "hypothesis_at_least_3x_reference_words_count": sum(
                           len(_words(r["prediction"])) >= 3 * r["ref_words"]
                           for r in small_rows),
                       "status": "Decode-quality failure; do not treat this raw WER as a model capability estimate"},
        "scoring": turbo["scorer"],
        "limits": ["No number-word equivalence mapping; spoken numerals versus written digits inflate raw WER.",
                   "Pilot model choice followed a ten-clip QC check; this is not a held-out model comparison.",
                   "States with few speakers cannot support regional fairness conclusions."],
        "packaging_script_sha256": _sha(Path(__file__)),
    }
    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"output": str(OUT), "clips": 104,
                      "wer_percent": result["primary"]["wer_percent"],
                      "ci95_percent": result["primary"]["speaker_bootstrap_95_ci_percent"],
                      "qc": result["primary"]["observed_quality"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
