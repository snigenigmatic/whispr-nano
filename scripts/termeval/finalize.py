"""Download, verify, and analyze the completed night-1 Modal decode.

Run from the repository root after all three full GPU functions finish:
    .venv/bin/python -m scripts.termeval.finalize
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/night1"
MANIFEST = ROOT / "data/termeval/manifest.jsonl"
VOLUME = "night1-term-results-v1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run(*args: str) -> None:
    subprocess.run(args, check=True, cwd=ROOT)


def get(dataset: str, condition: str, remote_name: str, local_name: str) -> None:
    run(str(ROOT / ".venv/bin/modal"), "volume", "get", VOLUME,
        f"/{dataset}_{condition}/{remote_name}", str(OUT / local_name), "--force")


def verify_decode() -> dict:
    launch = json.loads((OUT / "decode_launch.json").read_text())
    if sha256(MANIFEST) != launch["manifest_sha256"]:
        raise ValueError("local manifest changed after decode launch")
    overlap = json.loads((OUT / "speaker_holdout_overlap.json").read_text())
    if (overlap["manifest_sha256"] != sha256(MANIFEST)
            or overlap["split_sha256"] != sha256(OUT / "speaker_holdout_split.json")):
        raise ValueError("speaker-heldout overlap audit inputs changed")
    expected_ids = {dataset: set() for dataset in ("hiacc", "mucs_hien")}
    for row in map(json.loads, MANIFEST.open()):
        expected_ids[row["dataset"]].add(row["utt_id"])
    policy = json.loads((OUT / "decode_policy.json").read_text())
    specs = [
        ("hiacc", policy["primary"]["hiacc"]),
        ("mucs_hien", policy["secondary"]["mucs_hien_auto_language"]),
        ("mucs_hien", policy["primary"]["mucs_hien"]),
    ]
    verified = {}
    for dataset, spec in specs:
        condition = spec["condition_id"]
        get(dataset, condition, "config.json", spec["config_file"])
        get(dataset, condition, "receipt.json", spec["receipt_file"])
        get(dataset, condition, "predictions.jsonl", spec["prediction_file"])
        config = json.loads((OUT / spec["config_file"]).read_text())
        receipt = json.loads((OUT / spec["receipt_file"]).read_text())
        predictions = [json.loads(line) for line in (OUT / spec["prediction_file"]).open()]
        config_hash = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
        if (config["manifest_sha256"] != launch["manifest_sha256"]
                or config["model_revision"] != launch["model_revision"]
                or config["condition_id"] != condition):
            raise ValueError(f"{dataset}: config differs from frozen launch")
        if (not receipt["complete"] or receipt["target"] != len(expected_ids[dataset])
                or receipt["total_cached"] != len(expected_ids[dataset])
                or len(predictions) != len(expected_ids[dataset])
                or {p["utt_id"] for p in predictions} != expected_ids[dataset]
                or {p["config_hash"] for p in predictions} != {config_hash}
                or {p["condition_id"] for p in predictions} != {condition}):
            raise ValueError(f"{dataset}: incomplete or inconsistent decode")
        verified[f"{dataset}:{condition}"] = {
            "predictions": len(predictions), "config_hash": config_hash,
            "predictions_sha256": sha256(OUT / spec["prediction_file"]),
            "receipt_sha256": sha256(OUT / spec["receipt_file"]),
        }
    return verified


def main() -> None:
    verified = verify_decode()
    for module in ("validate_predictions", "score_results", "evaluate_heldout_lexicon",
                   "compare_mucs_decode"):
        run(sys.executable, "-m", f"scripts.termeval.{module}")
    run(sys.executable, str(OUT / "figs/plot_pilot.py"))
    run(sys.executable, "-m", "scripts.termeval.build_report")
    run(sys.executable, "-m", "pytest", "-q", "tests/termeval")
    outputs = ["metrics.csv", "score_receipt.json", "heldout_lexicon_metrics.csv",
               "heldout_lexicon_receipt.json", "prediction_qc.json",
               "mucs_full_language_comparison.json", "RUN_REPORT.md"]
    (OUT / "completion_check.json").write_text(json.dumps({
        "decode": verified,
        "decode_policy_sha256": sha256(OUT / "decode_policy.json"),
        "outputs_sha256": {name: sha256(OUT / name) for name in outputs},
        "tests": "pytest -q tests/termeval passed",
    }, indent=2))
    print(json.dumps({"event": "night1_finalized", "decode": verified,
                      "completion_check": str(OUT / "completion_check.json")}))


if __name__ == "__main__":
    main()
