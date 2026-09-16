"""Fixed orx command: read one committed action and dispatch it safely."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from .contract import BudgetLedger, StudyContract, atomic_json, canonical_json_hash


ROOT = Path(__file__).resolve().parents[2]
RUN_PATH = ROOT / "research/publication/run.json"
CONTRACT_PATH = ROOT / "research/publication/contract.json"
BUDGET_PATH = ROOT / "research/publication/budget.json"


def run_inspection(config: dict) -> dict:
    from research.publication.modal_runner import app, inspect_spire_train

    ledger = BudgetLedger(BUDGET_PATH)
    config_hash = canonical_json_hash(config)
    ledger.reserve(config["run_id"], 0.10, config_hash)
    try:
        with app.run():
            result = inspect_spire_train.remote()
    except BaseException:
        # Keep the reservation. A retry must use a new immutable run ID.
        raise
    # Conservative CPU estimate; the reservation remains the controlling cap.
    measured = min(0.10, float(result["wall_seconds"]) * 0.0001)
    receipt_hash = canonical_json_hash(result)
    ledger.complete(config["run_id"], measured, receipt_hash)
    out = ROOT / "research/publication/data/spire_train_schema.json"
    atomic_json(out, {**result, "config_sha256": config_hash, "receipt_sha256": receipt_hash})
    return result


def main() -> int:
    contract = StudyContract.load(CONTRACT_PATH)
    config = json.loads(RUN_PATH.read_text(encoding="utf-8"))
    allowed = {"action", "run_id"}
    if set(config) != allowed or not all(isinstance(config[k], str) and config[k] for k in allowed):
        raise SystemExit("run.json must contain exactly nonempty action and run_id")
    print(f"contract_sha256: {contract.sha256}", flush=True)
    print(f"run_config: {json.dumps(config, sort_keys=True)}", flush=True)
    if config["action"] == "inspect_spire_train":
        result = run_inspection(config)
    else:
        raise SystemExit(f"unknown publication action: {config['action']}")
    print("===== RESULT SUMMARY =====", flush=True)
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    print("==========================", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
