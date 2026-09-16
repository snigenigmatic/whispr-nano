"""Fixed orx command: read one committed action and dispatch it safely."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from .contract import BudgetLedger, StudyContract, atomic_json, canonical_json_hash


ROOT = Path(__file__).resolve().parents[2]
RUN_PATH = ROOT / "research/publication/run.json"
CONTRACT_PATH = ROOT / "research/publication/contract.json"
LEDGER_ROOT = Path.home() / ".local/share/openresearch/whispr-nano-publication-study"
BUDGET_PATH = Path(os.environ.get("WHISPR_PUBLICATION_BUDGET", LEDGER_ROOT / "budget.json"))


def ensure_live_budget() -> None:
    if not BUDGET_PATH.exists():
        template = json.loads((ROOT / "research/publication/budget.json").read_text(encoding="utf-8"))
        atomic_json(BUDGET_PATH, template)


def run_inspection(config: dict) -> dict:
    from research.publication.modal_runner import app, inspect_spire_train

    ensure_live_budget()
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


def _load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _feature_receipts() -> dict[str, dict]:
    receipts: dict[str, dict] = {}
    for path in sorted((ROOT / "research/pilot/feature_receipts").glob("*.json")):
        for row in json.loads(path.read_text(encoding="utf-8"))["files"]:
            if row["utt_id"] in receipts:
                raise ValueError(f"duplicate feature receipt: {row['utt_id']}")
            receipts[row["utt_id"]] = row
    return receipts


def source_smoke_plan(config: dict, contract: StudyContract) -> dict:
    pseudo = _load_jsonl(ROOT / "research/pilot/teacher_predictions/all_train.jsonl")
    if len(pseudo) != 9495:
        raise ValueError("pinned source pseudo-label count changed")
    # The source corpus is labeled, but this arm must expose only teacher text
    # to the CE objective. Keep source references out of the rows sent to Modal.
    train = []
    for row in pseudo:
        transformed = dict(row)
        transformed["reference_text"] = transformed.pop("hypothesis_text")
        transformed["label_source"] = "teacher_pseudo"
        train.append(transformed)
    dev = _load_jsonl(ROOT / "research/pilot/data/smoke_dev_ledger_manifest.jsonl")
    receipts = _feature_receipts()
    all_rows = train + dev
    missing = sorted({row["utt_id"] for row in all_rows} - set(receipts))
    if missing:
        raise ValueError(f"missing feature receipts: {missing[:5]}")
    training_path = ROOT / "research/pilot/training.py"
    import hashlib

    return {
        "run_id": config["run_id"],
        "output_path": f"seed12/{config['run_id']}",
        "training_code_sha256": hashlib.sha256(training_path.read_bytes()).hexdigest(),
        "rows": all_rows,
        "cache_receipts": {row["utt_id"]: receipts[row["utt_id"]] for row in all_rows},
        "config": {
            "arm_name": "smoke_source_pseudo",
            "seed": 12,
            "num_updates": 100,
            "batch_size": contract.batch_size,
            "grad_accum_steps": 1,
            "learning_rate": contract.learning_rate,
            "weight_decay": 0.01,
            "max_grad_norm": 1.0,
            "eval_every": 0,
            "eval_batch_size": 8,
            "max_new_tokens": 224,
            "model_id": contract.student_model,
            "model_revision": contract.student_revision,
        },
    }


def run_source_smoke(config: dict, contract: StudyContract) -> dict:
    from research.publication.modal_runner import app, train_source_smoke

    ensure_live_budget()
    ledger = BudgetLedger(BUDGET_PATH)
    plan = source_smoke_plan(config, contract)
    plan_hash = canonical_json_hash(plan)
    ledger.reserve(config["run_id"], 0.50, plan_hash)
    try:
        with app.run():
            result = train_source_smoke.remote(plan)
    except BaseException:
        raise
    measured = float(result["estimated_resource_cost_usd"])
    receipt_hash = canonical_json_hash(result)
    ledger.complete(config["run_id"], measured, receipt_hash)
    atomic_json(
        ROOT / f"research/publication/results/{config['run_id']}/receipt.json",
        {**result, "plan_sha256": plan_hash, "receipt_sha256": receipt_hash},
    )
    return result


def source_endpoint_plan(config: dict, contract: StudyContract, label_source: str) -> dict:
    manifest = _load_jsonl(ROOT / "research/pilot/data/manifest.jsonl")
    dev = [row for row in manifest if row["split"] == "dev"]
    if len(dev) != 974:
        raise ValueError("pinned source development count changed")
    if label_source == "source_real":
        train = [row for row in manifest if row["split"] == "train"]
    elif label_source == "source_pseudo":
        train = []
        for row in _load_jsonl(ROOT / "research/pilot/teacher_predictions/all_train.jsonl"):
            transformed = dict(row)
            transformed["reference_text"] = transformed.pop("hypothesis_text")
            transformed["label_source"] = "teacher_pseudo"
            train.append(transformed)
    else:
        raise ValueError(f"unknown source endpoint: {label_source}")
    if len(train) != 9495:
        raise ValueError("pinned source training count changed")
    all_rows = train + dev
    receipts = _feature_receipts()
    missing = sorted({row["utt_id"] for row in all_rows} - set(receipts))
    if missing:
        raise ValueError(f"missing source feature receipts: {missing[:5]}")
    import hashlib

    return {
        "run_id": config["run_id"] + "/" + label_source,
        "label_source": label_source,
        "output_path": f"seed12/{label_source}",
        "training_code_sha256": hashlib.sha256(
            (ROOT / "research/pilot/training.py").read_bytes()
        ).hexdigest(),
        "rows": all_rows,
        "cache_receipts": {row["utt_id"]: receipts[row["utt_id"]] for row in all_rows},
        "config": {
            "arm_name": label_source, "seed": 12, "num_updates": contract.updates,
            "batch_size": contract.batch_size, "grad_accum_steps": 1,
            "learning_rate": contract.learning_rate, "weight_decay": 0.01,
            "max_grad_norm": 1.0, "eval_every": 0, "eval_batch_size": 8,
            "max_new_tokens": 224, "model_id": contract.student_model,
            "model_revision": contract.student_revision,
        },
    }


def run_source_endpoints(config: dict, contract: StudyContract) -> dict:
    from research.publication.modal_runner import app, train_source_endpoint

    ensure_live_budget()
    ledger = BudgetLedger(BUDGET_PATH)
    plans = [source_endpoint_plan(config, contract, arm) for arm in ("source_real", "source_pseudo")]
    hashes = {plan["label_source"]: canonical_json_hash(plan) for plan in plans}
    for plan in plans:
        ledger.reserve(plan["run_id"], 0.65, hashes[plan["label_source"]])
    output = []
    with app.run():
        for result in train_source_endpoint.map(plans, order_outputs=False, return_exceptions=True):
            if isinstance(result, BaseException):
                raise result
            arm = result["label_source"]
            receipt_hash = canonical_json_hash(result)
            ledger.complete(result["run_id"], float(result["estimated_resource_cost_usd"]), receipt_hash)
            atomic_json(
                ROOT / f"research/publication/results/{arm}/receipt.json",
                {**result, "plan_sha256": hashes[arm], "receipt_sha256": receipt_hash},
            )
            output.append(result)
    if {row["label_source"] for row in output} != {"source_real", "source_pseudo"}:
        raise RuntimeError("source endpoint batch incomplete")
    return {"runs": sorted(output, key=lambda row: row["label_source"])}


def run_target_preparation(config: dict, contract: StudyContract) -> dict:
    from research.publication.modal_runner import app, prepare_target_manifest

    ensure_live_budget()
    ledger = BudgetLedger(BUDGET_PATH)
    plan = {"run_id": config["run_id"], "seed": contract.seed, "per_accent": 800}
    plan_hash = canonical_json_hash(plan)
    ledger.reserve(config["run_id"], 0.50, plan_hash)
    with app.run():
        result = prepare_target_manifest.remote(plan)
    manifest_text = result.pop("manifest_text")
    if len(manifest_text.splitlines()) != contract.target_examples:
        raise RuntimeError("target manifest row count differs from contract")
    if result["manifest_sha256"] != __import__("hashlib").sha256(manifest_text.encode()).hexdigest():
        raise RuntimeError("target manifest transfer hash mismatch")
    receipt_hash = canonical_json_hash(result)
    # CPU reservation is the measured upper bound because exact Modal CPU
    # billing is not exposed by the worker API.
    ledger.complete(config["run_id"], 0.50, receipt_hash)
    data_dir = ROOT / "research/publication/data"
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / "spire_target_train_v1.jsonl").write_text(manifest_text, encoding="utf-8")
    atomic_json(
        data_dir / "spire_target_train_v1.report.json",
        {**result, "plan_sha256": plan_hash, "receipt_sha256": receipt_hash},
    )
    return result


def run_target_teacher(config: dict, contract: StudyContract) -> dict:
    from research.publication.modal_runner import app, label_target_with_teacher

    ensure_live_budget()
    ledger = BudgetLedger(BUDGET_PATH)
    target_report = json.loads(
        (ROOT / "research/publication/data/spire_target_train_v1.report.json").read_text()
    )
    plan = {
        "run_id": config["run_id"], "manifest_sha256": target_report["manifest_sha256"],
        "teacher_model": contract.teacher_model, "teacher_revision": contract.teacher_revision,
        "batch_size": 16,
    }
    plan_hash = canonical_json_hash(plan)
    ledger.reserve(config["run_id"], 1.25, plan_hash)
    with app.run():
        result = label_target_with_teacher.remote(plan)
    pseudo_text = result.pop("pseudo_manifest_text")
    audit_text = result.pop("audit_manifest_text")
    import hashlib

    if hashlib.sha256(pseudo_text.encode()).hexdigest() != result["pseudo_manifest_sha256"]:
        raise RuntimeError("target pseudo-label transfer hash mismatch")
    if hashlib.sha256(audit_text.encode()).hexdigest() != result["audit_manifest_sha256"]:
        raise RuntimeError("target audit transfer hash mismatch")
    receipt_hash = canonical_json_hash(result)
    ledger.complete(config["run_id"], float(result["estimated_resource_cost_usd"]), receipt_hash)
    data = ROOT / "research/publication/data"
    (data / "spire_target_pseudo_large_v3_seed12.jsonl").write_text(pseudo_text)
    (data / "spire_target_audit_120_blind.jsonl").write_text(audit_text)
    atomic_json(
        data / "spire_target_pseudo_large_v3_seed12.receipt.json",
        {**result, "plan_sha256": plan_hash, "receipt_sha256": receipt_hash},
    )
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
    elif config["action"] == "smoke_source_pseudo":
        result = run_source_smoke(config, contract)
    elif config["action"] == "train_source_endpoints":
        result = run_source_endpoints(config, contract)
    elif config["action"] == "prepare_target_manifest":
        result = run_target_preparation(config, contract)
    elif config["action"] == "label_target_teacher":
        result = run_target_teacher(config, contract)
    else:
        raise SystemExit(f"unknown publication action: {config['action']}")
    print("===== RESULT SUMMARY =====", flush=True)
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    print("==========================", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
