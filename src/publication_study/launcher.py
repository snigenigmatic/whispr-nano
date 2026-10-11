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


def clustered_source_endpoint_plans(config: dict, contract: StudyContract) -> list[dict]:
    """Create the four frozen C=2 real/pseudo source training arms."""
    import hashlib

    assignment_path = ROOT / "research/publication/data/ecapa_c2_assignments.json"
    assignment_receipt = json.loads(
        (ROOT / "research/publication/results/ecapa_c2/receipt.json").read_text(encoding="utf-8")
    )
    if hashlib.sha256(assignment_path.read_bytes()).hexdigest() != assignment_receipt["assignments_sha256"]:
        raise ValueError("local ECAPA assignment hash changed")
    assignments = json.loads(assignment_path.read_text(encoding="utf-8"))["source_utterance_cluster"]
    if len(assignments) != 9495 or set(assignments.values()) != {0, 1}:
        raise ValueError("source ECAPA assignment set is incomplete")
    manifest = _load_jsonl(ROOT / "research/pilot/data/manifest.jsonl")
    real_by_id = {row["utt_id"]: row for row in manifest if row["split"] == "train"}
    dev = [row for row in manifest if row["split"] == "dev"]
    pseudo_by_id = {}
    for row in _load_jsonl(ROOT / "research/pilot/teacher_predictions/all_train.jsonl"):
        transformed = dict(row)
        transformed["reference_text"] = transformed.pop("hypothesis_text")
        transformed["label_source"] = "teacher_pseudo"
        pseudo_by_id[transformed["utt_id"]] = transformed
    if set(real_by_id) != set(assignments) or set(pseudo_by_id) != set(assignments):
        raise ValueError("cluster assignments differ from source endpoint cohorts")
    receipts = _feature_receipts()
    manifest_by_id = {
        row["utt_id"]: row for row in manifest if row["split"] in {"train", "dev"}
    }
    for uid, receipt in receipts.items():
        row = manifest_by_id[uid]
        mono_pcm_bytes = round(receipt["duration_s"] * receipt["original_sample_rate"]) * 2
        wav_overhead_bytes = row["audio_uncompressed_bytes"] - mono_pcm_bytes
        if not 0 <= wav_overhead_bytes <= 4096:
            raise ValueError(f"source WAV is not consistent with mono 16-bit PCM: {uid}")
        receipt["channels"] = 1
    training_hash = hashlib.sha256(
        (ROOT / "research/pilot/training.py").read_bytes()
    ).hexdigest()
    plans = []
    for cluster in (0, 1):
        cluster_ids = sorted(uid for uid, value in assignments.items() if value == cluster)
        if len(cluster_ids) < 1000:
            raise ValueError(f"source cluster {cluster} is too small")
        for endpoint_kind, rows_by_id in (
            ("source_real", real_by_id), ("source_pseudo", pseudo_by_id)
        ):
            arm = f"{endpoint_kind}_c{cluster}"
            train = [rows_by_id[uid] for uid in cluster_ids]
            all_rows = train + dev
            missing = sorted({row["utt_id"] for row in all_rows} - set(receipts))
            if missing:
                raise ValueError(f"missing clustered source feature receipts: {missing[:5]}")
            plans.append({
                "run_id": config["run_id"] + "/" + arm,
                "label_source": arm,
                "endpoint_kind": endpoint_kind,
                "cluster": cluster,
                "cluster_assignment_sha256": assignment_receipt["assignments_sha256"],
                "output_path": f"seed12/sc_c2/{arm}",
                "training_code_sha256": training_hash,
                "rows": all_rows,
                "cache_receipts": {row["utt_id"]: receipts[row["utt_id"]] for row in all_rows},
                "config": {
                    "arm_name": arm, "seed": 12, "num_updates": contract.updates,
                    "batch_size": contract.batch_size, "grad_accum_steps": 1,
                    "learning_rate": contract.learning_rate, "weight_decay": 0.01,
                    "max_grad_norm": 1.0, "eval_every": 0, "eval_batch_size": 8,
                    "max_new_tokens": 224, "model_id": contract.student_model,
                    "model_revision": contract.student_revision,
                },
            })
    return plans


def run_clustered_source_endpoints(config: dict, contract: StudyContract) -> dict:
    from research.publication.modal_runner import app, train_clustered_source_batch

    ensure_live_budget()
    ledger = BudgetLedger(BUDGET_PATH)
    plans = clustered_source_endpoint_plans(config, contract)
    hashes = {plan["label_source"]: canonical_json_hash(plan) for plan in plans}
    batch_plan = {
        "run_id": config["run_id"], "plans": plans,
        "plan_sha256_by_arm": hashes,
    }
    batch_hash = canonical_json_hash(batch_plan)
    ledger.reserve(config["run_id"], 0.50, batch_hash)
    with app.run():
        result = train_clustered_source_batch.remote(batch_plan)
    measured = float(result["estimated_resource_cost_usd"])
    if measured > 0.50:
        raise RuntimeError("clustered endpoint batch exceeded its reservation")
    receipt_hash = canonical_json_hash(result)
    ledger.complete(config["run_id"], measured, receipt_hash)
    for run in result["runs"]:
        arm = run["label_source"]
        atomic_json(
            ROOT / f"research/publication/results/sc_c2/{arm}/receipt.json",
            {
                **run, "batch_run_id": config["run_id"],
                "batch_receipt_sha256": receipt_hash,
                "plan_sha256": hashes[arm],
            },
        )
    atomic_json(
        ROOT / "research/publication/results/sc_c2/batch_receipt.json",
        {**result, "batch_plan_sha256": batch_hash, "receipt_sha256": receipt_hash},
    )
    output = result["runs"]
    expected = {f"source_{kind}_c{cluster}" for kind in ("real", "pseudo") for cluster in (0, 1)}
    if {row["label_source"] for row in output} != expected:
        raise RuntimeError("clustered source endpoint batch incomplete")
    return result


def run_missing_c0_endpoints(config: dict, contract: StudyContract) -> dict:
    """Complete only the two large C=2 endpoints missing after the timed batch."""
    from research.publication.modal_runner import app, train_clustered_source_batch

    ensure_live_budget()
    ledger = BudgetLedger(BUDGET_PATH)
    for arm in ("source_real_c1", "source_pseudo_c1"):
        receipt = json.loads(
            (ROOT / f"research/publication/results/sc_c2/{arm}/receipt.json").read_text()
        )
        if receipt.get("status") != "completed" or receipt.get("completed_updates") != 600:
            raise ValueError(f"accepted {arm} endpoint is missing")
    plans = [
        plan for plan in clustered_source_endpoint_plans(config, contract)
        if plan["cluster"] == 0
    ]
    hashes = {plan["label_source"]: canonical_json_hash(plan) for plan in plans}
    batch_plan = {
        "run_id": config["run_id"], "plans": plans,
        "plan_sha256_by_arm": hashes,
    }
    batch_hash = canonical_json_hash(batch_plan)
    ledger.reserve(config["run_id"], 0.50, batch_hash)
    with app.run():
        result = train_clustered_source_batch.remote(batch_plan)
    measured = float(result["estimated_resource_cost_usd"])
    if measured > 0.50:
        raise RuntimeError("cluster-0 endpoint batch exceeded its reservation")
    receipt_hash = canonical_json_hash(result)
    ledger.complete(config["run_id"], measured, receipt_hash)
    for run in result["runs"]:
        arm = run["label_source"]
        atomic_json(
            ROOT / f"research/publication/results/sc_c2/{arm}/receipt.json",
            {
                **run, "batch_run_id": config["run_id"],
                "batch_receipt_sha256": receipt_hash,
                "plan_sha256": hashes[arm],
            },
        )
    atomic_json(
        ROOT / "research/publication/results/sc_c2/c0_batch_receipt.json",
        {**result, "batch_plan_sha256": batch_hash, "receipt_sha256": receipt_hash},
    )
    return result


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


def ecapa_cluster_plan(config: dict, contract: StudyContract) -> dict:
    """Bind every speaker-embedding input to the frozen manifests and hashes."""
    import hashlib

    manifest = _load_jsonl(ROOT / "research/pilot/data/manifest.jsonl")
    receipts = _feature_receipts()
    source = []
    for row in manifest:
        if row["split"] != "train":
            continue
        receipt = receipts.get(row["utt_id"])
        if receipt is None:
            raise ValueError(f"missing source feature receipt: {row['utt_id']}")
        source.append({
            "utt_id": row["utt_id"],
            "speaker_id": row["speaker_id"],
            "native_language": row["native_language"],
            "audio_path": row["audio_path"],
            "audio_sha256": receipt["audio_sha256"],
            "duration_s": receipt["duration_s"],
        })
    if len(source) != 9495 or len({row["utt_id"] for row in source}) != 9495:
        raise ValueError("pinned source ECAPA cohort changed")
    target_report_path = ROOT / "research/publication/data/spire_target_train_v1.report.json"
    target_report = json.loads(target_report_path.read_text(encoding="utf-8"))
    target_manifest_path = ROOT / "research/publication/data/spire_target_train_v1.jsonl"
    if hashlib.sha256(target_manifest_path.read_bytes()).hexdigest() != target_report["manifest_sha256"]:
        raise ValueError("local target manifest hash changed")
    if contract.seed != 12:
        raise ValueError("ECAPA feasibility batch is frozen to seed 12")
    return {
        "run_id": config["run_id"],
        "model_id": "speechbrain/spkrec-ecapa-voxceleb",
        "model_revision": "0f99f2d0ebe89ac095bcc5903c4dd8f72b367286",
        "seed": contract.seed,
        "clusters": 2,
        "shards": 4,
        "kmeans": {
            "init": "k-means++", "n_init": 10, "max_iter": 300,
            "tol": 0.0001, "algorithm": "lloyd", "random_state": 12,
        },
        "source_train_rows": source,
        "target_manifest_sha256": target_report["manifest_sha256"],
        "target_reference_examples_read": 0,
    }


def run_ecapa_clusters(config: dict, contract: StudyContract) -> dict:
    from research.publication.modal_runner import app, extract_ecapa_shard, finalize_ecapa_clusters
    import hashlib

    ensure_live_budget()
    ledger = BudgetLedger(BUDGET_PATH)
    plan = ecapa_cluster_plan(config, contract)
    plan_hash = canonical_json_hash(plan)
    ledger.reserve(config["run_id"], 1.30, plan_hash)
    with app.run():
        shard_plans = [{**plan, "shard_index": index} for index in range(plan["shards"])]
        shard_results = []
        for shard in extract_ecapa_shard.map(
            shard_plans, order_outputs=False, return_exceptions=True
        ):
            if isinstance(shard, BaseException):
                raise shard
            shard_results.append(shard)
        shard_results.sort(key=lambda row: row["shard_index"])
        if [row["shard_index"] for row in shard_results] != list(range(4)):
            raise RuntimeError("ECAPA shard set is incomplete")
        result = finalize_ecapa_clusters.remote({
            **plan, "shard_receipts": shard_results,
        })
    assignment_text = result.pop("assignments_text")
    assignment_hash = hashlib.sha256(assignment_text.encode()).hexdigest()
    if assignment_hash != result["assignments_sha256"]:
        raise RuntimeError("ECAPA assignment transfer hash mismatch")
    assignments = json.loads(assignment_text)
    if (
        len(assignments["source_utterance_cluster"]) != 9495
        or len(assignments["target_speaker_cluster"]) != 799
        or set(assignments["source_utterance_cluster"].values()) != {0, 1}
        or set(assignments["target_speaker_cluster"].values()) != {0, 1}
    ):
        raise RuntimeError("ECAPA assignment cohort is incomplete")
    weights = assignments["target_speaker_conditioned_weights"]
    if len(weights) != 2 or abs(sum(weights) - 1.0) > 1e-12 or min(weights) <= 0:
        raise RuntimeError("invalid target-conditioned weights")
    measured = float(result["estimated_resource_cost_usd"])
    if measured > 1.30:
        raise RuntimeError("ECAPA extraction exceeded its reservation")
    receipt_hash = canonical_json_hash(result)
    ledger.complete(config["run_id"], measured, receipt_hash)
    data = ROOT / "research/publication/data"
    (data / "ecapa_c2_assignments.json").write_text(assignment_text, encoding="utf-8")
    atomic_json(
        ROOT / "research/publication/results/ecapa_c2/receipt.json",
        {**result, "plan_sha256": plan_hash, "receipt_sha256": receipt_hash},
    )
    return result


def run_recover_ecapa_shard2(config: dict, contract: StudyContract) -> dict:
    """Recover only timed-out shard 2 as two halves, then finalize all evidence."""
    from research.publication.modal_runner import app, extract_ecapa_shard, finalize_ecapa_clusters
    import hashlib

    ensure_live_budget()
    ledger = BudgetLedger(BUDGET_PATH)
    plan = ecapa_cluster_plan(config, contract)
    partial = ROOT / "research/publication/data/ecapa_c2_partial_v7"
    existing = [
        json.loads((partial / f"shard_{index}.receipt.json").read_text())
        for index in (0, 1, 3)
    ]
    if [row["shard_index"] for row in existing] != [0, 1, 3]:
        raise ValueError("persisted ECAPA recovery shard receipts changed")
    existing_cost = sum(float(row["estimated_resource_cost_usd"]) for row in existing)
    plan_hash = canonical_json_hash({**plan, "existing_shard_receipts": existing})
    ledger.reserve(config["run_id"], 0.50, plan_hash)
    with app.run():
        recovery_plans = [
            {
                **plan, "run_id": f"{config['run_id']}/part_{part}",
                "shard_index": 2, "subshards": 2, "subshard_index": part,
            }
            for part in (0, 1)
        ]
        recovered = []
        for shard in extract_ecapa_shard.map(
            recovery_plans, order_outputs=False, return_exceptions=True
        ):
            if isinstance(shard, BaseException):
                raise shard
            recovered.append(shard)
        recovered.sort(key=lambda row: row["subshard_index"])
        if [row["subshard_index"] for row in recovered] != [0, 1]:
            raise RuntimeError("ECAPA shard-2 recovery is incomplete")
        result = finalize_ecapa_clusters.remote({
            **plan, "shard_receipts": existing + recovered,
            "prior_completed_shard_cost_usd": existing_cost,
        })
    assignment_text = result.pop("assignments_text")
    if hashlib.sha256(assignment_text.encode()).hexdigest() != result["assignments_sha256"]:
        raise RuntimeError("recovered ECAPA assignment transfer hash mismatch")
    assignments = json.loads(assignment_text)
    if (
        len(assignments["source_utterance_cluster"]) != 9495
        or len(assignments["target_speaker_cluster"]) != 799
        or set(assignments["source_utterance_cluster"].values()) != {0, 1}
        or set(assignments["target_speaker_cluster"].values()) != {0, 1}
    ):
        raise RuntimeError("recovered ECAPA assignment cohort is incomplete")
    weights = assignments["target_speaker_conditioned_weights"]
    if len(weights) != 2 or abs(sum(weights) - 1.0) > 1e-12 or min(weights) <= 0:
        raise RuntimeError("invalid recovered target-conditioned weights")
    measured = float(result["estimated_resource_cost_usd"])
    if measured > 0.50:
        raise RuntimeError("ECAPA shard recovery exceeded its reservation")
    receipt_hash = canonical_json_hash(result)
    ledger.complete(config["run_id"], measured, receipt_hash)
    data = ROOT / "research/publication/data"
    (data / "ecapa_c2_assignments.json").write_text(assignment_text, encoding="utf-8")
    atomic_json(
        ROOT / "research/publication/results/ecapa_c2/receipt.json",
        {**result, "plan_sha256": plan_hash, "receipt_sha256": receipt_hash},
    )
    return result


def run_target_teacher(config: dict, contract: StudyContract) -> dict:
    from research.publication.modal_runner import (
        app,
        label_target_with_teacher,
        persist_target_teacher_artifacts,
    )
    from publication_study.integrity import validate_target_teacher_artifacts
    import hashlib

    ensure_live_budget()
    ledger = BudgetLedger(BUDGET_PATH)
    target_report = json.loads(
        (ROOT / "research/publication/data/spire_target_train_v1.report.json").read_text()
    )
    plan = {
        "run_id": config["run_id"], "manifest_sha256": target_report["manifest_sha256"],
        "teacher_model": contract.teacher_model, "teacher_revision": contract.teacher_revision,
        "batch_size": 32, "dtype": "float16", "shards": 2,
    }
    plan_hash = canonical_json_hash(plan)
    ledger.reserve(config["run_id"], 1.30, plan_hash)
    with app.run():
        shard_results = []
        shard_plans = [{**plan, "shard_index": index} for index in range(plan["shards"])]
        for shard in label_target_with_teacher.map(
            shard_plans, order_outputs=False, return_exceptions=True
        ):
            if isinstance(shard, BaseException):
                raise shard
            shard_results.append(shard)
        shard_results.sort(key=lambda row: row["shard_index"])
        if [row["shard_index"] for row in shard_results] != [0, 1]:
            raise RuntimeError("target teacher shard set is incomplete")
        pseudo_rows = []
        shard_receipts = []
        for shard in shard_results:
            text = shard.pop("pseudo_manifest_text")
            if hashlib.sha256(text.encode()).hexdigest() != shard["shard_manifest_sha256"]:
                raise RuntimeError("target pseudo-label shard transfer hash mismatch")
            pseudo_rows.extend(json.loads(line) for line in text.splitlines() if line)
            shard_receipts.append(shard)
        pseudo_rows.sort(key=lambda row: row["utt_id"])
        pseudo_text = "".join(
            json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n" for row in pseudo_rows
        )
        audit = []
        for accent in ("Bengali", "Hindi", "Kannada", "Marathi", "Tamil", "Telugu"):
            group = sorted(
                (row for row in pseudo_rows if row["accent"] == accent),
                key=lambda row: (row["teacher_mean_logprob"], row["utt_id"]),
            )
            audit.extend({**row, "confidence_stratum": "low"} for row in group[:10])
            audit.extend({**row, "confidence_stratum": "high"} for row in group[-10:])
        blind_fields = (
            "utt_id", "speaker_id", "native_language", "accent", "gender",
            "audio_path", "audio_sha256", "duration_s", "confidence_stratum",
            "teacher_mean_logprob",
        )
        audit_rows = [{key: row[key] for key in blind_fields} for row in audit]
        audit_rows.sort(key=lambda row: (row["accent"], row["confidence_stratum"], row["utt_id"]))
        audit_text = "".join(json.dumps(row, sort_keys=True) + "\n" for row in audit_rows)

    target_rows = _load_jsonl(ROOT / "research/publication/data/spire_target_train_v1.jsonl")
    integrity = validate_target_teacher_artifacts(target_rows, pseudo_rows, audit_rows)
    measured = sum(float(row["estimated_resource_cost_usd"]) for row in shard_receipts)
    if measured > 1.30:
        raise RuntimeError("sharded target teacher exceeded its conservative reservation")
    result = {
        "run_id": config["run_id"], "status": "completed",
        "model_id": contract.teacher_model, "model_revision": contract.teacher_revision,
        "dtype": "float16", "batch_size_per_shard": 32, "shards": 2,
        "attention_implementation": "sdpa",
        "batch_order": "duration_then_utt_id_strided_across_two_shards",
        "rows": len(pseudo_rows), "audit_rows": len(audit_rows),
        "manifest_sha256": target_report["manifest_sha256"],
        "pseudo_manifest_sha256": hashlib.sha256(pseudo_text.encode()).hexdigest(),
        "audit_manifest_sha256": hashlib.sha256(audit_text.encode()).hexdigest(),
        "integrity": integrity, "shard_receipts": shard_receipts,
        "validation_audio_or_reference_examples_read": 0,
        "wall_seconds_sum": sum(float(row["wall_seconds"]) for row in shard_receipts),
        "estimated_resource_cost_usd": measured,
    }
    receipt_hash = canonical_json_hash(result)
    full_receipt = {**result, "plan_sha256": plan_hash, "receipt_sha256": receipt_hash}
    with app.run():
        persisted = persist_target_teacher_artifacts.remote({
            "pseudo_manifest_text": pseudo_text, "audit_manifest_text": audit_text,
            "pseudo_manifest_sha256": result["pseudo_manifest_sha256"],
            "audit_manifest_sha256": result["audit_manifest_sha256"],
            "receipt": full_receipt,
        })
    if persisted["pseudo_manifest_sha256"] != result["pseudo_manifest_sha256"]:
        raise RuntimeError("persisted target pseudo-label hash mismatch")
    result["persistence"] = persisted
    ledger.complete(config["run_id"], measured, receipt_hash)
    data = ROOT / "research/publication/data"
    (data / "spire_target_pseudo_large_v3_seed12.jsonl").write_text(pseudo_text)
    (data / "spire_target_audit_120_blind.jsonl").write_text(audit_text)
    atomic_json(
        data / "spire_target_pseudo_large_v3_seed12.receipt.json",
        full_receipt,
    )
    return result


def target_pseudo_plan(config: dict, contract: StudyContract, selection: str = "all") -> dict:
    import hashlib

    data = ROOT / "research/publication/data"
    target_report = json.loads((data / "spire_target_train_v1.report.json").read_text())
    pseudo_path = data / "spire_target_pseudo_large_v3_seed12.jsonl"
    pseudo_rows = _load_jsonl(pseudo_path)
    if len(pseudo_rows) != contract.target_examples:
        raise ValueError("target pseudo-label count differs from the frozen contract")
    forbidden = {"reference", "reference_text", "transcript", "sentence", "text"}
    if any(forbidden & set(row) for row in pseudo_rows):
        raise ValueError("target pseudo-label manifest contains a reference-like field")
    if {row.get("utt_id") for row in pseudo_rows} != {
        row["utt_id"] for row in _load_jsonl(data / "spire_target_train_v1.jsonl")
    }:
        raise ValueError("target pseudo-label IDs differ from the frozen target manifest")
    teacher_receipt = json.loads(
        (data / "spire_target_pseudo_large_v3_seed12.receipt.json").read_text()
    )
    pseudo_sha = hashlib.sha256(pseudo_path.read_bytes()).hexdigest()
    if pseudo_sha != teacher_receipt["pseudo_manifest_sha256"]:
        raise ValueError("target pseudo-label receipt hash mismatch")
    manifest = _load_jsonl(ROOT / "research/pilot/data/manifest.jsonl")
    dev = [row for row in manifest if row["split"] == "dev"]
    if len(dev) != 974:
        raise ValueError("pinned source development count changed")
    receipts = _feature_receipts()
    missing = sorted({row["utt_id"] for row in dev} - set(receipts))
    if missing:
        raise ValueError(f"missing source development feature receipts: {missing[:5]}")
    if selection == "all":
        output_path, label_source, expected_train_pool = "seed12/target_pseudo", "target_pseudo", 4800
    elif selection == "top50_per_accent":
        output_path = "seed12/target_pseudo_conf50"
        label_source, expected_train_pool = "target_pseudo_conf50", 2400
    else:
        raise ValueError("unknown target pseudo-label selection")
    return {
        "run_id": config["run_id"], "output_path": output_path,
        "selection": selection, "label_source": label_source,
        "expected_train_pool": expected_train_pool,
        "requires_persisted_feature_cache": True,
        "training_code_sha256": hashlib.sha256(
            (ROOT / "research/pilot/training.py").read_bytes()
        ).hexdigest(),
        "target_manifest_sha256": target_report["manifest_sha256"],
        "pseudo_manifest_sha256": pseudo_sha,
        "source_dev_rows": dev,
        "source_dev_receipts": {row["utt_id"]: receipts[row["utt_id"]] for row in dev},
        "config": {
            "arm_name": label_source, "seed": 12, "num_updates": contract.updates,
            "batch_size": contract.batch_size, "grad_accum_steps": 1,
            "learning_rate": contract.learning_rate, "weight_decay": 0.01,
            "max_grad_norm": 1.0, "eval_every": 0, "eval_batch_size": 8,
            "max_new_tokens": 224, "model_id": contract.student_model,
            "model_revision": contract.student_revision,
        },
    }


def run_target_pseudo(config: dict, contract: StudyContract) -> dict:
    from research.publication.modal_runner import app, train_target_pseudo

    ensure_live_budget()
    ledger = BudgetLedger(BUDGET_PATH)
    plan = target_pseudo_plan(config, contract)
    plan_hash = canonical_json_hash(plan)
    ledger.reserve(config["run_id"], 0.55, plan_hash)
    with app.run():
        result = train_target_pseudo.remote(plan)
    measured = float(result["estimated_resource_cost_usd"])
    if measured > 0.55:
        raise RuntimeError("target endpoint exceeded its conservative reservation")
    receipt_hash = canonical_json_hash(result)
    ledger.complete(config["run_id"], measured, receipt_hash)
    atomic_json(
        ROOT / "research/publication/results/target_pseudo/receipt.json",
        {**result, "plan_sha256": plan_hash, "receipt_sha256": receipt_hash},
    )
    return result


def run_target_confidence(config: dict, contract: StudyContract) -> dict:
    from research.publication.modal_runner import app, train_target_pseudo

    ensure_live_budget()
    ledger = BudgetLedger(BUDGET_PATH)
    plan = target_pseudo_plan(config, contract, selection="top50_per_accent")
    plan_hash = canonical_json_hash(plan)
    ledger.reserve(config["run_id"], 0.65, plan_hash)
    with app.run():
        result = train_target_pseudo.remote(plan)
    measured = float(result["estimated_resource_cost_usd"])
    if measured > 0.65:
        raise RuntimeError("confidence-filter endpoint exceeded its reservation")
    receipt_hash = canonical_json_hash(result)
    ledger.complete(config["run_id"], measured, receipt_hash)
    atomic_json(
        ROOT / "research/publication/results/target_pseudo_conf50/receipt.json",
        {**result, "plan_sha256": plan_hash, "receipt_sha256": receipt_hash},
    )
    return result


def run_recover_target_pseudo(config: dict, contract: StudyContract) -> dict:
    from research.publication.modal_runner import app, recover_target_pseudo_receipt
    import hashlib

    ensure_live_budget()
    ledger = BudgetLedger(BUDGET_PATH)
    # This retry was explicitly stopped after discovering that v1 artifacts
    # had persisted; its 44-second app lifetime is recorded conservatively.
    raw = ledger.load()
    retry = [job for job in raw["jobs"] if job["job_id"] == "train-target-pseudo-seed12-v2"]
    if len(retry) == 1 and retry[0]["status"] == "reserved":
        ledger.cancel(
            "train-target-pseudo-seed12-v2", 0.07,
            "stopped after 44 seconds when the complete v1 artifact set was found on the result volume",
        )
    manifest = _load_jsonl(ROOT / "research/pilot/data/manifest.jsonl")
    dev_ids = sorted(row["utt_id"] for row in manifest if row["split"] == "dev")
    teacher_receipt = json.loads(
        (ROOT / "research/publication/data/spire_target_pseudo_large_v3_seed12.receipt.json").read_text()
    )
    plan = {
        "run_id": config["run_id"],
        "recovered_run_id": "train-target-pseudo-seed12-v1",
        "pseudo_manifest_sha256": teacher_receipt["pseudo_manifest_sha256"],
        "source_dev_ids_sha256": hashlib.sha256(
            json.dumps(dev_ids, separators=(",", ":")).encode()
        ).hexdigest(),
        "original_modal_app_id": "ap-GdSTVOGf4LxtKbVGAQvb6q",
        "original_app_wall_seconds": 340.0,
    }
    plan_hash = canonical_json_hash(plan)
    ledger.reserve(config["run_id"], 0.05, plan_hash)
    with app.run():
        result = recover_target_pseudo_receipt.remote(plan)
    receipt_hash = canonical_json_hash(result)
    ledger.complete(
        plan["recovered_run_id"], float(result["estimated_resource_cost_usd"]), receipt_hash
    )
    ledger.complete(config["run_id"], 0.05, receipt_hash)
    atomic_json(
        ROOT / "research/publication/results/target_pseudo/receipt.json",
        {**result, "recovery_plan_sha256": plan_hash, "receipt_sha256": receipt_hash},
    )
    return result


def global_pseudo2real_plan(config: dict, contract: StudyContract) -> dict:
    import hashlib

    receipts = {}
    for arm in ("source_real", "source_pseudo", "target_pseudo"):
        path = ROOT / f"research/publication/results/{arm}/receipt.json"
        receipt = json.loads(path.read_text())
        if receipt.get("status") != "completed":
            raise ValueError(f"{arm} endpoint is not complete")
        receipts[arm] = receipt
    initial = {receipts[arm].get("initial_state_sha256") for arm in receipts}
    if len(initial) != 1:
        raise ValueError("endpoint initial model states differ")
    manifest = _load_jsonl(ROOT / "research/pilot/data/manifest.jsonl")
    dev = [row for row in manifest if row["split"] == "dev"]
    feature_receipts = _feature_receipts()
    return {
        "run_id": config["run_id"], "output_path": "seed12/global_pseudo2real",
        "scale": contract.correction_scale,
        "checkpoint_sha256": {
            arm: receipts[arm]["checkpoint_sha256"] for arm in receipts
        },
        "training_code_sha256": hashlib.sha256(
            (ROOT / "research/pilot/training.py").read_bytes()
        ).hexdigest(),
        "arithmetic_code_sha256": hashlib.sha256(
            (ROOT / "src/publication_study/arithmetic.py").read_bytes()
        ).hexdigest(),
        "source_dev_rows": dev,
        "source_dev_receipts": {
            row["utt_id"]: feature_receipts[row["utt_id"]] for row in dev
        },
    }


def run_global_pseudo2real(config: dict, contract: StudyContract) -> dict:
    from research.publication.modal_runner import app, build_and_evaluate_global

    ensure_live_budget()
    ledger = BudgetLedger(BUDGET_PATH)
    plan = global_pseudo2real_plan(config, contract)
    plan_hash = canonical_json_hash(plan)
    ledger.reserve(config["run_id"], 0.50, plan_hash)
    with app.run():
        result = build_and_evaluate_global.remote(plan)
    measured = float(result["estimated_resource_cost_usd"])
    if measured > 0.50:
        raise RuntimeError("global correction evaluation exceeded its reservation")
    receipt_hash = canonical_json_hash(result)
    ledger.complete(config["run_id"], measured, receipt_hash)
    atomic_json(
        ROOT / "research/publication/results/global_pseudo2real/receipt.json",
        {**result, "plan_sha256": plan_hash, "receipt_sha256": receipt_hash},
    )
    return result


def run_tune_global_pseudo2real(config: dict, contract: StudyContract) -> dict:
    from research.publication.modal_runner import app, tune_global_pseudo2real

    ensure_live_budget()
    ledger = BudgetLedger(BUDGET_PATH)
    plan = global_pseudo2real_plan(config, contract)
    plan.pop("scale")
    plan["output_path"] = "seed12/global_pseudo2real_tuned_v3"
    plan["scales"] = [round(index / 10, 1) for index in range(1, 11)]
    target_receipt = json.loads(
        (ROOT / "research/publication/results/target_pseudo/receipt.json").read_text()
    )
    plan["uncorrected_lambda_0_source_dev"] = {
        "english_macro_wer": target_receipt["final_dev_metrics"]["english"]["macro_language_wer"],
        "english_micro_wer": target_receipt["final_dev_metrics"]["english"]["micro_wer"],
        "prediction_sha256": target_receipt["artifacts"]["final_dev_predictions.jsonl"],
    }
    plan_hash = canonical_json_hash(plan)
    ledger.reserve(config["run_id"], 0.50, plan_hash)
    with app.run():
        result = tune_global_pseudo2real.remote(plan)
    measured = float(result["estimated_resource_cost_usd"])
    if measured > 0.50:
        raise RuntimeError("global scale grid exceeded its reservation")
    receipt_hash = canonical_json_hash(result)
    ledger.complete(config["run_id"], measured, receipt_hash)
    atomic_json(
        ROOT / "research/publication/results/global_pseudo2real_tuned/receipt.json",
        {**result, "plan_sha256": plan_hash, "receipt_sha256": receipt_hash},
    )
    return result


def sc_correction_plan(config: dict) -> dict:
    import hashlib

    checkpoint_receipts = {}
    for kind in ("real", "pseudo"):
        for cluster in (0, 1):
            arm = f"source_{kind}_c{cluster}"
            receipt = json.loads(
                (ROOT / f"research/publication/results/sc_c2/{arm}/receipt.json").read_text()
            )
            if receipt.get("status") != "completed":
                raise ValueError(f"{arm} endpoint is not complete")
            checkpoint_receipts[arm] = receipt
    target = json.loads(
        (ROOT / "research/publication/results/target_pseudo/receipt.json").read_text()
    )
    initial_states = {
        receipt["initial_state_sha256"] for receipt in [*checkpoint_receipts.values(), target]
    }
    if len(initial_states) != 1:
        raise ValueError("SC endpoint initial model states differ")
    assignment_path = ROOT / "research/publication/data/ecapa_c2_assignments.json"
    assignment_receipt = json.loads(
        (ROOT / "research/publication/results/ecapa_c2/receipt.json").read_text()
    )
    if hashlib.sha256(assignment_path.read_bytes()).hexdigest() != assignment_receipt["assignments_sha256"]:
        raise ValueError("SC assignment hash changed")
    assignments = json.loads(assignment_path.read_text())
    target_weights = assignments["target_speaker_conditioned_weights"]
    if len(target_weights) != 2 or abs(sum(target_weights) - 1.0) > 1e-12:
        raise ValueError("invalid target-conditioned weights")
    manifest = _load_jsonl(ROOT / "research/pilot/data/manifest.jsonl")
    dev = [row for row in manifest if row["split"] == "dev"]
    feature_receipts = _feature_receipts()
    checkpoint_sha256 = {
        arm: receipt["checkpoint_sha256"] for arm, receipt in checkpoint_receipts.items()
    }
    checkpoint_sha256["target_pseudo"] = target["checkpoint_sha256"]
    return {
        "run_id": config["run_id"], "output_path": "seed12/sc_c2_tuned",
        "scales": [round(index / 10, 1) for index in range(1, 11)],
        "method_weights": {
            "published_uniform": [0.5, 0.5],
            "target_conditioned": target_weights,
        },
        "weighting_rules": {
            "published_uniform": "uniform mean of the C source correction vectors",
            "target_conditioned": (
                "fraction of target speakers whose mean ECAPA embedding is nearest to each "
                "source utterance k-means centroid"
            ),
        },
        "cluster_assignment_sha256": assignment_receipt["assignments_sha256"],
        "checkpoint_sha256": checkpoint_sha256,
        "training_code_sha256": hashlib.sha256(
            (ROOT / "research/pilot/training.py").read_bytes()
        ).hexdigest(),
        "arithmetic_code_sha256": hashlib.sha256(
            (ROOT / "src/publication_study/arithmetic.py").read_bytes()
        ).hexdigest(),
        "source_dev_rows": dev,
        "source_dev_receipts": {
            row["utt_id"]: feature_receipts[row["utt_id"]] for row in dev
        },
        "uncorrected_lambda_0_source_dev": {
            "english_macro_wer": target["final_dev_metrics"]["english"]["macro_language_wer"],
            "english_micro_wer": target["final_dev_metrics"]["english"]["micro_wer"],
            "prediction_sha256": target["artifacts"]["final_dev_predictions.jsonl"],
        },
        "target_validation_examples_read": 0,
    }


def run_tune_sc_correction_bank(config: dict) -> dict:
    from research.publication.modal_runner import app, tune_sc_correction_bank

    ensure_live_budget()
    ledger = BudgetLedger(BUDGET_PATH)
    plan = sc_correction_plan(config)
    plan_hash = canonical_json_hash(plan)
    ledger.reserve(config["run_id"], 0.70, plan_hash)
    with app.run():
        result = tune_sc_correction_bank.remote(plan)
    measured = float(result["estimated_resource_cost_usd"])
    if measured > 0.70:
        raise RuntimeError("SC correction grid exceeded its reservation")
    receipt_hash = canonical_json_hash(result)
    ledger.complete(config["run_id"], measured, receipt_hash)
    atomic_json(
        ROOT / "research/publication/results/sc_c2_tuned/receipt.json",
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
    elif config["action"] == "train_clustered_source_endpoints":
        result = run_clustered_source_endpoints(config, contract)
    elif config["action"] == "train_missing_c0_endpoints":
        result = run_missing_c0_endpoints(config, contract)
    elif config["action"] == "prepare_target_manifest":
        result = run_target_preparation(config, contract)
    elif config["action"] == "build_ecapa_clusters":
        result = run_ecapa_clusters(config, contract)
    elif config["action"] == "recover_ecapa_shard2":
        result = run_recover_ecapa_shard2(config, contract)
    elif config["action"] == "label_target_teacher":
        result = run_target_teacher(config, contract)
    elif config["action"] == "train_target_pseudo":
        result = run_target_pseudo(config, contract)
    elif config["action"] == "train_target_confidence":
        result = run_target_confidence(config, contract)
    elif config["action"] == "recover_target_pseudo":
        result = run_recover_target_pseudo(config, contract)
    elif config["action"] == "build_global_pseudo2real":
        result = run_global_pseudo2real(config, contract)
    elif config["action"] == "tune_global_pseudo2real":
        result = run_tune_global_pseudo2real(config, contract)
    elif config["action"] == "tune_sc_correction_bank":
        result = run_tune_sc_correction_bank(config)
    elif config["action"] == "lahaja_baseline":
        from research.lahaja.baseline import main as run_lahaja_baseline
        result = run_lahaja_baseline()
    else:
        raise SystemExit(f"unknown publication action: {config['action']}")
    print("===== RESULT SUMMARY =====", flush=True)
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    print("==========================", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
