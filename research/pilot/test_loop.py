"""Workflow invariants: provenance, evidence, transitions, and evaluation access."""

import copy
import json
from pathlib import Path

import pytest

from research.pilot.loop import apply, gate, initialize, status


def save(path: Path, data):
    path.write_text(json.dumps(data))
    return path


def make_contract(tmp_path):
    artifacts = {}
    for name, role in (("train", "input"), ("trainer", "code"), ("wer", "evaluator"),
                       ("dev", "development"), ("test", "sealed_test")):
        path = save(tmp_path / f"{name}.json", {"name": name, "version": 1})
        artifacts[name] = {"path": str(path), "role": role}
    check = {"source": "metrics", "pointer": "/valid", "op": "eq", "value": True}
    contract = {"schema_version": 1, "experiment_id": "unit-pilot", "max_jobs": 3,
                "max_running": 1, "max_dev_queries": 2, "final_test_policy": "sealed",
                "artifacts": artifacts,
                "jobs": {
                    "data": {"kind": "infrastructure", "dependencies": [], "required_evidence": ["metrics"],
                             "success_checks": [check], "dev_queries": {}},
                    "pilot": {"kind": "scientific", "dependencies": ["data"], "required_evidence": ["metrics"],
                              "success_checks": [check],
                              "refutation_checks": [{"source": "metrics", "pointer": "/refuted", "op": "eq", "value": True}],
                              "dev_queries": {"baseline": {"dataset": "dev", "evaluator": "wer", "purpose": "initial weights"},
                                              "final": {"dataset": "dev", "evaluator": "wer", "purpose": "fixed final update"}}},
                    "other": {"kind": "infrastructure", "dependencies": [], "required_evidence": ["metrics"],
                              "success_checks": [check], "dev_queries": {}}},
                "gates": {"scale_up": {"required_jobs": ["data", "pilot"],
                                       "checks": [{"job_id": "pilot", "source": "metrics", "pointer": "/effect", "op": "ge", "value": 0.1}]}}}
    return contract


@pytest.fixture
def workflow(tmp_path):
    source = save(tmp_path / "contract-source.json", make_contract(tmp_path))
    store = tmp_path / "ledger"
    initialize(store, source)
    return store, source, tmp_path


def begin(store, name, **kwargs):
    apply(store, "register", {"job_id": name, **kwargs})
    apply(store, "start", {"job_id": name})


def finish(store, tmp_path, name, *, outcome="passed", queries=None, metrics=None, reason=""):
    path = save(tmp_path / f"{name}-result.json", metrics or {"valid": True, "effect": 0.2})
    return apply(store, "record", {"job_id": name, "outcome": outcome, "reason": reason,
                                  "dev_query_ids": queries or [], "evidence": {"metrics": {"path": str(path)}}})


def ready_pilot(workflow):
    store, _, tmp_path = workflow
    begin(store, "data")
    finish(store, tmp_path, "data")
    begin(store, "pilot")
    for query in ("baseline", "final"):
        apply(store, "query", {"job_id": "pilot", "query_id": query, "checkpoint_sha256": "a" * 64})


def test_full_stage_readiness_requires_declared_jobs_checks_and_hashes(workflow):
    store, _, tmp_path = workflow
    assert gate(store, "scale_up")["decision"] == "blocked"
    ready_pilot(workflow)
    event = finish(store, tmp_path, "pilot", queries=["baseline", "final"])
    assert len(event["evidence"]["metrics"]["sha256"]) == 64
    assert gate(store, "scale_up")["decision"] == "ready"
    result = status(store)
    assert result["dev_queries_consumed"] == 2
    assert result["final_test"] == "sealed"


def test_failed_effect_gate_cannot_be_promoted_by_success_word(workflow):
    store, _, tmp_path = workflow
    ready_pilot(workflow)
    finish(store, tmp_path, "pilot", queries=["baseline", "final"], metrics={"valid": True, "effect": -0.3})
    assert gate(store, "scale_up")["decision"] == "blocked"


@pytest.mark.parametrize("target", ["source", "frozen", "input", "evaluator"])
def test_contract_or_registered_artifact_tampering_fails_closed(workflow, target):
    store, source, tmp_path = workflow
    paths = {"source": source, "frozen": store / "contract.json", "input": tmp_path / "train.json", "evaluator": tmp_path / "wer.json"}
    changed = json.loads(paths[target].read_text())
    changed["changed"] = True
    save(paths[target], changed)
    with pytest.raises(ValueError, match="(?i)mutation"):
        status(store)


def test_recorded_evidence_mutation_invalidates_future_gate(workflow):
    store, _, tmp_path = workflow
    begin(store, "data")
    finish(store, tmp_path, "data")
    save(tmp_path / "data-result.json", {"valid": False})
    with pytest.raises(ValueError, match="Artifact mutation"):
        gate(store, "scale_up")


@pytest.mark.parametrize("tamper", ["edit", "truncate", "reorder"])
def test_append_only_ledger_detects_edit_truncation_and_reordering(workflow, tamper):
    store, _, _ = workflow
    begin(store, "data")
    path = store / "events.jsonl"
    lines = path.read_text().splitlines()
    if tamper == "edit":
        changed = json.loads(lines[1])
        changed["origin"] = "retrospective"
        lines[1] = json.dumps(changed)
    elif tamper == "truncate":
        lines = lines[:-1]
    else:
        lines[1], lines[2] = lines[2], lines[1]
    path.write_text("\n".join(lines) + "\n")
    with pytest.raises(ValueError, match="(?i)(ledger|chain)"):
        status(store)


def test_missing_evidence_and_unproven_success_are_rejected(workflow):
    store, _, tmp_path = workflow
    begin(store, "data")
    with pytest.raises(ValueError, match="hashed evidence"):
        apply(store, "record", {"job_id": "data", "outcome": "passed", "dev_query_ids": []})
    with pytest.raises(ValueError, match="checks did not pass"):
        finish(store, tmp_path, "data", metrics={"valid": False})
    assert status(store)["jobs"]["data"] == "running"


def test_supplied_bad_evidence_hash_is_rejected(workflow):
    store, _, tmp_path = workflow
    begin(store, "data")
    evidence = save(tmp_path / "evidence.json", {"valid": True})
    with pytest.raises(ValueError, match="hash mismatch"):
        apply(store, "record", {"job_id": "data", "outcome": "passed", "dev_query_ids": [],
                                "evidence": {"metrics": {"path": str(evidence), "sha256": "0" * 64}}})


def test_dependencies_and_max_running_bound_the_work_queue(workflow):
    store, _, _ = workflow
    apply(store, "register", {"job_id": "pilot"})
    with pytest.raises(ValueError, match="Dependencies"):
        apply(store, "start", {"job_id": "pilot"})
    begin(store, "data")
    apply(store, "register", {"job_id": "other"})
    with pytest.raises(ValueError, match="max_running"):
        apply(store, "start", {"job_id": "other"})
    with pytest.raises(ValueError, match="immutable bounded queue"):
        apply(store, "register", {"job_id": "invented_after_results"})


def test_terminal_job_cannot_be_restarted_or_relabelled(workflow):
    store, _, tmp_path = workflow
    begin(store, "data")
    finish(store, tmp_path, "data")
    for command in ("register", "start", "record"):
        with pytest.raises(ValueError):
            apply(store, command, {"job_id": "data", "outcome": "passed"})


def test_infrastructure_failure_is_neither_refutation_nor_ready(workflow):
    store, _, tmp_path = workflow
    begin(store, "data")
    finish(store, tmp_path, "data", outcome="infrastructure_failed", reason="download checksum mismatch")
    assert status(store)["jobs"]["data"] == "infrastructure_failed"
    assert gate(store, "scale_up")["decision"] == "blocked"


def test_scientific_refutation_requires_its_own_predeclared_check(workflow):
    store, _, tmp_path = workflow
    ready_pilot(workflow)
    with pytest.raises(ValueError, match="checks did not pass"):
        finish(store, tmp_path, "pilot", outcome="scientifically_refuted", queries=["baseline", "final"],
               metrics={"valid": True, "refuted": False}, reason="hypothesis did not survive")
    finish(store, tmp_path, "pilot", outcome="scientifically_refuted", queries=["baseline", "final"],
           metrics={"valid": True, "refuted": True}, reason="registered refutation condition met")
    assert status(store)["jobs"]["pilot"] == "scientifically_refuted"


def test_inconclusive_run_keeps_spent_dev_queries_in_ledger(workflow):
    store, _, tmp_path = workflow
    begin(store, "data")
    finish(store, tmp_path, "data")
    begin(store, "pilot")
    apply(store, "query", {"job_id": "pilot", "query_id": "baseline", "checkpoint_sha256": "b" * 64})
    finish(store, tmp_path, "pilot", outcome="inconclusive", queries=["baseline"], reason="training stopped before final comparison")
    assert status(store)["dev_queries_consumed"] == 1
    assert gate(store, "scale_up")["decision"] == "blocked"


def test_repeated_and_unreported_dev_queries_are_rejected(workflow):
    store, _, tmp_path = workflow
    begin(store, "data")
    finish(store, tmp_path, "data")
    begin(store, "pilot")
    query = {"job_id": "pilot", "query_id": "baseline", "checkpoint_sha256": "c" * 64}
    apply(store, "query", query)
    with pytest.raises(ValueError, match="already consumed"):
        apply(store, "query", query)
    with pytest.raises(ValueError, match="account for every"):
        finish(store, tmp_path, "pilot", queries=[])
    with pytest.raises(ValueError, match="missing a predeclared"):
        finish(store, tmp_path, "pilot", queries=["baseline"])


def test_global_dev_query_cap_is_enforced_before_access(tmp_path):
    contract = make_contract(tmp_path)
    contract["max_dev_queries"] = 1
    source = save(tmp_path / "contract.json", contract)
    store = tmp_path / "store"
    initialize(store, source)
    begin(store, "data")
    finish(store, tmp_path, "data")
    begin(store, "pilot")
    apply(store, "query", {"job_id": "pilot", "query_id": "baseline", "checkpoint_sha256": "a" * 64})
    with pytest.raises(ValueError, match="budget exhausted"):
        apply(store, "query", {"job_id": "pilot", "query_id": "final", "checkpoint_sha256": "b" * 64})


def test_final_test_is_sealed_even_after_successful_promotion(workflow):
    store, _, tmp_path = workflow
    ready_pilot(workflow)
    with pytest.raises(ValueError, match="final test is sealed"):
        apply(store, "query", {"job_id": "pilot", "query_id": "test", "checkpoint_sha256": "f" * 64})
    finish(store, tmp_path, "pilot", queries=["baseline", "final"])
    assert gate(store, "scale_up")["final_test"] == "sealed"
    with pytest.raises(ValueError, match="not declared"):
        gate(store, "final_test")


@pytest.mark.parametrize("invalid", ["test_query", "test_gate", "cycle", "no_success_checks", "queue_limit"])
def test_invalid_contract_cannot_be_initialized(tmp_path, invalid):
    contract = make_contract(tmp_path)
    if invalid == "test_query":
        contract["jobs"]["pilot"]["dev_queries"]["final"]["dataset"] = "test"
    elif invalid == "test_gate":
        contract["gates"]["scale_up"]["final_test_access"] = True
    elif invalid == "cycle":
        contract["jobs"]["data"]["dependencies"] = ["pilot"]
    elif invalid == "no_success_checks":
        contract["jobs"]["data"]["success_checks"] = []
    else:
        contract["max_jobs"] = 2
    source = save(tmp_path / "contract.json", contract)
    with pytest.raises(ValueError):
        initialize(tmp_path / "store", source)


def test_retrospective_registration_does_not_pretend_to_be_preregistration(workflow):
    store, _, tmp_path = workflow
    begin(store, "data", origin="retrospective", reason="Existing preparation completed before this safeguard existed")
    finish(store, tmp_path, "data")
    begin(store, "pilot")
    for query in ("baseline", "final"):
        apply(store, "query", {"job_id": "pilot", "query_id": query, "checkpoint_sha256": "a" * 64})
    finish(store, tmp_path, "pilot", queries=["baseline", "final"])
    result = gate(store, "scale_up")
    assert result["decision"] == "blocked"
    assert any("retrospective" in reason for reason in result["reasons"])


def test_numeric_gate_does_not_treat_boolean_as_an_effect_size(workflow):
    store, _, tmp_path = workflow
    ready_pilot(workflow)
    finish(store, tmp_path, "pilot", queries=["baseline", "final"], metrics={"valid": True, "effect": True})
    assert gate(store, "scale_up")["decision"] == "blocked"


def test_precharged_reservations_do_not_fabricate_evaluations_or_refund_failures(workflow):
    store, _, tmp_path = workflow
    begin(store, "data")
    finish(store, tmp_path, "data")
    begin(store, "pilot")
    for query in ("baseline", "final"):
        event = apply(store, "query", {"job_id": "pilot", "query_id": query,
                                        "accounting": "reserved_before_dispatch", "reason": "Precharge fixed baseline/final before GPU dispatch"})
        assert event["checkpoint_sha256"] is None
        assert event["observation_status"] == "reserved_not_observed"
    before = status(store)
    assert before["dev_queries_consumed"] == before["dev_reservations_charged"] == 2
    assert before["dev_queries_actual_reported"] == 0
    evidence = save(tmp_path / "error.json", {"error": "GPU worker failed before evaluation"})
    event = apply(store, "record", {"job_id": "pilot", "outcome": "infrastructure_failed", "reason": "Worker failed",
                                    "dev_query_ids": ["baseline", "final"], "actual_dev_query_ids": [],
                                    "evidence": {"log": {"path": str(evidence)}}})
    assert event["unperformed_reservations_remain_charged"] == ["baseline", "final"]
    assert status(store)["dev_queries_consumed"] == 2


def test_precharged_success_requires_actual_query_accounting(workflow):
    store, _, tmp_path = workflow
    begin(store, "data")
    finish(store, tmp_path, "data")
    begin(store, "pilot")
    for query in ("baseline", "final"):
        apply(store, "query", {"job_id": "pilot", "query_id": query, "accounting": "reserved_before_dispatch", "reason": "Fixed two-query job"})
    evidence = save(tmp_path / "valid.json", {"valid": True, "effect": 0.2})
    request = {"job_id": "pilot", "outcome": "passed", "dev_query_ids": ["baseline", "final"],
               "evidence": {"metrics": {"path": str(evidence)}}}
    with pytest.raises(ValueError, match="actual_dev_query_ids"):
        apply(store, "record", request)
    with pytest.raises(ValueError, match="actually occurred"):
        apply(store, "record", {**request, "actual_dev_query_ids": ["baseline"]})
    with pytest.raises(ValueError, match="Bind reserved queries"):
        apply(store, "record", {**request, "actual_dev_query_ids": ["baseline", "final"]})
    for query in ("baseline", "final"):
        apply(store, "query", {"job_id": "pilot", "query_id": query, "checkpoint_sha256": "d" * 64,
                               "accounting": "receipt_after_run", "reason": "Runner returned completed evaluation",
                               "receipt": {"path": str(evidence)}})
    apply(store, "record", {**request, "actual_dev_query_ids": ["baseline", "final"]})
    assert status(store)["dev_queries_actual_reported"] == 2
    assert status(store)["dev_queries_consumed"] == 2
    assert status(store)["dev_execution_receipts"] == 2


def test_frozen_job_parameters_cannot_be_changed_at_registration(tmp_path):
    contract = make_contract(tmp_path)
    contract["jobs"]["pilot"]["parameters"] = {"num_updates": 600, "seed": 0}
    source = save(tmp_path / "contract.json", contract)
    store = tmp_path / "store"
    initialize(store, source)
    with pytest.raises(ValueError, match="parameters differ"):
        apply(store, "register", {"job_id": "pilot", "parameters": {"num_updates": 1000, "seed": 0}})
    event = apply(store, "register", {"job_id": "pilot"})
    assert event["parameters"] == {"num_updates": 600, "seed": 0}


def test_separate_reservation_command_validates_config_and_receipt(workflow):
    store, _, tmp_path = workflow
    begin(store, "data")
    finish(store, tmp_path, "data")
    begin(store, "pilot")
    with pytest.raises(ValueError, match="config hash differs"):
        apply(store, "reserve_query", {"job_id": "pilot", "query_id": "baseline", "config_sha256": "f" * 64, "reason": "Precharge"})
    event = apply(store, "reserve_query", {"job_id": "pilot", "query_id": "baseline", "reason": "Precharge baseline"})
    assert event["type"] == "query_reserved"
    with pytest.raises(ValueError, match="evidence file"):
        apply(store, "query", {"job_id": "pilot", "query_id": "baseline", "accounting": "receipt_after_run",
                               "checkpoint_sha256": "e" * 64, "reason": "Receipt"})
    receipt = save(tmp_path / "baseline-receipt.json", {"completed": True, "checkpoint_sha256": "e" * 64})
    bound = apply(store, "query", {"job_id": "pilot", "query_id": "baseline", "accounting": "receipt_after_run",
                                   "checkpoint_sha256": "e" * 64, "reason": "Receipt", "receipt": {"path": str(receipt)}})
    assert bound["budget_charge"] == 0
    save(receipt, {"completed": False})
    with pytest.raises(ValueError, match="Artifact mutation"):
        status(store)
