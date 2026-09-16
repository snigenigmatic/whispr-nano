"""Local experiment contracts and an append-only research decision ledger.

This tool schedules no compute and evaluates no speech. It makes declared work,
evidence, development queries, and deterministic promotion rules auditable.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
from typing import Any


ROLES = {"input", "code", "evaluator", "development", "sealed_test"}
OUTCOMES = {"passed", "infrastructure_failed", "scientifically_refuted", "inconclusive"}
OPS = {"eq", "ne", "lt", "le", "gt", "ge", "in"}
ZERO = "0" * 64


def encoded(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                      allow_nan=False).encode("utf-8")


def digest(value: Any) -> str:
    return hashlib.sha256(encoded(value)).hexdigest()


def file_sha(path: str | Path) -> str:
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def read_json(path: str | Path) -> Any:
    def invalid(value):
        raise ValueError(f"Non-finite JSON number: {value}")
    return json.loads(Path(path).read_text(), parse_constant=invalid)


def _atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as stream:
        stream.write(encoded(value) + b"\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def _artifact(spec: dict, base: Path) -> dict:
    path = (base / spec["path"]).resolve()
    if not path.is_file():
        raise ValueError(f"Evidence/input must be an existing regular file: {path}")
    checksum = file_sha(path)
    if spec.get("sha256") and spec["sha256"] != checksum:
        raise ValueError(f"Supplied artifact hash mismatch: {path}")
    return {**spec, "path": str(path), "sha256": checksum, "bytes": path.stat().st_size}


def _verify_artifacts(artifacts: dict) -> None:
    for name, spec in artifacts.items():
        path = Path(spec["path"])
        if not path.is_file() or file_sha(path) != spec["sha256"]:
            raise ValueError(f"Artifact mutation or disappearance: {name} ({path})")


def _validate_check(check: dict) -> None:
    if not check.get("source") or check.get("op") not in OPS or "value" not in check:
        raise ValueError("Each gate check needs source, pointer, op, and value")
    pointer = check.get("pointer")
    if not isinstance(pointer, str) or (pointer and not pointer.startswith("/")):
        raise ValueError("Check pointer must be a JSON pointer (empty string means the whole document)")
    if check["op"] == "in" and not isinstance(check["value"], list):
        raise ValueError("The in operator needs an array of allowed values")


def _validate_contract(contract: dict) -> None:
    if contract.get("schema_version") != 1 or not contract.get("experiment_id"):
        raise ValueError("Contract needs schema_version=1 and experiment_id")
    for name in ("max_jobs", "max_running", "max_dev_queries"):
        value = contract.get(name)
        if type(value) is not int or value < (0 if name == "max_dev_queries" else 1):
            raise ValueError(f"Invalid {name}")
    artifacts, jobs = contract.get("artifacts", {}), contract.get("jobs", {})
    if not artifacts or {s.get("role") for s in artifacts.values()} - ROLES:
        raise ValueError("Artifact roles must be input, code, evaluator, development, or sealed_test")
    if not {"input", "code", "evaluator", "development", "sealed_test"}.issubset({s["role"] for s in artifacts.values()}):
        raise ValueError("Contract must freeze inputs, code, evaluator, development manifest, and sealed test manifest")
    if not jobs or len(jobs) > contract["max_jobs"]:
        raise ValueError("Declared queue exceeds max_jobs or is empty")
    if contract.get("final_test_policy") != "sealed":
        raise ValueError("This pilot requires final_test_policy=sealed; it cannot unseal a test")
    for name, spec in jobs.items():
        if not isinstance(name, str) or not name:
            raise ValueError("Job IDs must be nonempty strings")
        if spec.get("kind") not in {"infrastructure", "scientific"}:
            raise ValueError("Job kind must be infrastructure or scientific")
        if not spec.get("success_checks") or not spec.get("required_evidence"):
            raise ValueError(f"Job {name} needs predeclared success checks and evidence labels")
        for check in spec["success_checks"] + spec.get("refutation_checks", []):
            _validate_check(check)
            if check["source"] not in spec["required_evidence"]:
                raise ValueError(f"Job {name}: gate source must be a required evidence label")
        if set(spec.get("dependencies", [])) - set(jobs) or name in spec.get("dependencies", []):
            raise ValueError(f"Invalid dependencies for {name}")
        for query in spec.get("dev_queries", {}).values():
            if artifacts.get(query.get("dataset"), {}).get("role") != "development":
                raise ValueError("Only the frozen development artifact can be queried; final test is sealed")
            if artifacts.get(query.get("evaluator"), {}).get("role") != "evaluator" or not query.get("purpose"):
                raise ValueError("Development queries need the frozen evaluator and a declared purpose")
    visited, active = set(), set()

    def visit(name):
        if name in active:
            raise ValueError("Cyclic job dependencies")
        if name in visited:
            return
        active.add(name)
        for dep in jobs[name].get("dependencies", []):
            visit(dep)
        active.remove(name)
        visited.add(name)
    for name in jobs:
        visit(name)
    for name, gate in contract.get("gates", {}).items():
        if gate.get("final_test_access") or name == "final_test":
            raise ValueError("No pilot promotion gate may unseal the final test")
        if not gate.get("required_jobs") or set(gate["required_jobs"]) - set(jobs):
            raise ValueError(f"Gate {name} needs declared required_jobs")
        for check in gate.get("checks", []):
            _validate_check(check)
            if check.get("job_id") not in gate["required_jobs"]:
                raise ValueError("Promotion checks must refer to required jobs")


def initialize(store: str | Path, contract_source: str | Path) -> dict:
    store, source = Path(store), Path(contract_source).resolve()
    store.mkdir(parents=True, exist_ok=True)
    with _lock(store):
        if any(path.name != ".lock" for path in store.iterdir()):
            raise ValueError("Contract store must be empty; use a new experiment ID for a revised protocol")
        contract = read_json(source)
        _validate_contract(contract)
        contract["artifacts"] = {name: _artifact(spec, source.parent) for name, spec in contract["artifacts"].items()}
        contract["source_contract_sha256"] = file_sha(source)
        contract["source_contract_path"] = str(source)
        _atomic_json(store / "contract.json", contract)
        anchor = {"contract_sha256": digest(contract), "events": 0, "head": ZERO}
        _atomic_json(store / "anchor.json", anchor)
        (store / "events.jsonl").touch()
        _append(store, anchor, {"type": "initialized", "contract_sha256": anchor["contract_sha256"]})
    return status(store)


@contextmanager
def _lock(store: Path):
    if not store.is_dir():
        raise ValueError("Store does not exist; initialize it first")
    with (store / ".lock").open("a") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def _load(store: Path) -> tuple[dict, dict, list[dict]]:
    contract, anchor = read_json(store / "contract.json"), read_json(store / "anchor.json")
    if digest(contract) != anchor["contract_sha256"]:
        raise ValueError("Frozen contract mutation detected")
    if file_sha(contract["source_contract_path"]) != contract["source_contract_sha256"]:
        raise ValueError("Source contract mutation detected; create a new experiment instead")
    _verify_artifacts(contract["artifacts"])
    events, previous = [], ZERO
    for line in (store / "events.jsonl").read_text().splitlines():
        event = json.loads(line)
        claimed = event.pop("sha256")
        if event.get("sequence") != len(events) + 1 or event.get("previous") != previous or digest(event) != claimed:
            raise ValueError("Ledger event mutation, reorder, or hash-chain break detected")
        event["sha256"] = claimed
        events.append(event)
        previous = claimed
        if event["type"] == "recorded":
            _verify_artifacts(event["evidence"])
        elif event["type"] == "dev_query" and event.get("receipt"):
            _verify_artifacts({"query_receipt": event["receipt"]})
    if len(events) != anchor["events"] or previous != anchor["head"]:
        raise ValueError("Ledger truncation or incomplete append detected; audit before recovery")
    return contract, anchor, events


def _append(store: Path, anchor: dict, body: dict) -> dict:
    event = {**body, "sequence": anchor["events"] + 1, "previous": anchor["head"],
             "recorded_at": datetime.now(timezone.utc).isoformat(), "contract_sha256": anchor["contract_sha256"]}
    event["sha256"] = digest(event)
    with (store / "events.jsonl").open("ab") as stream:
        stream.write(encoded(event) + b"\n")
        stream.flush()
        os.fsync(stream.fileno())
    _atomic_json(store / "anchor.json", {**anchor, "events": event["sequence"], "head": event["sha256"]})
    return event


def _states(contract: dict, events: list[dict]) -> dict:
    jobs = {name: {"status": "unregistered", "queries": {}} for name in contract["jobs"]}
    for event in events:
        if "job_id" not in event:
            continue
        job = jobs[event["job_id"]]
        if event["type"] == "registered":
            job.update(status="queued", registration=event)
        elif event["type"] == "started":
            job["status"] = "running"
        elif event["type"] in {"query_reserved", "dev_query"}:
            old = job["queries"].get(event["query_id"])
            job["queries"][event["query_id"]] = {**event}
            if old and old["type"] == "query_reserved":
                job["queries"][event["query_id"]]["reservation"] = old
        elif event["type"] == "recorded":
            job.update(status=event["outcome"], result=event)
    return jobs


def _pointer(document: Any, pointer: str) -> Any:
    value = document
    if not pointer:
        return value
    for key in pointer[1:].split("/"):
        key = key.replace("~1", "/").replace("~0", "~")
        value = value[int(key)] if isinstance(value, list) else value[key]
    return value


def _checks(checks: list[dict], evidence: dict) -> list[dict]:
    results = []
    for check in checks:
        try:
            actual = _pointer(read_json(evidence[check["source"]]["path"]), check["pointer"])
            expected, op = check["value"], check["op"]
            if op in {"lt", "le", "gt", "ge"}:
                if type(actual) not in (int, float) or type(expected) not in (int, float) or not math.isfinite(actual) or not math.isfinite(expected):
                    raise ValueError("Numeric gates require finite numbers, not booleans")
                passed = {"lt": actual < expected, "le": actual <= expected,
                          "gt": actual > expected, "ge": actual >= expected}[op]
            elif op == "in":
                passed = actual in expected
            else:
                equal = type(actual) is type(expected) and actual == expected
                passed = equal if op == "eq" else not equal
            results.append({"check": check, "actual": actual, "passed": bool(passed)})
        except (KeyError, IndexError, TypeError, ValueError) as error:
            results.append({"check": check, "passed": False, "error": str(error)})
    return results


def apply(store: str | Path, command: str, request: dict, evidence_base: str | Path = ".") -> dict:
    """Apply one structured register/start/query/record command under a file lock."""
    store = Path(store)
    with _lock(store):
        contract, anchor, events = _load(store)
        states = _states(contract, events)
        name = request.get("job_id")
        if name not in states:
            raise ValueError("Job is not in the immutable bounded queue")
        spec, state = contract["jobs"][name], states[name]
        base = {"job_id": name}
        if command == "register":
            if state["status"] != "unregistered":
                raise ValueError("Job already registered; retries need a separately declared job ID")
            origin = request.get("origin", "prospective")
            if origin not in {"prospective", "retrospective"}:
                raise ValueError("origin must be prospective or retrospective")
            if origin == "retrospective" and not request.get("reason"):
                raise ValueError("Retrospective registration requires an explicit reason")
            parameters = request.get("parameters", spec.get("parameters", {}))
            if "parameters" in spec and parameters != spec["parameters"]:
                raise ValueError("Job parameters differ from the frozen contract")
            body = {**base, "type": "registered", "origin": origin,
                    "parameters": parameters, "reason": request.get("reason", "")}
        elif command == "start":
            if state["status"] != "queued":
                raise ValueError("Only queued jobs can start")
            missing = [dep for dep in spec.get("dependencies", []) if states[dep]["status"] != "passed"]
            if missing:
                raise ValueError(f"Dependencies have not passed: {missing}")
            if sum(s["status"] == "running" for s in states.values()) >= contract["max_running"]:
                raise ValueError("Concurrent work exceeds max_running")
            body = {**base, "type": "started", "external_job_id": request.get("external_job_id")}
        elif command in {"query", "reserve_query"}:
            if state["status"] != "running":
                raise ValueError("Development queries require a running job")
            query_id = request.get("query_id")
            query = spec.get("dev_queries", {}).get(query_id)
            if not query:
                raise ValueError("Query not predeclared; final test is sealed and extra dev searches are forbidden")
            accounting = "reserved_before_dispatch" if command == "reserve_query" else request.get("accounting", "before_evaluation")
            old = state["queries"].get(query_id)
            if old and (old["type"] != "query_reserved" or accounting == "reserved_before_dispatch"):
                raise ValueError("Query slot already consumed; repeated dev evaluation cannot be hidden")
            if not old and sum(len(s["queries"]) for s in states.values()) >= contract["max_dev_queries"]:
                raise ValueError("Development query budget exhausted")
            if accounting not in {"before_evaluation", "reserved_before_dispatch", "retrospective_import", "receipt_after_run"}:
                raise ValueError("Invalid development-query accounting mode")
            if accounting == "retrospective_import" and state["registration"]["origin"] != "retrospective":
                raise ValueError("Retrospective query imports require a retrospective job registration")
            checkpoint = request.get("checkpoint_sha256")
            if accounting in {"before_evaluation", "receipt_after_run"} and not re.fullmatch(r"[0-9a-f]{64}", checkpoint or ""):
                raise ValueError("Record the checkpoint SHA256 for each immediate development query")
            if checkpoint is not None and not re.fullmatch(r"[0-9a-f]{64}", checkpoint):
                raise ValueError("Malformed checkpoint SHA256")
            if accounting != "before_evaluation" and not request.get("reason"):
                raise ValueError("Precharged reservations/imports need an explicit accounting reason")
            config_sha = digest(state["registration"]["parameters"])
            if request.get("config_sha256", config_sha) != config_sha:
                raise ValueError("Reservation config hash differs from immutable registered parameters")
            receipt = None
            if accounting == "receipt_after_run":
                if not old or old["type"] != "query_reserved":
                    raise ValueError("Prospective execution receipts require a reservation made before dispatch")
                if not request.get("receipt"):
                    raise ValueError("Execution receipts need a hashed evidence file")
                receipt = _artifact(request["receipt"], Path(evidence_base).resolve())
            body = {**base, "type": "query_reserved" if accounting == "reserved_before_dispatch" else "dev_query", "query_id": query_id, **query,
                    "checkpoint_sha256": checkpoint, "origin": state["registration"]["origin"],
                    "accounting": accounting, "reason": request.get("reason", ""),
                    "config_sha256": config_sha, "receipt": receipt,
                    "budget_charge": 0 if old else 1,
                    "observation_status": "reserved_not_observed" if accounting == "reserved_before_dispatch" else
                                          "execution_receipt_recorded" if accounting == "receipt_after_run" else "declared_query"}
        elif command == "record":
            if state["status"] != "running" or request.get("outcome") not in OUTCOMES:
                raise ValueError("Record requires a running job and an explicit valid terminal outcome")
            evidence = {key: _artifact(value, Path(evidence_base).resolve()) for key, value in request.get("evidence", {}).items()}
            if not evidence:
                raise ValueError("Every outcome requires hashed evidence, including failures")
            claimed_queries = request.get("dev_query_ids")
            if not isinstance(claimed_queries, list) or len(claimed_queries) != len(set(claimed_queries)) or set(claimed_queries) != set(state["queries"]):
                raise ValueError("Outcome must account for every consumed dev query exactly once")
            reservations = any(q["type"] == "query_reserved" or q.get("reservation") for q in state["queries"].values())
            actual_queries = request.get("actual_dev_query_ids", None if reservations else claimed_queries)
            if not isinstance(actual_queries, list) or len(actual_queries) != len(set(actual_queries)) or set(actual_queries) - set(claimed_queries):
                raise ValueError("Record actual_dev_query_ids separately; actual queries must be a subset of charged slots")
            outcome = request["outcome"]
            if outcome != "passed" and not request.get("reason"):
                raise ValueError("Non-passing outcomes need an explicit reason")
            checks = []
            if outcome in {"passed", "scientifically_refuted"}:
                if set(spec["required_evidence"]) - set(evidence):
                    raise ValueError("Required evidence is missing")
                if set(state["queries"]) != set(spec.get("dev_queries", {})):
                    raise ValueError("Scientific/success outcome is missing a predeclared dev evaluation")
                if set(actual_queries) != set(spec.get("dev_queries", {})):
                    raise ValueError("Success/refutation requires evidence that every reserved evaluation actually occurred")
                if any(q["type"] == "query_reserved" for q in state["queries"].values()):
                    raise ValueError("Bind reserved queries to real checkpoint execution receipts before claiming success")
                rules = spec["success_checks"] if outcome == "passed" else spec.get("refutation_checks", [])
                if outcome == "scientifically_refuted" and spec["kind"] != "scientific":
                    raise ValueError("Infrastructure failure is not scientific refutation")
                checks = _checks(rules, evidence)
                if not checks or not all(c["passed"] for c in checks):
                    raise ValueError(f"Predeclared {outcome} checks did not pass: {checks}")
            body = {**base, "type": "recorded", "outcome": outcome, "evidence": evidence,
                    "observations": request.get("observations", {}), "reason": request.get("reason", ""),
                    "dev_query_ids": claimed_queries, "actual_dev_query_ids": actual_queries,
                    "unperformed_reservations_remain_charged": sorted(set(claimed_queries) - set(actual_queries)),
                    "checks": checks}
        else:
            raise ValueError(f"Unknown workflow command: {command}")
        return _append(store, anchor, body)


def status(store: str | Path) -> dict:
    store = Path(store)
    with _lock(store):
        contract, anchor, events = _load(store)
        states = _states(contract, events)
        ready = [name for name, state in states.items() if state["status"] == "queued" and
                 all(states[dep]["status"] == "passed" for dep in contract["jobs"][name].get("dependencies", []))]
        return {"experiment_id": contract["experiment_id"], "contract_sha256": anchor["contract_sha256"],
                "ledger_events": len(events), "ledger_head": anchor["head"],
                "jobs": {name: state["status"] for name, state in states.items()}, "ready_jobs": ready,
                "dev_queries_consumed": sum(len(s["queries"]) for s in states.values()),
                "dev_queries_actual_reported": sum(len(s.get("result", {}).get("actual_dev_query_ids", [])) for s in states.values()),
                "dev_reservations_charged": sum(bool(q["type"] == "query_reserved" or q.get("reservation")) for s in states.values() for q in s["queries"].values()),
                "dev_execution_receipts": sum(q.get("accounting") == "receipt_after_run" for s in states.values() for q in s["queries"].values()),
                "max_dev_queries": contract["max_dev_queries"], "final_test": "sealed"}


def gate(store: str | Path, name: str) -> dict:
    """Record a deterministic readiness decision; never launch or unseal anything."""
    store = Path(store)
    with _lock(store):
        contract, anchor, events = _load(store)
        spec = contract.get("gates", {}).get(name)
        if not spec:
            raise ValueError("Promotion gate was not declared in the frozen contract")
        states = _states(contract, events)
        reasons, checks = [], []
        for job in spec["required_jobs"]:
            if states[job]["status"] != "passed":
                reasons.append(f"{job}: {states[job]['status']}")
            elif states[job]["registration"]["origin"] == "retrospective" and not spec.get("allow_retrospective", False):
                reasons.append(f"{job}: retrospective result cannot satisfy a prospective gate")
        if not reasons:
            for rule in spec.get("checks", []):
                checks.extend(_checks([rule], states[rule["job_id"]]["result"]["evidence"]))
            if not all(c["passed"] for c in checks):
                reasons.append("Predeclared promotion checks did not pass")
        return _append(store, anchor, {"type": "gate_decision", "gate": name,
                                      "decision": "ready" if not reasons else "blocked",
                                      "reasons": reasons, "checks": checks, "final_test": "sealed",
                                      "meaning": "readiness only; no scientific or publication claim"})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store", type=Path, required=True)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("init").add_argument("--contract", type=Path, required=True)
    for name in ("register", "start", "reserve_query", "query", "record"):
        sub.add_parser(name).add_argument("--request", type=Path, required=True)
    sub.add_parser("status")
    sub.add_parser("gate").add_argument("name")
    args = parser.parse_args()
    if args.command == "init":
        result = initialize(args.store, args.contract)
    elif args.command == "status":
        result = status(args.store)
    elif args.command == "gate":
        result = gate(args.store, args.name)
    else:
        result = apply(args.store, args.command, read_json(args.request), args.request.resolve().parent)
    print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
