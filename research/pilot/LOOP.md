# An auditable research loop

`loop.py` is a local, standard-library-only workflow safeguard. It launches no
compute, performs no model evaluation, and makes no publication judgment. It
freezes what was planned, charges development-set access, checks evidence, and
records a bounded sequence of research decisions.

The first controlled experiment is the reference-supervised selection pilot
described in `docs/RESEARCH_RESET_20260909.md`. Its scientific protocol and
schedule belong in separate JSON files. This tool does not choose effect-size
thresholds, interpret p-values, or turn an infrastructure check into evidence
for a scientific hypothesis.

## Files and integrity

One store contains:

* `contract.json`: the resolved contract, including SHA256 hashes of inputs,
  implementation, evaluator, development manifest, and sealed final-test manifest.
* `events.jsonl`: an append-only, ordered hash chain of registrations, starts,
  evaluation reservations/receipts, terminal outcomes, and promotion decisions.
* `anchor.json`: the current contract hash, ledger length, and head hash.
* `.lock`: an advisory process lock. Supported platforms are macOS and Linux.

Every command verifies the original contract, frozen copy, all registered files,
ledger chain, and previously recorded evidence. Input, code, evaluator, or evidence
mutation fails closed. Evidence must therefore be immutable snapshots, not a log
file that will continue growing. Appends are locked and flushed to disk. A crash
between the event append and anchor update is reported as an incomplete append;
the tool does not silently discard or invent the missing history.

These are audit controls, not a security boundary. A person who rewrites the
contract, entire ledger, and anchor together can forge local history. Preserve
the anchor externally or in version control for independent attestation. The
tool also cannot discover evaluations run outside the guarded dispatcher.

## Contract shape

The following is a mechanical example, **not the scientific contract for this
study**. Paths are resolved relative to the source contract. Replace the artifact
paths with actual existing files. `sha256` may be supplied and verified, or omitted
so initialization computes it. A sealed-test artifact should be its manifest;
the tool does not need test audio.

```json
{
  "schema_version": 1,
  "experiment_id": "selection-pilot-v1",
  "max_jobs": 2,
  "max_running": 1,
  "max_dev_queries": 2,
  "final_test_policy": "sealed",
  "artifacts": {
    "training_manifest": {"path": "train.jsonl", "role": "input"},
    "schedule": {"path": "schedule.json", "role": "input"},
    "training_code": {"path": "training.py", "role": "code"},
    "workflow_code": {"path": "loop.py", "role": "code"},
    "evaluator": {"path": "evaluate_runs.py", "role": "evaluator"},
    "development": {"path": "dev.jsonl", "role": "development"},
    "final_test": {"path": "test.jsonl", "role": "sealed_test"}
  },
  "jobs": {
    "data": {
      "kind": "infrastructure",
      "dependencies": [],
      "parameters": {},
      "required_evidence": ["validation"],
      "success_checks": [
        {"source": "validation", "pointer": "/speaker_content_disjoint", "op": "eq", "value": true}
      ],
      "dev_queries": {}
    },
    "pilot": {
      "kind": "scientific",
      "dependencies": ["data"],
      "parameters": {"num_updates": 600, "batch_size": 8, "seed": 0, "eval_every": 0},
      "required_evidence": ["audit"],
      "success_checks": [
        {"source": "audit", "pointer": "/artifact_integrity_passed", "op": "eq", "value": true}
      ],
      "dev_queries": {
        "baseline": {"dataset": "development", "evaluator": "evaluator", "purpose": "initial weights"},
        "final": {"dataset": "development", "evaluator": "evaluator", "purpose": "fixed final update"}
      }
    }
  },
  "gates": {
    "artifact_ready": {
      "required_jobs": ["data", "pilot"],
      "checks": []
    }
  }
}
```

Each job needs nonempty predeclared `required_evidence` and `success_checks`.
`parameters`, when present, cannot be changed at registration. New jobs or
retries cannot be invented after results: declare separate IDs and dependencies
within `max_jobs`. Dependencies must form an acyclic graph. Only jobs whose
dependencies passed may start, up to `max_running`.

Checks read values from hashed JSON evidence using JSON Pointer syntax. Supported
operators are `eq`, `ne`, `lt`, `le`, `gt`, `ge`, and `in`. Numeric comparisons
require finite numbers and reject booleans. Equality preserves types, so `true`
does not equal `1`. Missing fields fail the check. Free-form `observations` are
retained as notes; they do not override the evidence or checks.

A real scale-up gate should require the independent integrity audit, complete
predeclared arms/seeds, comparable exposures/configurations, and the explicit
scientific decision criteria frozen by the researcher. The example gate above
certifies artifact readiness only. It is deliberately not an effectiveness claim.

## CLI and Python API

Run from the repository root. There are no network calls or paid actions.

```bash
.venv/bin/python -m research.pilot.loop --store research/pilot/ledger init --contract research/pilot/contract.json
.venv/bin/python -m research.pilot.loop --store research/pilot/ledger register --request register.json
.venv/bin/python -m research.pilot.loop --store research/pilot/ledger start --request start.json
.venv/bin/python -m research.pilot.loop --store research/pilot/ledger reserve_query --request reserve-baseline.json
.venv/bin/python -m research.pilot.loop --store research/pilot/ledger query --request baseline-receipt.json
.venv/bin/python -m research.pilot.loop --store research/pilot/ledger record --request result.json
.venv/bin/python -m research.pilot.loop --store research/pilot/ledger status
.venv/bin/python -m research.pilot.loop --store research/pilot/ledger gate artifact_ready
```

Register/start requests are `{"job_id":"pilot"}`. Registration inherits the
contract's fixed parameters when the request omits them. A start request may
include `external_job_id` for a Modal identifier. No command starts that external
job; the guarded dispatcher does so separately.

Python entry points are `initialize(store, contract_path)`,
`apply(store, command, request, evidence_base=".")`, `status(store)`, and
`gate(store, gate_name)`. CLI evidence paths are relative to the request JSON;
Python callers can supply `evidence_base` or absolute paths.

## Charge development access before dispatch

For fixed baseline/final evaluations, reserve both slots **before dispatching the
GPU job**. Reserving a slot consumes one unit even if the worker later fails.
The reservation is not an assertion that evaluation happened and invents no
future checkpoint hash.

```json
{
  "job_id": "pilot",
  "query_id": "baseline",
  "reason": "Precharge fixed baseline and final evaluations before GPU dispatch"
}
```

Send that request to `reserve_query`, then another with `query_id: "final"`.
The tool records a canonical SHA256 of the immutable registered parameters.
Optional `config_sha256` must match that hash. Both reservations must succeed
before the dispatcher launches compute. Reservations cannot be released or reused;
a retry has its own declared job ID and fresh query charges. One query means one
checkpoint evaluated over the registered development set, not each minibatch.

After the runner returns, bind each consumed slot to the actual checkpoint and
an immutable evaluation receipt by calling `query`:

```json
{
  "job_id": "pilot",
  "query_id": "baseline",
  "accounting": "receipt_after_run",
  "checkpoint_sha256": "ACTUAL_64_CHARACTER_LOWERCASE_SHA256",
  "reason": "Completed evaluation returned by the fixed-schedule runner",
  "receipt": {"path": "/absolute/path/to/immutable-baseline-receipt.json"}
}
```

Receipt binding creates a separate event and does not charge a second unit. The
receipt must exist and is hashed. The evaluator/runner is responsible for
verifying what its checkpoint hash represents, such as an initial state hash or
a saved checkpoint artifact hash; record that basis inside the receipt. Never
substitute a config hash for a weight/checkpoint hash.

When a wrapper can call the guard immediately before decoding, `query` also
supports `accounting: "before_evaluation"` with the checkpoint hash. Without a
prior reservation it charges one unit; with a reservation it binds that slot.
Extra or repeated query IDs are rejected. `query` with
`accounting: "reserved_before_dispatch"` remains an alias for `reserve_query`.

## Record evidence and distinguish outcomes

```json
{
  "job_id": "pilot",
  "outcome": "passed",
  "evidence": {"audit": {"path": "/absolute/path/to/immutable-audit.json"}},
  "dev_query_ids": ["baseline", "final"],
  "actual_dev_query_ids": ["baseline", "final"],
  "observations": {"note": "Completed fixed schedule; interpretation deferred to the registered comparison"}
}
```

`dev_query_ids` must list every charged slot exactly once.
`actual_dev_query_ids` separately reports what occurred. Successful or
scientifically refuted outcomes require every declared evaluation, checkpoint
bindings for reservations, all required evidence, and passing predeclared checks.
No status can be overwritten or a terminal job restarted.

The outcomes have different meanings:

* `passed`: this job's declared checks passed. For an infrastructure job that
  means operational validity, not support for the research hypothesis.
* `infrastructure_failed`: the intended experiment failed to execute correctly.
  Hash a failure log/snapshot and state the reason. This does not refute a method.
* `scientifically_refuted`: a scientific job met its separately predeclared
  `refutation_checks`. These use the same evidence-pointer syntax as success
  checks. Without those checks, this status is unavailable.
* `inconclusive`: evidence does not warrant the declared success/refutation
  decision, or the experiment is incomplete. Preserve evidence and explain why.

Failures still require evidence and actual-query accounting. For a worker that
failed before decoding, the two reservations remain in `dev_query_ids`,
`actual_dev_query_ids` is empty, and both units remain charged. A failure after
baseline records that actual query even if final evaluation never happened.

## Existing runs, amendments, and final-test sealing

Work completed before the contract existed must register with
`origin: "retrospective"` and an explicit reason. Import its known development
queries using `accounting: "retrospective_import"`, retaining receipts as outcome
evidence and reporting unknown historical checkpoint hashes honestly. These
queries still consume budget. A retrospective result cannot satisfy a prospective
promotion gate unless that gate explicitly declares `allow_retrospective: true`.
Allowing existing infrastructure evidence is different from claiming retrospective
scientific results were preregistered.

Changing the experiment requires a new contract/store and an explicit amendment.
A new store does not restore an untouched development set. Carry previous
development exposure into the new ledger as retrospective job/query records, and
freeze the previous ledger/anchor as inputs. Budgets are per store; the tool does
not search the filesystem for unrelated previous experiments.

`gate` appends `ready` or `blocked` with the evidence checks and reasons. It neither
launches the next stage nor authorizes a publication claim. The final test stays
sealed after every gate. This pilot has no unseal command, and contracts cannot
declare a query against an artifact with the `sealed_test` role. A later independent
confirmation protocol must account for the complete development history.

Run the offline safeguards with:

```bash
.venv/bin/python -m pytest research/pilot/test_loop.py -q
```
