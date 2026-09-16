# Controlled teacher-selection pilot

This experiment tests one input-selection mechanism on L2-ARCTIC. It does not
claim a new distillation method or a submission-ready paper. The final narrative
is in `output/research_report.md` and `output/pdf/accent_selection_research_report.pdf`.

The three arms use corpus-provided reference targets: the full training pool,
the teacher-filtered pool, and a random pool matched to the filter's speaker,
reference-length, and duration composition. The fixed schedule requires three
training seeds per arm. Test recordings are never decoded or scored.

## Evidence

- `schedule.json`: scientific settings and the prospective pilot signal rule.
- `data/`: corpus identities, frozen speaker/content splits, and audio checks.
- `selections/`: the actual three training pools and their measured balance.
- `teacher_predictions/`: training-only teacher predictions and runtime receipts.
- `runs/<arm>-seed<seed>/`: predictions, loss/exposure history and provenance.
- `results/audit.json`: independently recomputed results, including every seed.
- `results/decision.json`: the fixed pilot decision; no publication guarantee.
- `contract.json` and `ledger/`: immutable inputs, reserved/observed evaluation
  accesses, explicit run states, and the replication gate.
- `protocol_amendments/`: the matching repair made before any proper training.
  The failed earlier selection is retained in `selections_preflight_failed/`.
- `budget.json`: reservations and measured resource estimates, not an invoice.

## Offline verification

From the extracted project root, use Python 3.12 and install the minimal audit
dependency, then recompute the complete result from raw predictions:

```bash
python3 -m venv .venv-audit
.venv-audit/bin/python -m pip install -r research/pilot/requirements-audit.txt
.venv-audit/bin/python -m research.pilot.reproduce
```

The audit itself makes no network requests or inference calls. It requires no
Modal credentials or checkpoint tensors. It verifies original identities,
references, metrics, configurations, exposure sequences and input hashes.
It writes fresh verification files into `tmp/reproduction/` without changing
the delivered results or their ledger.

For the adversarial integrity tests and report renderer, install
`requirements-control.txt`. Run `python -m pytest research/pilot/test_*.py -q`.
The real tiny-architecture forward/backward test needs Torch and explicitly
skips if Torch is absent; it also ran and passed in the GPU runtime.

## Training provenance

`training.py` is the standalone `train(config, manifest_rows, audio_root,
output_dir)` API. `modal_runner.py` fixes the GPU environment and bounds paid
calls. Full checkpoint files remain in the `accent-selection-pilot-results`
Modal volume under `training/<arm>-seed<seed>/checkpoint_last/`; every completed
run records its checkpoint hash. The data and CPU feature caches remain in
`accent-selection-pilot-data`.

The historical launcher and ledger intentionally refuse to overwrite results,
exceed the evaluation/spending budgets, or ignore the delivery deadline. A
future retraining study needs a new output namespace, experiment registration
and resource budget. The reported settings and data identities are fully
preserved for that purpose.

See `LOOP.md` for the controller API and
`docs/AUTORESEARCH_LOOP_REVIEW.md` for the code-grounded upstream assessment.
