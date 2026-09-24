# Complete H100 experiment: start here

**All nine runs completed and passed independent audit.** Three input-selection
arms were trained across three seeds on H100, with the original fixed scientific
settings. The prior partial T4 study is preserved separately as history.

| Input pool | Mean English L1-macro WER |
|---|---:|
| Full reference pool | 25.092% |
| Teacher-filtered pool | 24.911% |
| Matched random pool | 25.029% |

Filtered minus random WER was **-0.118 percentage
points** under English normalization and
**-0.151 points** under minimal normalization.
Positive would favor random selection. The registered harmful-selection
criterion was **not met**. This does not establish equivalence or rule out other
filtering effects. All targets were the supplied corpus references, so this
experiment does not demonstrate a new practical distillation method.

**Recommendation: do not submit this as a completed new-method paper.** It is a
completed, controlled pilot that can support an honest research decision.

## Read the evidence

- `output/pdf/h100_selection_research_report.pdf`: complete report, with a vector figure.
- `output/h100/research_report.md`: editable report.
- `research/h100/results/evidence.md`: every arm, seed, L1 and normalization.
- `research/h100/results/decision.json`: the fixed numerical decision.
- `research/h100/README.md`: runtime, registration and reproduction details.
- `docs/AUTORESEARCH_LOOP_REVIEW.md`: source-level upstream review and adaptations.
- `research/pilot/LOOP.md`: the evidence-controller API and integrity gates.

## Verify without GPU or Modal

From this extracted directory, using Python 3.12:

```bash
python3 -m venv .venv-audit
.venv-audit/bin/python -m pip install -r research/pilot/requirements-audit.txt
.venv-audit/bin/python -m research.h100.reproduce
```

Dependency installation may access the package index. The audit itself is
offline: no inference, audio download, network request, GPU, or Modal credential.
It verifies every published error count, reference and utterance identity,
training exposure sequence, hardware/source identity, and the registered
decision. `HASHES.sha256` covers every other archive entry; run
`shasum -a 256 -c HASHES.sha256` on macOS to check integrity.

## Models and retained history

The separate `trained_checkpoints.zip` contains all nine H100 model weights and
their saved processor/configuration files. `output/h100/model_index.json`
records the exact model hashes checked against the audited training receipts.
The model archive is separate to keep this evidence download compact. No audio
is redistributed here. Optimizer states and audio/features remain in the Modal
volumes described in the report.

`research/pilot/results/` and the older
`output/pdf/accent_selection_research_report.pdf` record the first stop with five
audited T4 runs. They are historical, not part of the nine-run H100 comparison.
The H100 study reused the same development speakers after those partial results
were inspected; all 18 additional accesses are explicitly disclosed. This is
not independent confirmation. Final-test recordings were never decoded or scored.

The execution ledgers preserve original absolute paths. The reproduction
command above resolves the extracted project layout and remains portable.
The known stopping-reason logging defect is disclosed in
`research/pilot/DIAGNOSTIC_FIELD_NOTE.md`; it does not enter scoring or promotion.
