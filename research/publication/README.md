# Correction-transfer publication study

The fixed command is:

```bash
/Users/prateekraj/whispr-nano/.venv/bin/python experiments/publication_run.py
```

Every OpenResearch node changes only `run.json`, commits the change, and runs
that same command. `contract.json` freezes the scientific and spending bounds.
`budget.json` is the committed template. Runs share a live ledger at
`~/.local/share/openresearch/whispr-nano-publication-study/budget.json`, which
reserves worst-case cost before dispatch and refuses a repeated run ID or a
total above US$12. This path is outside immutable node snapshots, so a new node
cannot reset prior reservations.

The first node inspects only the gated SPIRE train schema and safe identity
metadata. It does not iterate validation examples. The target manifest builder
will be finalized against that receipt rather than guessing field names.
