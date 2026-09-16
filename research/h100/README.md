# H100 execution of the focused selection experiment

This is a complete-rerun design: three reference-supervised selection arms,
three seeds each, all on H100. The scientific settings and original training
implementation remain fixed. This study reuses development speakers whose
partial T4 outcomes were already inspected; it is exploratory replication of
execution, not an independent confirmation dataset.

The three workers each stage the exact union of their scheduled audio and
feature inputs onto local disk, checking both against recorded SHA-256 values.
They train seeds0,1,2 sequentially from identical initial weights. All models
remain full FP32, with deterministic operations and TF32 disabled. No mixed
precision, loss changes, checkpoint selection, extra training steps, or test
decoding were introduced for speed.

The ledger registers three arm workers with six baseline/final evaluation
slots each, before dispatch. Eighteen new evaluations are disclosed in addition
to the earlier14 conservatively charged slots (including the interrupted T4
run and the two small smoke evaluations). Metadata for test recordings remains
available for checking disjointness; recordings are never decoded or scored.

## Offline verification

From the extracted project root with Python3.12:

```bash
python3 -m venv .venv-audit
.venv-audit/bin/python -m pip install -r research/pilot/requirements-audit.txt
.venv-audit/bin/python -m research.h100.reproduce
```

Dependency installation may use the package index. The verification itself is
offline and needs no model weights, audio, Modal credentials, or GPU. It checks
all nine raw prediction sets, exact reference/utterance coverage, normalization,
initial weights, hardware identity, per-update examples/audio/token exposures,
and the frozen numerical decision. Results go into `tmp/h100-reproduction/`.

`results/evidence.md` includes every arm, seed, normalizer and L1. The narrative
report is `output/h100/research_report.md`; its PDF is
`output/pdf/h100_selection_research_report.pdf`. Checkpoints are a separate
archive under `output/h100/`, with a hash-verified model index. `run.py` and
`contract.json` preserve the actual worker/ledger configuration. The launcher
intentionally refuses to overwrite this experiment; a future study needs a new
namespace and registration.

To regenerate figures/report, install `requirements-render.txt` and run
`python -m research.h100.report`. The figure is embedded as vector graphics in
the PDF and also exported as PDF/SVG. The stopping-reason diagnostic inherited
from the original implementation is invalid and unused in scoring; see
`research/pilot/DIAGNOSTIC_FIELD_NOTE.md`.
