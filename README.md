# PW25_BJD_05 — Whose Indian Accent?

This repository is the working home of a PES University capstone on
Indian-English automatic speech recognition.

| | |
|---|---|
| **Project ID** | PW25_BJD_05 |
| **Guide** | Dr. Bhaskarjyoti Das |
| **Team** | Aditya Sharma, Adithya V Holla, C Kaustubh, Aditi Mangala Udaya |
| **Active study** | Pseudo-label correction transfer across Indian-English speaker groups |

## Active research question

> When do corrections learned from a teacher's pseudo-label errors transfer to
> Indian-English speakers, and when do they fail?

The publication study uses L2-ARCTIC as the labeled source, untranscribed
SPIRE-SIES training audio as the target adaptation pool, and the speaker-disjoint
SPIRE-SIES validation cohort as a final evaluation set. The closest method is
[Pseudo2Real](https://aclanthology.org/2026.findings-acl.59/). The planned
extension retains its source correction bank and conditions the mixture on
unlabeled target-speaker embeddings.

The final test set stays closed until the method, checkpoints, normalizers, and
claims are frozen. Negative results are retained; seeds are never selected by
test performance.

## Preserved evidence

- [START_HERE.md](START_HERE.md) indexes the completed nine-run H100 pilot.
- [research/pilot](research/pilot/README.md) contains the original experiment
  contract, ledgers, manifests, predictions, and independent checks.
- [research/h100](research/h100/README.md) contains the focused H100 extension.
- [paper](paper/README.md) preserves the matched-panel reliability report.
- [docs/AUTORESEARCH_LOOP_REVIEW.md](docs/AUTORESEARCH_LOOP_REVIEW.md) records
  the evidence and run-control rules adapted from AutoResearch.

The H100 pilot found no supported advantage for teacher-filtered selection over
matched-random selection under reference-label training. It did not test
pseudo-label correction transfer and is not presented as evidence that a new
method improves Indian-English ASR.

## Historical audit limitations

The earlier six-system Svarah audit is exploratory. Saaras and the open models
used unequal cohorts, Svarah does not expose authoritative public speaker IDs,
and Bodo is the audit's only Sino-Tibetan language. Historical Poisson p-values
do not establish population-level disparities. See
[audit/svarah_fairness_audit](audit/svarah_fairness_audit/README.md) and the
[matched-panel report](paper/README.md) for the corrected interpretation.

## Current components

```text
data/research_reset/          frozen audit inputs and contracts
src/research_reset/           sampling, scoring, pairing, and stop rules
listen/                       blinded listening and reconciliation lab
research/pilot/               verified L2-ARCTIC manifests and H100 evidence
research/h100/                completed H100 extension
paper/                        historical matched-panel technical report
experiments/                  fixed-command OpenResearch runner
```

## Existing checks

```bash
PYTHONPATH=src python -m pytest tests/test_research_reset.py tests/test_experiment_runner.py -v
cd listen && npm ci && npm run lint && npm run build
```

Phase 1/2 code remains in
[`snigenigmatic/ast-asr`](https://github.com/snigenigmatic/ast-asr) on the
`fair-cispo-work` branch.
