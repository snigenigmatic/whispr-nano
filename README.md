# PW25_BJD_05 — Whose Indian Accent?

**A significance-tested subgroup audit of Indian and frontier ASR systems**,
plus the on-policy-distillation side-quest that led to it. This repo is the
working home for the Phase 3 pivot of PW25_BJD_05, a PES University
capstone project.

| | |
|---|---|
| **Project ID** | PW25_BJD_05 |
| **Guide** | Dr. Bhaskarjyoti Das |
| **Team** | Aditya Sharma, Adithya V Holla, C Kaustubh, Aditi Mangala Udaya |
| **Course repo** | [`snigenigmatic/ast-asr`](https://github.com/snigenigmatic/ast-asr) (`fair-cispo-work` branch) — the original Phase 1/2 codebase (fairness-training pipeline, LoRA/GRPO experiments). **This repo, not that one, is where the Phase 3 pivot's audit, deck, and results now live and get pushed from** — see [why](#relationship-to-ast-asr) below. |

## The one-paragraph pivot

Phase 2 built an evaluation protocol for Indian-accented ASR: per-family
(Indo-Aryan / Dravidian / Sino-Tibetan) WER on the [Svarah](https://huggingface.co/datasets/ai4bharat/svarah)
benchmark, with a Poisson significance test so a gap is only reported as
real if it clears p < 0.05. In August, Sarvam's flagship Saaras V3 was
evaluated on that exact benchmark by its own team — with only an aggregate
number, no subgroup breakdown, no significance test. Rather than keep
competing on a training result against a funded lab, Phase 3 points our own
protocol at every ASR system we can reach. Full story: [`docs/TEAM_BRIEF.md`](docs/TEAM_BRIEF.md).

## The headline result: six systems, six structural gaps

| System | Overall WER | Worst family | ΔDP (pp) | Poisson p | n | Cost |
|---|---|---|---|---|---|---|
| whisper-tiny | 20.60% | Sino-Tibetan | 14.19 | 3.3e-24 | 2,500 | $0.034 |
| whisper-small | 10.88% | Sino-Tibetan | 3.18 | 0.0045 | 2,500 | $0.095 |
| whisper-large-v3 | 7.04% | Sino-Tibetan | 2.38 | 0.0152 | 2,500 | $0.411 |
| distil-large-v3 | 10.03% | Sino-Tibetan | 4.02 | 0.00025 | 2,500 | $0.285 |
| **Saaras V3** | **5.81%** (best overall) | Sino-Tibetan | 3.45 | 0.00023 | 450 | $0.226 |
| Qwen3-ASR-0.6B | 15.49% | Sino-Tibetan | 3.57 | 0.0013 | 2,500 | $0.142 |

Every system shows a statistically significant per-family gap, and
Sino-Tibetan is the worst-performing family in all six — including Sarvam's
own flagship (best overall WER, but doesn't close the fairness gap) and
Qwen3-ASR (general-ASR SOTA elsewhere, but *worse* on this set than several
Whisper variants). Total cost across all six systems: **$1.193**. Full
writeup, per-family breakdown, and reproduction steps:
[`audit/svarah_fairness_audit/README.md`](audit/svarah_fairness_audit/README.md).

These numbers aren't just claimed — `audit/svarah_fairness_audit/assemble.py`
independently recomputes every one of them straight from the per-utterance
CSVs and cross-checks against each run's summary JSON (9/9 tests green, zero
mismatches):

```bash
cd audit/svarah_fairness_audit
python assemble.py                     # writes master_audit_table.{csv,md}
python -m pytest test_assemble.py -v   # schema hard-fail + math regression
```

## Repository layout

```
docs/
  ONPOLICY_DISTILLATION.md    # the OPD side-quest's full research writeup (was this repo's README)
  TEAM_BRIEF.md               # the pivot story, for teammates, start here for context
  WEEKEND_PLAN.md             # the execution plan with per-person agent-task briefs
  MONDAY_REVIEW.md            # the scoped-down solo plan that produced the audit below
  RESULTS_BRIEF.md            # what the audit found, where each number came from
  RECEIPTS_PACK.md / .pdf     # verified quotes backing the deck's literature claims
  PW25_BJD_05_Phase3_Review1.pptx   # the review deck, built on the official template
  deck_assets/                # the deck's python-pptx generator + figures (reproducible)
audit/
  svarah_fairness_audit/      # the six-system audit: scripts, results, assembler, tests
  opd_bias_extension/         # earlier, single-model Svarah bias check from the OPD side-quest
src/onpolicy_distill/         # the on-policy distillation implementation (see docs/ONPOLICY_DISTILLATION.md)
tests/                        # OPD unit + CPU integration tests
modal_app.py                  # the OPD PoC's Modal App (image, GPU function, CLI entrypoints)
results/                      # OPD PoC metrics/plots from the reported Modal runs
PROPOSAL.md                   # the Modal-credits pitch built on the OPD PoC's results
references.bib                # citations for the OPD writeup
```

## The on-policy distillation side-quest

While researching for the pivot, this repo's original project took shape:
a proof of concept that **on-policy distillation** — the technique behind
recent LLM post-training recipes — transfers to autoregressive ASR.
`whisper-tiny` learns from `whisper-large-v3` by having the teacher score
the *student's own sampled transcripts*, on a single Modal T4, for **$0.16**.
Parked as a future direction (two 2026 papers already did OPD for ASR at
scale) rather than this cycle's contribution, but built and validated with
real numbers. Full writeup: [`docs/ONPOLICY_DISTILLATION.md`](docs/ONPOLICY_DISTILLATION.md).
Modal-credits pitch built on it: [`PROPOSAL.md`](PROPOSAL.md).

## Relationship to `ast-asr`

The capstone's Phase 1/2 code — the fairness-training pipeline, the LoRA and
GRPO experiments, `ast_asr.taxonomy` / `ast_asr.metrics` (the scoring
convention this audit reuses) — lives in
[`snigenigmatic/ast-asr`](https://github.com/snigenigmatic/ast-asr) on the
`fair-cispo-work` branch. The audit, the report-bug fixes, and the receipts
pack were also committed there, locally. But that checkout has had
persistent GitHub push-credential problems in this working environment, so
those commits haven't reliably landed on GitHub.

Rather than keep the Phase 3 deliverables blocked on that, this repo is now
the one that actually gets pushed and presented from — it has push access
that works, and it already had the Modal skeleton (image, volumes, cost
guardrails) this audit's Whisper-family systems reuse directly. If/when the
`ast-asr` push access is fixed, the identical audit files can be copied
back over (exact file map + the two report-bug patches that only exist in
`ast-asr`: [`docs/AST_ASR_PUSH_HANDOFF.md`](docs/AST_ASR_PUSH_HANDOFF.md));
until then, treat **this repo as authoritative** for anything Phase 3.
