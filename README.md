# PW25_BJD_05 — Whose Indian Accent?

> **Completed H100 experiment:** see [START_HERE.md](START_HERE.md) for all nine
> audited runs, the final research report, evidence archive and offline
> reproduction. The [AutoResearch review](docs/AUTORESEARCH_LOOP_REVIEW.md)
> explains the frozen experiment contract, evaluation-access ledger and
> independent result checks now implemented in [research/pilot](research/pilot/README.md).

> **Research direction, September 9, 2026:** the sprint manuscript is retained as
> a technical report and is **not recommended for submission in its current
> form**. The [reviewer assessment and replacement experiment](docs/RESEARCH_RESET_20260909.md)
> move the project toward a controlled study of teacher filtering, accent
> coverage and recognition improvement. The [literature review](docs/RESEARCH_FRONTIER_20260909.md)
> identifies the existing methods that a new contribution must improve upon.

> **September 2026 reliability study:** the [paper package](paper/README.md)
> supersedes the preliminary interpretation below. The original Saaras and open
> model aggregates used different samples; independent-utterance Poisson tests
> do not establish structural disparities. The revised study uses a matched
> 450-utterance panel, two shared normalizers, native-language breakdowns, and
> explicitly limited dependence diagnostics. Historical predictions are retained
> unchanged for reproduction.

**A subgroup audit of Indian and frontier ASR systems**,
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
benchmark, originally using nominal Poisson tests. Those tests did not account
for speaker or recording dependence and do not establish structural disparities.
In August, Sarvam's flagship Saaras V3 was
evaluated on that exact benchmark by its own team — with only an aggregate
number, no subgroup breakdown, no significance test. Rather than keep
competing on a training result against a funded lab, Phase 3 points our own
protocol at every ASR system we can reach. Full story: [`docs/TEAM_BRIEF.md`](docs/TEAM_BRIEF.md).

## Historical pilot results: original scoring and nominal Poisson tests

| System | Overall WER | Worst family | ΔDP (pp) | Poisson p | n | Cost |
|---|---|---|---|---|---|---|
| whisper-tiny | 20.60% | Sino-Tibetan | 14.19 | 3.3e-24 | 2,500 | $0.034 |
| whisper-small | 10.88% | Sino-Tibetan | 3.18 | 0.0045 | 2,500 | $0.095 |
| whisper-large-v3 | 7.04% | Sino-Tibetan | 2.38 | 0.0152 | 2,500 | $0.411 |
| distil-large-v3 | 10.03% | Sino-Tibetan | 4.02 | 0.00025 | 2,500 | $0.285 |
| **Saaras V3** | **5.81%** (best overall) | Sino-Tibetan | 3.45 | 0.00023 | 450 | $0.226 |
| Qwen3-ASR-0.6B | 15.49% | Sino-Tibetan | 3.57 | 0.0013 | 2,500 | $0.142 |

The original scoring reported Sino-Tibetan as the highest-WER family for all
six systems. The nominal p-values above do not establish population-level
significance, and the unequal samples prevent a matched comparison between
Saaras and the open models. The original recorded resource estimate was
**$1.193**. Historical results, limitations and reproduction steps:
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
