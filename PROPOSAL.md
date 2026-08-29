# Proposal: Modal credits for ASR fairness auditing and on-policy distillation research

**TL;DR:** We are an undergraduate research team working on fairness in ASR for Indian-accented
English. We built and ran a full on-policy-vs-off-policy distillation comparison for Whisper —
student `whisper-tiny`, teacher `whisper-large-v3` — entirely on Modal, for **$0.1575** (the full
comparison run) and **~$0.22** across every iteration including two failed debugging runs. It works:
both arms measurably improve the student, and the on-policy arm shows the specific calibration
signature (lower reverse-KL against the teacher on its own rollouts) that the recent LLM on-policy
distillation literature predicts. We're asking for **Modal for Academics** credits to fund a
two-track research program that runs entirely on Modal: an immediate, significance-tested fairness
audit of Indian and frontier ASR systems, and a scaled-up fairness-aware distillation study built on
the pipeline in this repo.

Full technical writeup: [`docs/ONPOLICY_DISTILLATION.md`](docs/ONPOLICY_DISTILLATION.md). Full results: [`results/`](results/). Code:
[`modal_app.py`](modal_app.py) + [`src/onpolicy_distill/`](src/onpolicy_distill/).

## Who we are

Four B.Tech (CSE — AI/ML) students at PES University, Bengaluru, working under faculty guidance
(Dr. Bhaskarjyoti Das) on a capstone research project about fairness and robustness of ASR for
Indian-accented English. Our existing work: an evaluation protocol that pairs per-language-family
WER gaps on the Svarah benchmark ([Javed et al., Interspeech 2023](https://arxiv.org/abs/2305.15760))
with Poisson significance tests — a protocol paper is currently under review at ACL SRW 2026 — plus
GRL and GRPO training studies whose subgroup-level findings motivate the audit below. This repo
(whispr-nano) is our from-scratch validation that on-policy distillation works for the Whisper
family, built and run entirely on Modal.

## What we proved, on Modal, for pocket change

On-policy distillation — sampling a rollout from the *student*, scoring it with the *teacher*, and
minimizing the reverse KL between them — is the mechanism behind recent LLM post-training work
([Agarwal et al., 2024](https://arxiv.org/abs/2306.13649); this exact recipe is described in
[Thinking Machines Lab's 2025 post](https://thinkingmachines.ai/blog/on-policy-distillation/), and
used in production per the [Qwen3 report](https://arxiv.org/abs/2505.09388)). It hasn't been widely
demonstrated as a small, from-scratch, reproducible recipe for the Whisper family, where — unlike the
one prior ASR application we found ([Ark-ASR, May 2026](https://arxiv.org/abs/2605.28139), which
needs cross-tokenizer top-k alignment because its student/teacher don't share a vocabulary) — student
and teacher share (almost) the same tokenizer, so the reverse KL can be computed *exactly*, over the
full vocabulary, with no truncation.

We implemented that recipe end to end — rollout sampling, cross-model teacher-forced scoring, exact
vocabulary alignment (Whisper's large-v3 checkpoint isn't simply a vocab superset of smaller
checkpoints — see `docs/ONPOLICY_DISTILLATION.md` for the specific `<|yue|>`-token subtlety we had to handle correctly),
reverse-KL loss, and a matched off-policy (pseudo-label) baseline — and validated it twice: first on
CPU with a same-model student/teacher stand-in (fast, free, catches logic bugs), then for real on a
Modal T4 with the actual 1.55B teacher.

**Real numbers, one Modal T4, 60 steps, 50 training clips, 20 held-out eval clips:**

| Arm | WER before → after | Reverse KL before → after | Cost | Wall time |
|---|---|---|---|---|
| On-policy | 10.09% → 7.89% | 0.633 → **0.496** | $0.049 | 300s |
| Off-policy (pseudo-label) | 10.09% → 7.57% | 0.633 → 0.524 | $0.105 | 643s |

Both arms improve the student meaningfully in under 5 minutes of combined GPU time. The on-policy arm
reaches the lower final KL against the teacher on its *own* rollouts — exactly the quantity it
directly optimizes and exactly the calibration effect the literature attributes to on-policy training
— while the off-policy arm is marginally ahead on WER at this tiny scale. That split, and whether it
holds up or reverses with more data, is precisely the open question a larger run would answer (see
below).

## Why Modal

- **Per-second billing on real GPUs, no idle cost.** Every number above is billed only for the
  seconds the container actually ran; there's no reserved-instance minimum, which is what makes a
  sub-$1 PoC possible in the first place — and what makes GPU research viable for a student team at
  all.
- **`modal.Volume` for the HF cache.** whisper-large-v3 is a 3GB download; we cache it in a Volume
  once and every subsequent run (smoke, poc, and the scaled-up runs below) reuses it instead of
  re-downloading, which is a direct cost and time saving on Modal specifically.
- **Fast cold starts, `--max-cost-usd` guardrails that actually work in practice.** Our pre-flight
  cost estimator (steps × measured seconds/step × Modal's published per-second rate) is only useful
  because Modal's rates are transparent and per-second; we caught and fixed both an fp16 numerical
  bug and an OOM in eval batching over three short, cheap iterations (~$0.02 total) before spending
  real money on the full run — exactly the fast, low-stakes iteration loop Modal is built for.
- **No infrastructure to manage.** The entire GPU-side implementation is ~250 lines in
  [`modal_app.py`](modal_app.py); there's no cluster, no queue, no idle A100 burning money between
  our debugging sessions.

## What credits would fund: two tracks, one codebase

### Track 1 (now): *"Whose Indian accent?"* — a significance-tested audit of Indian and frontier ASR

India's flagship ASR systems are now validated on Svarah in the aggregate — Sarvam's Saaras V3
reports a single overall number on it — but **no published work tests whether any of these systems
serve all Indian accent groups equally**, with a significance test, under a frozen decode path. Our
Phase 2 evaluation pipeline (per-family WER, ΔDP/ΔEO, per-family noise robustness at 10 dB SNR, and
a mixed-effects Poisson test following [ASR-FAIRBENCH, Interspeech 2025](https://arxiv.org/abs/2505.11572))
does exactly this. We are porting it onto this repo's Modal skeleton and pointing it at the Whisper
family, Distil-Whisper, IndicWhisper, Saaras V3, Qwen3-ASR, and whatever else we can reach.

The economics on Modal, from measured whispr-nano throughput: a large-v3-class system over Svarah's
1,998 eval utterances × 2 conditions is roughly 1.5–3 T4-hours ≈ **$1–2 per system**; small models
are cents. A full audit sweep is **$5–15**, and — this is the part that matters — it's *repeatable
for that price every time a new model ships*, which is what makes a living audit (rather than a
one-off table) feasible on Modal specifically.

### Track 2 (the scale story): fairness-aware on-policy distillation — *"whose accent survives distillation?"*

This is where the two threads merge into a question nobody has claimed.
[Distil-Whisper](https://arxiv.org/abs/2311.00430) distills off-policy at massive scale;
[Ark-ASR](https://arxiv.org/abs/2605.28139) does on-policy distillation for ASR; our audit measures
subgroup fairness with significance tests. **No published work measures whether distillation —
either kind — preserves or amplifies accent bias in the student.** The whispr-nano pipeline plus our
audit machinery is precisely the instrument: distill Whisper students on Indian-accented and
standard corpora, monitor *per-accent-family* reverse KL during training, and report
significance-tested per-family WER for the distilled students alongside their teachers.

Every number below is extrapolated from this PoC's **measured** throughput (3.37s/step on-policy,
9.10s/step off-policy, batch size 4, T4, fp32) — not a guess:

| Scale target | Training exposures | On-policy | Off-policy | Combined GPU-hours | Combined cost (T4) |
|---|---|---|---|---|---|
| This PoC (done) | 240 | $0.05 | $0.11 | 0.26 hr | $0.16 |
| 10x PoC (quick trend check) | 2,400 | $0.33 | $0.89 | 2.1 hr | $1.23 |
| 1 epoch, LibriSpeech `train-clean-100` (~28.5k utterances) | 28,539 | $3.95 | $10.64 | 24.7 hr | $14.58 |
| 1 epoch, full LibriSpeech-960 (~281k utterances) | 281,241 | $38.88 | $104.81 | 243.6 hr | $143.69 |

The program, in priority order:

1. **Full-scale on-policy vs. off-policy, multiple epochs, both arms** (~$150–450 depending on epoch
   count) — tests whether the on-policy calibration advantage seen in the PoC compounds into a
   durable WER gap, the central open question the PoC's scale can't answer.
2. **Multiple seeds and student sizes** (`whisper-tiny` and `whisper-base`, 3 seeds each) for a
   statistically defensible result — roughly 6x the cost of (1). Our team's whole methodological
   identity is significance testing; we won't publish single-seed claims.
3. **The fairness axis:** repeat (1) with Indian-accented training audio and per-family evaluation —
   the *"whose accent survives distillation?"* study, targeting Interspeech 2027. Comparable cost to
   (1)–(2) since the machinery is identical; only the data and the evaluation slices change.
4. **A larger teacher-student gap or a larger student** (e.g., `whisper-small` student), directly
   testing the [Gu et al. (2024)](https://arxiv.org/abs/2306.08543) hypothesis that reverse-KL /
   on-policy training matters most exactly when the student is under-parameterized.

On a faster GPU (e.g. `A100-80GB` at $2.50/hr, ~4x Modal's T4 rate), our conservative estimate is
roughly 3–4x the throughput for the teacher's forward/generate passes, so **wall-clock time would
drop substantially for similar total dollar cost** — useful for iterating faster on (2)–(4) even
though it isn't a $ savings. We'd default to T4/L4 for the bulk of the sweep and reserve A100 for the
largest single runs.

## The ask

We're applying through **[Modal for Academics](https://modal.com/academics)** as university students
with a faculty-guided research project. Itemized:

| Item | Estimate |
|---|---|
| Track 1: audit sweeps (~6–10 systems, clean+noisy), repeated across model releases and re-runs | ~$50 |
| Track 2, items 1–2: full-scale OPD comparison, multi-epoch, multi-seed, two student sizes | ~$900–1,500 |
| Track 2, item 3: the fairness-distillation study (Interspeech 2027 target) | ~$300–500 |
| Headroom for mistakes, ablations, and reviewer-requested re-runs | ~$250 |

**Total: on the order of $1,500–$2,500 in credits.** We'll happily scope down — the guardrail
tooling in this repo exists precisely because we treat compute budgets as hard constraints, and the
entire track record above was produced on $0.22.

## Honest caveats

whispr-nano is a proof of concept, not a paper: 50 training clips, one seed, one language, one
student/teacher pair, fp32-only for numerical-stability reasons documented in `docs/ONPOLICY_DISTILLATION.md`. The
on-policy-vs-off-policy split we observed is exactly the kind of small-scale, noisy result that could
go either way with a different seed — which is the point of the scale-up ask above, not a claim that
the result is already conclusive. Likewise, the audit (Track 1) is in progress, not finished; its
first rows are being produced on Modal now, on the free tier, and we'll gladly share them as they
land.
