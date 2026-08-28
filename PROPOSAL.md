# Proposal: Modal credits for on-policy distillation research on ASR

**TL;DR:** We built and ran a full on-policy-vs-off-policy distillation comparison for Whisper —
student `whisper-tiny`, teacher `whisper-large-v3` — entirely on Modal, for **$0.1575** (the full
comparison run) and **~$0.22** across every iteration of this project including two failed debugging
runs. It works: both arms measurably improve the student, and the on-policy arm shows the specific
calibration signature (lower reverse-KL against the teacher on its own rollouts) that the recent LLM
on-policy distillation literature predicts. We're asking for credits to run the same pipeline — code
already written, tested, and working — at a scale that can actually test whether that signature
compounds into the WER advantage the literature reports for LLMs.

Full technical writeup: [`README.md`](README.md). Full results: [`results/`](results/). Code:
[`modal_app.py`](modal_app.py) + [`src/onpolicy_distill/`](src/onpolicy_distill/).

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
checkpoints — see `README.md` for the specific `<|yue|>`-token subtlety we had to handle correctly),
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
  sub-$1 PoC possible in the first place.
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

## What credits would fund

The code is done and tested; what's missing is scale. Every number below is extrapolated from this
PoC's **measured** throughput (3.37s/step on-policy, 9.10s/step off-policy, batch size 4, T4, fp32) —
not a guess:

| Scale target | Training exposures | On-policy | Off-policy | Combined GPU-hours | Combined cost (T4) |
|---|---|---|---|---|---|
| This PoC (done) | 240 | $0.05 | $0.11 | 0.26 hr | $0.16 |
| 10x PoC (quick trend check) | 2,400 | $0.33 | $0.89 | 2.1 hr | $1.23 |
| 1 epoch, LibriSpeech `train-clean-100` (~28.5k utterances) | 28,539 | $3.95 | $10.64 | 24.7 hr | $14.58 |
| 1 epoch, full LibriSpeech-960 (~281k utterances) | 281,241 | $38.88 | $104.81 | 243.6 hr | $143.69 |

A credible research program, roughly in order of priority:

1. **Full LibriSpeech-960, multiple epochs, both arms** (~$150–450 depending on epoch count) — tests
   whether the on-policy calibration advantage seen here compounds into a durable WER gap over off-policy,
   the central open question this PoC's scale can't answer.
2. **Multiple seeds and student sizes** (`whisper-tiny` and `whisper-base`, 3 seeds each) for a
   statistically defensible result rather than a single run — roughly 6x the cost of (1).
3. **Out-of-distribution / noisy audio and multilingual extension** (e.g., Common Voice), since
   [Li et al. (2026)](https://arxiv.org/abs/2604.13016) and the Distil-Whisper robustness claims both
   suggest this is where on-policy vs. off-policy differences should matter most, not on clean
   in-distribution read speech like LibriSpeech.
4. **A larger teacher-student gap or a larger student** (e.g., `whisper-small` student), to test
   whether the on-policy advantage grows or shrinks as the student's capacity approaches the
   teacher's — directly testing the [Gu et al. (2024)](https://arxiv.org/abs/2306.08543) hypothesis
   that reverse-KL / on-policy training matters most exactly when the student is under-parameterized.

On a faster GPU (e.g. `A100-80GB` at $2.50/hr, ~4x Modal's T4 rate), our conservative estimate is
roughly 3–4x the throughput for the teacher's forward/generate passes, so **wall-clock time would
drop substantially for similar total dollar cost** — useful for iterating faster on (2)–(4) even
though it isn't a $ savings. We'd default to T4/L4 for the bulk of the sweep and reserve A100 for the
largest single run.

**Ask:** enough credit to fund items 1–2 above comfortably with room for mistakes and re-runs — on
the order of **$500–$1,500**, matched to whichever Modal program (academic, individual/hobbyist, or
startup) fits the applicant's actual situation; happy to scope up or down based on what's available.

## Honest caveats

This is a proof of concept, not a paper. 50 training clips, one seed, one language, one
student/teacher pair, fp32-only for numerical-stability reasons documented in `README.md`. The
on-policy-vs-off-policy split we observed is exactly the kind of small-scale, noisy result that could
go either way with a different seed — which is the point of the scale-up ask above, not a claim that
the result is already conclusive.
