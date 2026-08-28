# `whisper-large-v3` on Svarah: per-family WER audit

**Feasibility check, not a publication-grade result.** See
["How this differs from the source project's protocol"](#how-this-differs-from-the-source-projects-protocol)
below before citing these numbers anywhere.

- Model: `openai/whisper-large-v3` (fp32, zero-shot, greedy decoding, no fine-tuning on Svarah)
- Data: [`ai4bharat/svarah`](https://huggingface.co/datasets/ai4bharat/svarah)
  (Javed et al., "Svarah: Evaluating English ASR Systems on Indian Accents," Interspeech 2023,
  [arXiv:2305.15760](https://arxiv.org/abs/2305.15760)) -- 2500 of 6656
  total utterances (a random subsample; see below)
- Compute: a single Modal `T4`, fp32 -- **actual cost: $0.5833**,
  wall time 3559.1s
  (model load 15.8s + dataset load
  32.0s + transcription
  3511.2s)

## Headline numbers

**Overall WER: 6.59%** (1740 word errors / 26397 reference words)

| Family | n utterances | WER |
|---|---|---|
| indo_aryan | 1673 | 6.67% |
| dravidian | 688 | 5.92% |
| sino_tibetan | 139 | 8.94% |

**ΔDP (max family WER − min family WER): 3.02 percentage points**

**Poisson drop-in-deviance test:** deviance difference = 14.904,
df = 2, **p = 0.0005803**

> **Interpretation: this cross-family WER gap is structural.**
> (Convention: p < 0.05 = "structural" / a real gap; p ≥ 0.05 = not
> distinguishable from sampling noise at this eval size.)

Method: fit a Poisson GLM on per-utterance error counts (substitutions +
deletions + insertions) with `log(reference length in words)` as an offset;
the null model has no family term, the alternative adds a categorical
family fixed effect; the deviance difference between the two nested models
is chi-squared distributed under the null, and the reported p is the
upper-tail probability of that statistic.

## How this differs from the source project's protocol

This audit is a fast, cheap feasibility check run ahead of a review
deadline, reusing this repo's existing Modal/cost-guardrail infrastructure
-- it is **not** a re-run of the (separate, unseen) capstone project's
pipeline, and differs from it in at least three material ways:

1. **Full dataset vs. held-out eval split.** This run evaluates a random
   subsample of Svarah (see below), drawn uniformly from the *entire*
   6,656-utterance `test` split (Svarah's only split) rather than any
   particular held-out slice the source project may have reserved for
   evaluation. Since `whisper-large-v3` was never trained or tuned on
   Svarah, there is no train/eval leakage risk here either way, but the
   *sampling* itself (uniform random vs. a specific curated held-out set)
   is not guaranteed to match the source project's protocol.
2. **Independently-reconstructed family mapping, not the authoritative
   one.** `audit/family_mapping.py`'s Svarah-accent -> family table was
   built from scratch from public linguistic classification and
   cross-checked against the Svarah paper's own language list, landing on
   14 Indo-Aryan / 4 Dravidian / 1 Sino-Tibetan (19 total) -- close to, but
   not exactly, the "15/4/1" figure this task's brief attributed to the
   source project (which does not itself sum to 19, suggesting an
   approximate recollection). **This mapping has not been verified against
   the source project's authoritative file and must be cross-checked before
   any of the per-family numbers above are used in a publication-track
   deliverable.** See the disclaimer and full reasoning at the top of
   `audit/family_mapping.py`.
3. **A single feasibility-check run, not a publication-grade result.** One
   model, one random seed, one Modal T4, greedy decoding only -- no repeated
   seeds/variance estimate, no comparison across ASR systems, and (per
   point 1) `2500` of Svarah's `6656` utterances
   rather than the full corpus, sized to fit comfortably under the
   `--max-cost-usd` budget and the Modal function's hard timeout backstop.

## Reproduction

```bash
uv sync --extra audit                                    # adds statsmodels/pandas for the Poisson fit
uv run modal run audit/modal_svarah_audit.py::smoke      # ~20-40 utterances, validates the pipeline
uv run modal run audit/modal_svarah_audit.py::run        # the real pass reported above
```
