# Cloud ASR study: results and research decision

C. Kaustubh · 13 October 2026 · Prepared for the October 16 review

**Status:** Completed runs below passed independent prediction-level validation. This is an inference study, not a trained-model breakthrough. The Hindi results do not establish Indian-English performance.

## Corrected baseline comparison

Our earlier decoding recipe overstated the room for a custom loop repair. Official Whisper Small scores 60.31% WER on the 405-clip replication set and 62.13% on 396 additional recordings, versus 79.41% and 74.84% with the earlier HF five-beam recipe. The fresh-recording difference is −12.71 points, paired speaker-bootstrap 95% CI [−19.22, −6.72]. These recipes differ in several settings and implementation details; this is not a one-knob attribution.

| Model and decoding recipe | Hindi replication (405) | Hindi fresh audio (396) | Keywords ×1 (100) | Keywords ×5 (100) | Keywords ×8 (100) |
|---|---:|---:|---:|---:|---:|
| Small · HF beam5 | 79.41% | 74.84% | 37.00% | 518.80% | 658.88% |
| Small · official | 60.31% | 62.13% | 26.00% | 24.00% | 36.75% |
| Turbo · HF beam5 | 38.18% | 39.31% | 29.00% | 231.20% | 212.00% |
| Turbo · official | 38.55% | 38.67% | 22.00% | 40.40% | 66.00% |

WER may exceed 100% when insertion errors exceed the reference-word count. All such clips remain in the denominator. Scores use NFC/case/punctuation normalization, without script conversion or numeral equivalence.

## Repairs: aggregate improvement is not preservation

Whisper Turbo HF outputs on the constructed repeated-speech controls:

| Rule | Five-repeat WER | Five-repeat clips harmed | Eight-repeat WER | Eight-repeat clips harmed |
|---|---:|---:|---:|---:|
| Raw output | 231.20% | 0/100 | 212.00% | 0/100 |
| Collapse ≥5 identical words | 117.40% | 77/100 | 92.00% | 74/100 |
| Zero-duration alignment gate | 86.00% | 3/100 | 64.00% | 19/100 |
| Published duration cap | 85.40% | 0/100 | 141.38% | 0/100 |

The text-collapse rule improves pooled five-repeat WER while harming 77/100 clips. A few huge insertion reductions dominate the aggregate. The alignment gate reduces the harm but still damages genuinely supported repetitions. The duration cap is a prior-work baseline and shows no observed clip-level harm on these two constructed sets; this does not prove safety elsewhere.

![Pooled WER versus damage to repeated speech](figures/repetition_tradeoff.svg)

[Figure with caption](figures/repetition_tradeoff.caption.txt) · [PDF figure](figures/repetition_tradeoff.pdf) · [Offline audio demonstration](demo/index.html)

## Why the loop-only branch is closed for the proper Small baseline

The official Small decoder has **zero five-word-loop triggers across all 801 Hindi recordings**. Any repair constrained to leave a hypothesis unchanged when that trigger is absent must therefore return the original hypothesis on every recording here. Formally, if `T(h_i)=0` for every evaluated `i` and `R(x,h)=h` whenever `T(h)=0`, then `R(x_i,h_i)=h_i` for every `i`; all edit counts and pooled WER are exactly unchanged. This is a finite-sample identity, not a claim that repetition never occurs in other data.

A simulated loop-trigger cascade from HF Small to Turbo scores 61.41% on fresh Hindi, but its difference from official Small has a 95% interval crossing zero [−4.11, +2.91]. It adds a larger model without establishing an improvement over the proper Small baseline. The zero-duration gate on Turbo likewise has a fresh-Hindi gain whose interval reaches zero. Neither is accepted as the research contribution.

## Natural-reference repetition check

All 27 non-development-speaker Lahaja recordings with at least three adjacent identical reference words were evaluated (24 speakers; six recording IDs overlap earlier evaluation). Only one reference has five or more repetitions. These are released annotations, not newly adjudicated listening labels.

| Official decoder | Raw WER | Text-collapse WER | Clips helped / harmed |
|---|---:|---:|---:|
| official_default | 72.12% | 71.57% | 1 / 0 |
| official_turbo | 59.62% | 59.62% | 0 / 0 |

The one-clip Small improvement is inconclusive; its interval reaches zero. This selected set cannot estimate a population failure rate. [Natural-control evidence](natural_repetition_validated_20261013.json) · [Selection protocol](NATURAL_REPETITION_PROTOCOL_20261013.md).

## Short-Hindi follow-up

The one-setting ablation compares official Small with its no-speech skip disabled. Both use the same A100 hardware, pinned model and 654-input manifest.

| Condition | Short-word WER / exact accuracy (414) | Empty short-word outputs | Five-repeat Hindi WER (102) | Eight-repeat Hindi WER (102) | Silence inputs emitting words (36) |
|---|---:|---:|---:|---:|---:|
| official_default | 125.36% / 19.81% | 8 | 106.27% | 108.09% | 0/36 |
| official_turbo | 121.01% / 28.74% | 0 | 94.12% | 99.26% | 36/36 |
| official_no_speech_skip | 149.28% / 19.81% | 2 | 106.86% | 108.70% | 34/36 |

Disabling skipping changes short-word WER by +23.91 points, paired speaker-bootstrap 95% CI [+0.74, +59.95]; 0 clips improve and 5 worsen. Silence has no reference-word denominator, so its WER is undefined and is never pooled into speech WER.

[Full short-Hindi evidence](short_hindi_validated_20261013.json) · [Frozen protocol](SHORT_HINDI_PROTOCOL_20261013.md)

### Secondary isolated-word diagnostic

This assumes the task requires exactly one word; it is not a general transcription method. All conditions select the most frequent normalized hypothesis word, resolving ties by first occurrence. No reference word is used for selection.

| Decoder | Original accuracy | Five-copy accuracy | Eight-copy accuracy |
|---|---:|---:|---:|
| official_default | 19.61% | 16.67% | 17.65% |
| official_turbo | 31.37% | 27.45% | 23.53% |
| official_no_speech_skip | 19.61% | 16.67% | 17.65% |

[Complete paired effects and time costs](short_word_replication_20261013.json) · [Diagnostic specification](SHORT_WORD_REPLICATION_DIAGNOSTIC_20261013.md). These 102 examples have one source speaker each; this is exploratory, not proof of novelty or independent confirmation.

## Fixed-vocabulary diagnostic

This tests isolated-word recognition with 92 candidate words from development speakers, plus the original ASR output and an empty candidate. The vocabulary covers 196/414 evaluation words. The remaining 218 clips and all 36 silence controls are retained. No evaluation reference enters the candidate list.

| Candidate objective | All-speech WER | All-speech exact accuracy | In-vocabulary accuracy (196) | Out-of-vocabulary accuracy (218) | Silence emissions (36) |
|---|---:|---:|---:|---:|---:|
| baseline | 125.36% | 19.81% | 28.57% | 11.93% | 0/36 |
| sum_ll | 83.57% | 25.60% | 42.35% | 10.55% | 14/36 |
| mean_ll | 96.86% | 23.67% | 36.73% | 11.93% | 36/36 |
| sum_prior_corrected | 93.96% | 23.91% | 37.76% | 11.47% | 32/36 |
| mean_prior_corrected | 88.41% | 26.81% | 44.39% | 11.01% | 32/36 |

The specified primary contrast, mean prior correction minus mean likelihood, changes exact accuracy by +3.14 points (paired speaker-bootstrap 95% CI [+0.00, +6.56]). It gains 25 correct outputs and loses 12.

The declared improvement gate fails: the primary accuracy interval touches zero and mean prior correction emits words on 32/36 silence controls, compared with 0/36 for the original decoder. Ordinary summed likelihood also achieves lower WER than the prior-corrected choices. The speech improvement is therefore a limited vocabulary-assisted result; it is not evidence that the proposed prior correction is a safe or superior method.

A useful limitation follows directly from the candidate set: if the true word lies outside the fixed vocabulary, the original ASR output is the only candidate that could be correct. This selector can preserve or damage out-of-vocabulary exact accuracy, but cannot increase it. Changes in total WER must therefore be read alongside exact-word accuracy and coverage. This is a known constrained-recognition/prior-correction diagnostic, not a new unrestricted ASR architecture. [Validated candidate evidence](vocabulary_validated_20261013.json) · [Protocol and conditional mathematical identity](VOCABULARY_PROTOCOL_20261013.md).

## Limits and reproducibility

The fresh Hindi recordings share speakers with earlier evaluation. No population fairness claim is justified. Repeated-speech controls concatenate recorded words and are not naturally repeated conversation. Silence controls are artificial/calibration floors, cropped explicitly and deduplicated. One decoded run is used per condition; bootstrap intervals describe sampling uncertainty over speakers, not seed variance, and are not multiplicity-adjusted.

All neural inference ran on cloud GPUs: HF A100 or Modal H100 for decoding, with a native Modal L4 App for the candidate-scoring follow-up. Hardware differs across recipe comparisons, so raw timing ratios are not decoder speedups. Per-recording seeds, model/package revisions, code commits, manifests, audio hashes, hypotheses, complete word edits and prediction hashes are saved. The independent validator refuses incomplete bundles, corrupted hashes or mismatched cohorts. Gated references remain private. Modal Sandboxes are disabled; new Modal runs use bounded Apps with no warm workers.

A repeatability check on the six recordings shared by the natural-repetition extension and earlier runs found minor differences despite matching audio, references, decoder settings, packages, GPU type and per-recording seed policy. Small changed two hypotheses (98 → 99 errors over 153 words); Turbo changed one (105 → 104). The cause is not isolated. These single-run scores must not be interpreted as bitwise deterministic, and the speaker-bootstrap intervals do not include this run-to-run variation. [Repeatability receipt](cloud_repeated_recordings_20261013.json).

[Full aggregate evidence](cloud_decoder_validated_20261013.json) · [Main protocol and prior work](CLOUD_PROTOCOL_20261013.md) · [Validation code](../../scripts/termeval/package_cloud_decoder_study.py) · [Short-study validation](../../scripts/termeval/package_short_hindi_study.py)

## Research decision

The defensible result so far is a decoder-and-evaluation audit: the apparent repair gain depends strongly on the baseline recipe, and pooled WER can conceal deletion of valid repeated speech. This is real, reproducible evidence, but it does not yet establish a novel deployable improvement or guarantee a publication. Keep the properly configured decoder as the baseline, report failed controls, and use any positive short-word result only to justify a separately held-out follow-up. Do not launch training merely to spend the remaining credits.
