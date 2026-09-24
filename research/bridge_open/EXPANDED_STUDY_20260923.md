# Expanded ASR study — 23 September 2026

Prepared for C. Kaustubh. All five jobs in this expansion completed successfully. The study now contains 7,384 primary predictions over 2,322 distinct FLEURS official-test recordings, plus a separate 44-recording paired Whisper diagnostic and nine-recording Omni implementation check. These subsets overlap the benchmark and are not additional independent data.

| System | Hindi, n=418 | Bengali, n=920 | Assamese, n=984 |
|---|---:|---:|---:|
| Whisper Turbo | 31.35% | 85.86% | 134.71% |
| Whisper Large-v3 | 28.21% | 74.70% | 113.30% |
| Qwen3-ASR-1.7B | 13.09% | Not evaluated | Not evaluated |
| OmniASR CTC-1B | 21.50% | 24.52% | 33.76% |

Entries are micro WER; lower is better. WER can exceed 100% because insertions count as errors. These are comparisons of declared system configurations, not changes to a single trained model. No new weights were trained in this expansion.

Omni is the strongest evaluated Bengali/Assamese system; Qwen remains strongest on Hindi. Relative to Whisper Large-v3, Omni reduces Bengali WER by 50.19 percentage points (sentence-cluster 95% interval 46.20–56.19) and Assamese WER by 79.55 points (74.67–86.42). These intervals resample source sentences within these fixed datasets; they do not establish speaker-population or demographic fairness.

Omni has zero dominant tracked-script mismatches on Bengali and Assamese, but 17 on Hindi. This is a Unicode diagnostic and does not replace human evaluation of transcript language or meaning.

## Controlled language diagnostic

Whisper Large-v3 used the same native checkpoint, float32/eval state, complete waveform, seed and decoding options in both arms; only forced versus automatic language selection changed. Ten short clips per language were chosen by fixed hash order, plus all 14 long clips. The short and long strata are reported separately in the detailed analysis.

| Diagnostic language | n | Forced WER | Automatic WER |
|---|---:|---:|---:|
| hi | 11 | 34.74% | 59.09% |
| bn | 19 | 81.82% | 81.82% |
| as | 14 | 108.06% | 94.33% |

Automatic mode selected Urdu for 4 Hindi clips; Bengali for 13 Assamese clips; and Sinhala for one Assamese clip. The tracked-script heuristic reports that Sinhala output as unknown, not correct. Bengali outputs were unchanged. Automatic selection therefore does not provide a general repair. Four recordings had native timestamp bounds outside the waveform tolerance in both arms; those outputs remain in the analysis. Native public decoding strips EOS, so generation-cap counts remain unknown.

## Validation and reproducibility

The frozen manifest has SHA-256 967d8dca02f9f729c1ed7548dce7283cf9dfc62623815a7c131f71acb13ce2c7. Each source WAV/reference hash, sample rate, channel count, finite sample range and duration was checked. No resampling was needed: the released inputs were already mono 16 kHz. All source archives and model assets were pinned and hashed.

Omni ran the pinned official fairseq2/OmniASR CTC implementation in BF16/eval mode, one complete waveform per recording. All nine acceptance examples matched the official public API exactly, including a long recording per language. Each full shard used identical native inference code. Raw CTC IDs, collapse evidence, source snapshots, configurations, metrics and predictions are retained. Local independent rescoring and eight corruption-rejection checks passed. Empty outputs and runtime failures are never silently removed; all configured IDs were required.

Whisper primary results retain the previously audited HF float32 beam5 short-form route and native full-waveform long-form correction. Qwen uses its native BF16 greedy API, batches up to eight and explicit Hindi. Inference precision, decoding and training differ across systems. The primary scorer preserves Indic marks and applies no numeral conversion.

## Deliverables

- [Transcript/audio viewer](artifacts/benchmark_review_v2/index.html)
- [Blinded listening review](listening_review/index.html): 50 utterances, all unreviewed; autosaved forms and JSON/CSV exports. This frozen packet contains the previous Whisper/Qwen outputs.
- [Omni results and paired intervals](artifacts/fleurs_omni_comparison_v1/RESULTS.md)
- [Language diagnostic](artifacts/language_conditioning_analysis_v1/RESULTS.md)
- [Comparison figure](artifacts/fleurs_omni_comparison_v1/figures/system_comparison.pdf)
- [Prior-art assessment](artifacts/expanded_study_v1/PRIOR_ART.md)

## Jobs and spending

The user removed the total budget cap for this expansion. Five explicit A100 jobs retained 30-minute provider limits; the shared ledger added $7.50 in conservative reservations. Reservations are not measured spending, and provider billing has not been verified. No job in this batch remains running. The full-worker inference guard was 1500 seconds including model loading; an earlier sizing note said 1450 seconds, and the executed configuration is authoritative.

| Run | Source commit | Job |
|---|---|---|
| whisper_language_conditioning_v1 | 58f1cc8 | [Completed](https://huggingface.co/jobs/snigenigmatic/6ab37dae51992417dfcd618c) |
| omni_fleurs_acceptance_v1 | 79bb18b | [Completed](https://huggingface.co/jobs/snigenigmatic/6ab37e7352d0dbd7f1d8320c) |
| omni_fleurs_full_hi_v1 | 3012771 | [Completed](https://huggingface.co/jobs/snigenigmatic/6ab37fa052d0dbd7f1d83281) |
| omni_fleurs_full_bn_v1 | f9f0a02 | [Completed](https://huggingface.co/jobs/snigenigmatic/6ab37fb752d0dbd7f1d83283) |
| omni_fleurs_full_as_v1 | 91aab01 | [Completed](https://huggingface.co/jobs/snigenigmatic/6ab37fcc52d0dbd7f1d8328b) |

## Publication status and next decision

This is a completed benchmark expansion with an auditable negative result for automatic language selection. It is not yet evidence of publishable novelty. Wrong-script ASR output has direct prior art; see [Script collapse in multilingual ASR](https://arxiv.org/abs/2604.08786). A three-language read-speech benchmark does not establish Indian-English accent fairness, conversational robustness, phone performance, or full replication of Humyn BRIDGE.

The next acceptance gate is a fluent human review of the frozen examples, followed by an explicit contribution statement tested against that prior work. Any proposed repair should be frozen and evaluated on independent data. IndicConformer remains unavailable because this HF account does not have gated-file access (403); its preparation is not a completed baseline. No messages, model terms, paper submission or external publication were performed.
