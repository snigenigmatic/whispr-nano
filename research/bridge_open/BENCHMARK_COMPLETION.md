# Completed ASR benchmark and evidence package
Latest expansion: [23 September study](EXPANDED_STUDY_20260923.md) adds complete OmniASR results and the controlled language diagnostic. The report below preserves the earlier completed benchmark.


Prepared for **C. Kaustubh**. This closes the FLEURS pipeline correction and the focused Qwen Hindi extension. It contains **5,062 primary predictions over 2,322 distinct recordings**: two Whisper systems on all three languages, plus Qwen3-ASR-1.7B on all 418 Hindi recordings.

| Declared system | Hindi (418) | Bengali (920) | Assamese (984) |
|---|---:|---:|---:|
| Whisper Turbo | 31.35% | 85.86% | 134.71% |
| Whisper Large-v3 | 28.21% | 74.70% | 113.30% |
| Qwen3-ASR-1.7B | 13.09% | Not evaluated | Not evaluated |

Values are micro WER, including observed model-quality failures. The same pinned source files, references and primary scorer are used for shared cohorts. WER can exceed 100% through insertions.

## What was corrected and checked

- Verified all 2,322 WAVs and reference joins against the pinned FLEURS source. Files are already mono 16 kHz; independently decoded waveforms agree. No sample-rate relabeling or extra resampling was needed.
- Matched native/HF Whisper weights, evaluation state and frontend features on the diagnostic cohorts. Short-form batch, single-example and pipeline results reproduced the saved outputs. Native checks reproduced the severe Assamese failures.
- Replaced independent fixed 30-second splits for **every long recording**: 14 per model, not only favorable examples. Full-waveform native transcription now handles them. All 2,308 short recordings per Whisper model retain the audited HF recipe.
- Qwen uses its documented native transcribe API with explicit Hindi, complete waveforms, actual bfloat16/evaluation state, frozen greedy generation and batch size up to eight. Serial and batched diagnostics matched exactly on all three recordings before full dispatch.
- Recovered exact source snapshots, transport hashes, configurations, model/data hashes, predictions, raw Qwen generation traces and receipts. Independently rescored every accepted Qwen output and all corrected Whisper outputs.
- Passed 54 local benchmark checks and seven Qwen runner/trace checks. The old long-audio launcher requires an explicit historical-diagnostic override; ordinary dispatch is blocked.

## Read and inspect

- [Offline transcript and audio viewer](artifacts/benchmark_review/index.html): all references and predictions, per-clip scores, decoding routes and script diagnostics. Audio playback uses the local verified FLEURS data.
- [Full Hindi comparison and paired intervals](artifacts/fleurs_qwen_comparison_v1/RESULTS.md).
- [Whisper comparison, including separate short/long strata](artifacts/fleurs_whisper_corrected_v3/RESULTS.md).
- [Input/source audit](artifacts/fleurs_input_audit/CONCLUSIONS.md), [native Whisper audit](FLEURS_DILIGENCE_REPORT.md), and [scorer sensitivity](artifacts/fleurs_scoring_audit/REPORT.md).
- [Qwen batch acceptance](artifacts/qwen_native_batch_canary_v1/BATCH_ACCEPTANCE.json), [full-run validation](artifacts/qwen_native_full_hi_v1/independent_validation.json), and [validation receipt](artifacts/benchmark_review/validation_receipt.json).
- [Unannotated human review queue](artifacts/benchmark_review/human_review_queue.json): low, middle and high observed WER examples for each evaluated model/language cell. This is a diagnostic selection, not a representative quality sample.

## Completed GPU jobs

- [Whisper correction + Turbo parity](https://huggingface.co/jobs/snigenigmatic/6ab2c32252d0dbd7f1d7fc83) — source commit `4eb3cdafc6eb`, preserved in `fleurs_native_long_correction/source_snapshot.tar.gz`.
- [Qwen serial acceptance](https://huggingface.co/jobs/snigenigmatic/6ab2c4f652d0dbd7f1d7fd0f) — source commit `a15755acaac3`, preserved in `qwen_native_canary_v1/source_snapshot.tar.gz`.
- [Qwen batch acceptance](https://huggingface.co/jobs/snigenigmatic/6ab2c71951992417dfcd3ec0) — source commit `60fb572f7dd3`, preserved in `qwen_native_batch_canary_v1/source_snapshot.tar.gz`.
- [Qwen full Hindi](https://huggingface.co/jobs/snigenigmatic/6ab2c7ac52d0dbd7f1d7fd9a) — source commit `6df7cc65424f`, preserved in `qwen_native_full_hi_v1/source_snapshot.tar.gz`.

**Budget:** $40.50/$50.00 conservatively reserved in the shared HF ledger; $9.50 unreserved. Actual provider billing has not been verified. No new Modal spend was made for this closeout. [Budget receipt](artifacts/benchmark_review/budget_receipt.json).

## Interpretation and remaining research work

This is a completed, reproducible descriptive benchmark of these configurations. It provides evidence of output failures and decoding sensitivity. It does not yet establish a new ASR method or the novelty required for a publication.

The Whisper pooled scores use a duration-based routing rule; the report preserves separate strata and the historical fixed-split sensitivity results. This correction was made after inspecting test results and is not a held-out method gain. Qwen and Whisper differ in precision, search and API postprocessing, so a ranking here compares declared systems, not architectures under identical decoding.

FLEURS is read speech. This study does not reproduce Humyn’s conversational BRIDGE audit or establish Indian-English, speaker-demographic fairness, mobile-device performance, or universal best-model claims. Sentence-cluster intervals do not provide speaker/population uncertainty. Native Whisper EOS/cap status is unknown; observed short-form caps remain included.

Before a paper claim is made, the remaining necessary work is independent linguistic listening/reference review and a clearly scoped contribution checked against prior work. Earlier Humyn public-sample comparisons affected by silence/reference problems remain withdrawn. The results here should not be presented as a finished submission or as all current open ASR models.

## Local replay

From `/Users/prateekraj/whispr-nano`, the analysis commands require no GPU:

```sh
.venv-bridge/bin/python -m research.bridge_open.finish_qwen_comparison research/bridge_open/artifacts/qwen_native_full_hi_v1 /tmp/qwen-comparison-recheck
.venv-bridge/bin/python -m research.bridge_open.finish_whisper_comparison research/bridge_open/artifacts/fleurs_native_long_correction /tmp/whisper-comparison-recheck
```

Choose output paths that do not already exist. Portable source archives retain the exact executed code. Frozen experiment nodes remain unchanged; any future inference variant must be a new child.
