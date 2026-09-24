# Audio/reference content audit — 2026-09-22

## Finding

Five of the 37 pinned public conversations contain nonempty transcript intervals whose audio samples are all exactly zero. They are the two Assamese files and all three Bengali files. This was missed by the original structural/hash checks. These affected full-conversation WER/CER results must not be interpreted as model quality; mixed aggregates containing them are withdrawn pending adjudication. Raw predictions, metrics and frozen ORX nodes remain preserved.

| Conversation | Language | Zero samples | All-zero intervals with text | Raw reference words in those intervals |
|---|---|---:|---:|---:|
| humyn-00-00 | as | 55.27% | 3 | 50/442 |
| humyn-00-01 | as | 43.47% | 3 | 82/1458 |
| humyn-01-00 | bn | 91.13% | 12 | 1253/1649 |
| humyn-01-01 | bn | 93.86% | 6 | 1667/1795 |
| humyn-02-00 | bn | 87.98% | 14 | 1546/1812 |

Word counts above are raw whitespace counts, not the normalized WER denominator. The all-zero interval test uses fully in-bounds [floor(start × sr), ceil(end × sr)) samples and does not inspect ASR predictions.

## Source verification

All 37 prepared arrays exactly match reconstruction from the pinned original encoded audio after the documented downmix/resampling. All source parquet, encoded-audio, prepared-WAV and reference hashes passed. All source files have one channel, ruling out destructive stereo downmix here. An independent scipy WAV decoder also reproduces the Bengali zero-sample mask. The tested Bengali file begins with over 203 seconds of exact zeros; six silent reference intervals contain 1667 of 1795 raw reference words. This is an observed mismatch in this public release, not evidence about the private BRIDGE evaluation data or its published rankings.

## Effect on results

- Withdraw the Bengali Turbo/fallback/Large-v3 WER comparisons as model-quality evidence. All three completed HF jobs used this same affected input. Hardware reproducibility of the identical input/output remains a computational observation only.
- Withdraw Assamese/Bengali quality comparisons and mixed 22-conversation headline aggregates. Other files are not invalidated automatically, but absence of this exact-silence flag is not human validation.
- Twenty-five conversations remain after combining prior structural checks with this exact-silence screen. No replacement benchmark score is reported here; selecting a repaired evaluation panel must be explicit and versioned.
- Preserve the source revision, hashes, predictions and historical metrics. Do not silently remove reference words or splice out silence to improve WER.
- Suspend GPU expansion until audio/reference alignment is reviewed, including high-zero Gujarati files not caught by the exact-interval criterion.

## Execution

This CPU-only audit completed in 10 seconds with no cloud compute. ORX experiment `29b15b38-1cdb-40f9-b620-2ec3a4b0889b`, run `8cc9e313-fcbf-4bc2-8e58-af74d897ede2`, commit `fa5bc6262dc45777713a46ac151da551ae145a8b`. See [full evidence](evidence.json), [independent decoder check](independent_decoder_check.json) and [ORX receipt](orx_receipt.json).
