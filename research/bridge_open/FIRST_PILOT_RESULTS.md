> **Audio QC correction (2026-09-22):** Five public files contain transcript text over exactly silent audio. Affected Assamese/Bengali quality comparisons and aggregates containing them are withdrawn pending adjudication. Original metrics below are preserved as historical records. See the [source-verified audit](artifacts/audio_content_audit/RESULTS.md).

# First public Hindi pilot — Whisper-small

Run `81da5556-52b1-4eb4-b3e5-b3bedb6e3903`, commit `835a8000aab9cc71a57b4e4cc0f415a5f5d9332a`. All three full conversations completed and artifact validation passed. Total audio: 43.92 minutes; run wall time: 9.46 minutes; cloud cost: $0.

These are poor baseline results under the frozen greedy, fixed-30-second-window setup. All three conversations contained generation-capped windows. Preserve those errors in the denominator. WER can exceed 100% because insertions add errors beyond the reference word count. This pilot does not establish how the model performs under its strongest long-form decoding configuration.

| Conversation | WER | CER | Status |
|---|---:|---:|---|
| humyn-13-00 | 108.88% | 87.98% | generation_limit |
| humyn-13-01 | 103.60% | 81.81% | generation_limit |
| humyn-14-00 | 106.98% | 84.26% | generation_limit |

Pooled WER: **106.46%** (6923 edits / 6503 reference words). Pooled CER: **84.59%**. No population inference from three public recordings.

Raw predictions, chunk boundaries, token IDs, metrics, exact configuration, model/config/tokenizer hashes and environment are in `artifacts/hindi_small_v1/`. `validation.json` confirms complete audio coverage and recomputed scores. Scorer/acquisition/validator tests: 19 passed.

The first 30 seconds of Hindi_1 were used for engineering checks, so this is a development pilot. It is not a held-out result or a numerical comparison with BRIDGE. Next comparison: the same three files with Whisper-large-v3-turbo; no hyperparameters changed in response to WER.
