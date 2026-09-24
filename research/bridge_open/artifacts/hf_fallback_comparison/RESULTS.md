> **Audio QC correction (2026-09-22):** Five public files contain transcript text over exactly silent audio. Affected Assamese/Bengali quality comparisons and aggregates containing them are withdrawn pending adjudication. Original metrics below are preserved as historical records. See the [source-verified audit](../audio_content_audit/RESULTS.md).

# Matched HF repetition-fallback diagnostic

One previously exposed Bengali conversation (humyn-01-01, 15.05 minutes), inference seed 12. Both conditions use identical frozen audio, references, Turbo weights, code, package versions, A100 GPU, float32 precision and scoring.

| Condition | WER | CER | Selected calls without EOS | Total decoding attempts | Inference time |
|---|---:|---:|---:|---:|---:|
| No fallback | 98.28% | 76.73% | 25/31 | 31 | 95.0 s |
| Compression fallback | 99.83% | 89.51% | 2/31 | 176 | 271.9 s |

Fallback changes WER by +1.56 percentage points and CER by +12.78 points. Negative values favor fallback.

The changed recipe uses the Transformers token-byte compression threshold 1.35 and temperatures 0, 0.2, 0.4, 0.6, 0.8, 1.0. Previous-token context, log-probability fallback and silence skipping are disabled in both conditions. The token cap is unchanged. This compares the Transformers native temperature-fallback recipe. Its sampling transition also leaves later temperature-zero windows at beam 1; the observed WER change cannot be attributed to the compression threshold alone. It is not a controlled beam-5 comparison or a test of all recommended Whisper decoding options.

Effective beam counts across fallback attempts: {'5': 1, '1': 175}. Sampling switches to one beam; describing every attempt as beam 5 would be incorrect. 24 windows exhausted their temperature schedule while still triggering the detector.

All rejected and selected sequences are retained. Hash checks, full input coverage, compression ratios, retry decisions and WER/CER were independently replayed. This is a development diagnostic, not a new ASR method, a held-out improvement, demographic evidence or a publication acceptance claim.
