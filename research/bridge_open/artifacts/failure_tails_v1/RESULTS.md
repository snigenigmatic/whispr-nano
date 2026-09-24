# Descriptive failure tails on the frozen FLEURS benchmark

This post-hoc analysis reuses accepted predictions; no new inference or model selection was performed. The threshold WER ≥ 100% means word edits meet or exceed the reference word count. It does not certify a semantic failure. Percentiles describe recordings in this corpus, not a population confidence interval.

| System | Language | n | Micro WER | Median WER | 95th percentile WER | WER ≥ 100% (count) | Zero word errors (count) |
|---|---|---:|---:|---:|---:|---:|---:|
| Whisper Turbo | as | 984 | 134.71% | 100.00% | 253.39% | 965 | 0 |
| Whisper Turbo | bn | 920 | 85.86% | 83.33% | 114.29% | 200 | 0 |
| Whisper Turbo | hi | 418 | 31.35% | 28.57% | 57.32% | 2 | 2 |
| Whisper Large-v3 | as | 984 | 113.30% | 104.08% | 130.00% | 969 | 0 |
| Whisper Large-v3 | bn | 920 | 74.70% | 70.59% | 100.00% | 55 | 0 |
| Whisper Large-v3 | hi | 418 | 28.21% | 26.12% | 50.00% | 3 | 3 |
| Qwen3-ASR-1.7B | hi | 418 | 13.09% | 11.76% | 31.66% | 0 | 53 |
| OmniASR CTC-1B | as | 984 | 33.76% | 33.33% | 60.00% | 0 | 8 |
| OmniASR CTC-1B | bn | 920 | 24.52% | 23.81% | 47.07% | 0 | 26 |
| OmniASR CTC-1B | hi | 418 | 21.50% | 17.27% | 63.32% | 18 | 12 |

## Paired recording counts

| Omni versus | Language | Fewer word edits | Equal word edits | More word edits |
|---|---|---:|---:|---:|
| Whisper Turbo | hi | 326 | 32 | 60 |
| Whisper Turbo | bn | 918 | 2 | 0 |
| Whisper Turbo | as | 984 | 0 | 0 |
| Whisper Large-v3 | hi | 294 | 53 | 71 |
| Whisper Large-v3 | bn | 915 | 1 | 4 |
| Whisper Large-v3 | as | 984 | 0 | 0 |
| Qwen3-ASR-1.7B | hi | 76 | 75 | 267 |

These system configurations differ in model training, precision and decoding. Pair counts weight recordings equally; micro WER weights reference words. Neither establishes demographic fairness, human semantic accuracy, nor a trained improvement. Input/source hashes and total cohort coverage were rechecked. The blinded review sample remains fixed independently of this additional analysis.
