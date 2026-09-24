# Core ASR quality results

Each accepted cell is independently rejoined to frozen source/reference bytes and rescored. Pending cells have no quality value.

| Model | Track | Status | N | Micro WER (%) | Micro CER (%) | Mean utterance WIL (%) | Empty (normalized) | WER ≥100% |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| whisper-small | fleurs-hi | accepted | 418 | 57.80% | 34.50% | 72.60% | 0 | 14 |
| whisper-small | fleurs-bn | accepted | 920 | 194.58% | 121.45% | 99.99% | 40 | 916 |
| whisper-small | fleurs-as | accepted | 984 | 214.50% | 131.69% | 99.99% | 42 | 979 |
| whisper-small | svarah-recovered-en | accepted | 518 | 9.69% | 4.76% | 17.55% | 0 | 32 |
| whisper-turbo | fleurs-hi | accepted | 418 | 31.35% | 13.62% | 46.45% | 0 | 2 |
| whisper-turbo | fleurs-bn | accepted | 920 | 85.86% | 35.61% | 92.46% | 0 | 200 |
| whisper-turbo | fleurs-as | accepted | 984 | 134.71% | 116.35% | 99.97% | 1 | 965 |
| whisper-turbo | svarah-recovered-en | accepted | 518 | 12.47% | 5.58% | 16.55% | 0 | 36 |
| whisper-large-v3 | fleurs-hi | accepted | 418 | 28.21% | 11.72% | 42.87% | 0 | 3 |
| whisper-large-v3 | fleurs-bn | accepted | 920 | 74.70% | 27.86% | 86.21% | 0 | 55 |
| whisper-large-v3 | fleurs-as | accepted | 984 | 113.30% | 108.23% | 99.98% | 0 | 969 |
| whisper-large-v3 | svarah-recovered-en | accepted | 518 | 8.24% | 4.28% | 16.03% | 0 | 41 |
| qwen3-asr-0.6b | fleurs-hi | accepted | 418 | 18.50% | 9.55% | 29.65% | 0 | 0 |
| qwen3-asr-0.6b | svarah-recovered-en | accepted | 518 | 16.39% | 10.26% | 21.49% | 0 | 49 |
| qwen3-asr-1.7b | fleurs-hi | accepted | 418 | 13.09% | 5.65% | 21.80% | 0 | 0 |
| qwen3-asr-1.7b | svarah-recovered-en | accepted | 518 | 13.60% | 9.21% | 18.04% | 0 | 41 |
| omniasr-ctc-1b | fleurs-hi | accepted | 418 | 21.50% | 10.04% | 33.43% | 0 | 18 |
| omniasr-ctc-1b | fleurs-bn | accepted | 920 | 24.52% | 6.51% | 39.13% | 0 | 0 |
| omniasr-ctc-1b | fleurs-as | accepted | 984 | 33.76% | 9.07% | 51.61% | 0 | 0 |
| omniasr-ctc-1b | svarah-recovered-en | accepted | 518 | 64.63% | 52.72% | 75.44% | 26 | 315 |

No aggregate winner is reported: cells use different models, languages, and fixed source cohorts.
