# OmniASR CTC complete-cohort comparison

All 2,322 FLEURS test recordings use the pinned native CTC pipeline, one full waveform per recording, BF16, evaluation mode, and the same primary scorer as the other systems. The acceptance cohort verified official API/token-decoder equality. Full coverage and independent rescoring passed.

| Language | n | Word errors / reference words | WER | CER | Script mismatches |
|---|---:|---:|---:|---:|---:|
| hi | 418 | 2220 / 10326 | 21.50% | 10.04% | 17 |
| bn | 920 | 4355 / 17764 | 24.52% | 6.51% | 0 |
| as | 984 | 6554 / 19414 | 33.76% | 9.07% | 0 |

Paired differences are Omni minus comparator in percentage points; 95% sentence-cluster bootstrap intervals (2,000 draws, seed 12):
- hi, Whisper Turbo: -9.85 [-14.52, -6.27].
- hi, Whisper Large-v3: -6.71 [-11.24, -3.24].
- hi, Qwen3-ASR-1.7B: +8.41 [+6.36, +10.73].
- bn, Whisper Turbo: -61.34 [-64.44, -58.74].
- bn, Whisper Large-v3: -50.19 [-56.19, -46.20].
- as, Whisper Turbo: -100.95 [-112.55, -91.01].
- as, Whisper Large-v3: -79.55 [-86.42, -74.67].

These are comparisons of declared configurations, with different precision, decoding and model training. Training-data overlap is not independently ruled out. No human transcript adjudication or speaker-population inference is implied. Script matching does not establish correct meaning. Provider charges remain unverified.
