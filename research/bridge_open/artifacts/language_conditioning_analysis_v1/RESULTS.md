# Native Whisper language-conditioning diagnostic

Both arms use the same checkpoint, float32 evaluation state, complete waveform, seed and native decoding settings. Only the supplied language differs. This deliberately exposed diagnostic does not establish an untouched-test improvement.

| Cohort | Language | n | Forced WER | Automatic WER | Difference (pp) | Script mismatches forced / auto | Unknown scripts forced / auto |
|---|---|---:|---:|---:|---:|---:|---:|
| all_diagnostic | hi | 11 | 34.74% | 59.09% | +24.35 | 0 / 4 | 0 / 0 |
| all_diagnostic | bn | 19 | 81.82% | 81.82% | +0.00 | 0 / 0 | 0 / 0 |
| all_diagnostic | as | 14 | 108.06% | 94.33% | -13.73 | 14 / 0 | 0 / 1 |
| sampled_short | hi | 10 | 29.11% | 60.76% | +31.65 | 0 / 4 | 0 / 0 |
| sampled_short | bn | 10 | 69.77% | 69.77% | +0.00 | 0 / 0 | 0 / 0 |
| sampled_short | as | 10 | 113.58% | 95.68% | -17.90 | 10 / 0 | 0 / 1 |
| all_long | hi | 1 | 53.52% | 53.52% | +0.00 | 0 / 0 | 0 / 0 |
| all_long | bn | 9 | 88.28% | 88.28% | +0.00 | 0 / 0 | 0 / 0 |
| all_long | as | 4 | 102.89% | 93.06% | -9.83 | 4 / 0 | 0 / 0 |

The 30 short recordings were selected by a fixed hash order, ten per language. All 14 recordings longer than 30 seconds were added separately. These strata must not be pooled to estimate corpus prevalence.

Script matching is a Unicode diagnostic, not a judgment of spoken language or transcript meaning. Native decoding strips EOS from its public result; generation-cap counts remain unknown. Any timestamp flags are retained in analysis.json. No human adjudication is implied.
