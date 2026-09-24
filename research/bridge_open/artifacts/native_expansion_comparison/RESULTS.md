> **Audio QC correction (2026-09-22):** Five public files contain transcript text over exactly silent audio. Affected Assamese/Bengali quality comparisons and aggregates containing them are withdrawn pending adjudication. Original metrics below are preserved as historical records. See the [source-verified audit](../audio_content_audit/RESULTS.md).

# Native versus fixed-window Turbo across nine languages

All 22 matched conversations were fully processed and independently replayed against immutable source hashes. This is the same public development cohort used previously. No held-out or demographic claim is supported.

| Language | Conversations | Fixed 30 s WER % | Native WER % | Difference (points) | Fixed CER % | Native CER % |
|---|---:|---:|---:|---:|---:|---:|
| as | 2 | 99.85 | 99.74 | -0.10 | 99.19 | 98.88 |
| bn | 3 | 98.67 | 98.37 | -0.30 | 86.43 | 77.81 |
| gu | 3 | 93.14 | 85.79 | -7.35 | 81.83 | 66.11 |
| hi | 3 | 52.02 | 46.09 | -5.94 | 43.23 | 34.34 |
| kn | 3 | 86.76 | 88.99 | +2.23 | 58.79 | 44.60 |
| ml | 2 | 160.39 | 99.91 | -60.47 | 110.15 | 85.18 |
| mr | 1 | 86.67 | 85.35 | -1.31 | 66.76 | 60.32 |
| ta | 2 | 99.39 | 93.66 | -5.73 | 78.94 | 65.14 |
| te | 3 | 95.12 | 93.52 | -1.59 | 77.99 | 71.25 |

Native decoding improves WER on 17/22 conversations. Median paired change: -1.98 points (descriptive IQR -6.39 to -0.37).

Negative differences favor native decoding. No confidence interval or significance claim: speaker independence is unverified and each language has few conversations. Cross-language pooled scores in JSON describe this exact file mixture only.

Both recipes use the same Turbo weights, float32, H100, beam 5 and scoring. Native uses timestamp-guided sequential decoding, batch 1, no previous-token context and no temperature fallback. Fixed uses independent 30 s audio windows, batch 8. This is a recipe comparison; batching and segmentation are not isolated. All capped generations stay in the primary scores.

Uncapped-in-both diagnostic subset: 0/22 conversations. This subset is selected on model outcomes and cannot replace the primary comparison.

The per-conversation CSV includes S/D/I changes, output/reference length ratios, capped underlying calls and the native input-coverage trace summary. The independent validator checks every trace transition and final frame coverage; it does not infer coverage from predicted timestamp endpoints.
