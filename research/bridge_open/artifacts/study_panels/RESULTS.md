> **Audio QC correction (2026-09-22):** Five public files contain transcript text over exactly silent audio. Affected Assamese/Bengali quality comparisons and aggregates containing them are withdrawn pending adjudication. Original metrics below are preserved as historical records. See the [source-verified audit](../audio_content_audit/RESULTS.md).

# Open study diagnostic panels — 22 September 2026

Local CPU analysis of seven validated artifact directories representing six model recipes. No transcription, model training, paid APIs or cloud calls occurred. These are post-hoc descriptive analyses of the existing sample, not additional independent observations or an exact BRIDGE replication.

## Hindi error decomposition

Same three development conversations and 6,503 reference words for every row. Rates below use reference-word counts; S + D + I equals WER before rounding. WIL is pooled from alignment counts as `1 - H²/(N_ref × N_hyp)` (1 when there are no hypothesis words); it is not the mean of conversation WIL values. Both forms are available in the JSON/CSV.

| Recipe | Substitutions | Deletions | Insertions | WER | Pooled WIL |
|---|---:|---:|---:|---:|---:|
| openai/whisper-large-v3-turbo | 24.93% | 25.33% | 1.77% | 52.02% | 67.63% |
| openai/whisper-small | 72.61% | 12.01% | 7.89% | 92.51% | 97.53% |
| Qwen/Qwen3-ASR-0.6B | 23.54% | 6.50% | 2.08% | 32.12% | 48.80% |
| Qwen/Qwen3-ASR-1.7B | 17.04% | 5.63% | 1.92% | 24.59% | 37.89% |
| openai/whisper-large-v3 | 24.04% | 17.32% | 5.60% | 46.95% | 61.04% |
| facebook/omniASR-CTC-1B | 42.66% | 35.86% | 0.22% | 78.73% | 92.83% |

For example, Qwen 1.7B's deletion rate is 5.63%, against Turbo's 25.33% in this recipe comparison. Alignment deletions measure missing reference tokens; script mismatch, reference errors and alternate phrasing can affect the alignment, so these rates do not prove when or why audio was lost.

## Routing bounds

All selectors below use evaluation references. They are **optimistic in-sample oracle bounds**, not cross-validated routing performance. Every row preserves identical conversations and the same word denominator. Selection ties are broken deterministically by model name.

| Cohort | Models | Conversations | Best fixed model WER | Language oracle WER | Per-conversation oracle WER |
|---|---:|---:|---:|---:|---:|
| hindi_development | 6 | 3 | 24.59% | 24.59% | 24.59% |
| shared_nine_languages | 3 | 22 | 84.13% | 78.87% | 78.79% |

Qwen 1.7B has the lowest WER on each of the three Hindi files, so this sample gives no per-file routing gain over that fixed model. On the nine-language cohort, the oracle pool is only Small, Turbo and Omni. Its very high errors and decoder/script limitations make it unsuitable for deployment conclusions. Do not compare that pooled score to the six-model Hindi score or Humyn's leaderboard.

The JSON also retains exact file IDs, selected models, per-file min/max WER and descriptive paired Pearson correlations. Correlations are suppressed below five recordings or for constant scores; they have no p-values, do not control for language and cannot separate model quality from recording difficulty. Unknown cross-file speaker overlap further limits inference.

## Files and validation

- [All panels and provenance](panels.json)
- [Per-language error profiles](error_profiles.csv)
- [Study scope and remaining requirements](../../FULL_STUDY_SPEC.md)
- [Original full model comparison](../../EXPANDED_RESULTS_20260922.md)

The CPU runner independently validates all input data/reference/model/config hashes, chunk coverage and scores before aggregation. New tests cover rates exceeding 100%, pooled-denominator oracle selection, unequal cohorts and mismatched reference lengths (3 passed).

ORX experiment: `92faa204-73e8-423b-836a-71a139d7019e`; frozen source commit: `fd9b7b6`; completed run: `9e171f51-90d3-4f40-af8f-90fde914e9b5`. The fixed entrypoint dispatches the committed `study_panels` executor and prints configuration plus complete results. Original GPU experiment branches are unchanged.
