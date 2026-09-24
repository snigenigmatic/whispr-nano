# Svarah text-error diagnostics

Post-hoc inspection of the accepted 518-record subset. These checks describe written outputs and edit counts; they do not establish spoken language, semantic accuracy or the cause of a model failure. Primary scoring and cohort membership are unchanged.

| System | Outputs with non-Latin letters | Outputs mixing Arabic and Latin letters | Outputs with no letters |
|---|---:|---:|---:|
| qwen | 2 | 0 | 0 |
| omni | 309 | 82 | 27 |

Unicode letter names are a script heuristic, not a language classifier. Empty outputs and punctuation-only outputs count as having no letters. Arabic-script characters do not by themselves establish Arabic language.

| System | Duration | Recordings | Reference words | Micro WER |
|---|---|---:|---:|---:|
| qwen | 0–<2 s | 133 | 222 | 25.23% |
| qwen | 2–<5 s | 169 | 1378 | 10.23% |
| qwen | ≥5 s | 216 | 3789 | 14.15% |
| omni | 0–<2 s | 133 | 222 | 83.78% |
| omni | 2–<5 s | 169 | 1378 | 68.87% |
| omni | ≥5 s | 216 | 3789 | 61.97% |

Duration bins were chosen after the primary results. They are descriptive, not a pre-registered hypothesis test. The gap also occurs among recordings ≥5 s, so it is not confined to very short utterances. That observation does not isolate the effect of duration, accent or audio quality.

The [machine-readable profile](analysis.json) preserves letter counts, word-edit components, per-record flags and the 12 largest paired disagreements with references and both outputs. Selection emphasizes disagreements and cannot estimate error prevalence. No listening judgments have been filled in.

## Next controlled question

The declared conditions provide different language information: Qwen receives explicit English, whereas the native Omni CTC path has no corresponding language input. This comparison does not isolate model architecture from language conditioning.

Verify the most discrepant audio/reference pairs and inspect the native CTC frame/blank and output traces. If testing a script-constrained decoder or input preprocessing, define the rule on separate development data and compare against the unchanged accepted native baseline. Treat this inspected Svarah subset as exploratory; reserve independent confirmation for a frozen method. A Unicode/transliteration patch alone must not be reported as recovered spoken meaning.
