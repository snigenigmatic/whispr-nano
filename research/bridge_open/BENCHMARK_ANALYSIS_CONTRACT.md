# Open benchmark analysis contract v1

`analyze_benchmark.py` is a CPU-only descriptive analyzer for completed
`run_whisper.py`/`run_qwen.py` style artifacts. It reports per-conversation
status, S/D/I counts, WER/CER, capped chunks, real-time factor, and available
CUDA peak bytes; per-language micro WER/CER and mean conversation WER; and
macro language WER (the unweighted mean of per-language micro WERs); the
conversation-mean variant is exposed separately as `macro_conversation_wer`.

All `success`, `empty_output`, and `generation_limit` records with metrics are
scored. Capped output remains in the estimand. Failures and unsupported or
excluded items are counted as coverage/failure and never converted to zero
errors. Duplicate model/revision/condition/conversation identities are fatal.

Comparisons use only exact shared conversation IDs. Paired exploratory
bootstrap deltas use 2,000 repetitions, seed `20260922`, and language-stratified
conversation resampling. The output contains descriptive percentile intervals
only; no p-values are reported. Intervals are suppressed below five shared
conversations. Results are explicitly exploratory on a sampled corpus and do
not support population, architecture-only, or publication claims; each run
retains its concrete model revision and decoding recipe.
