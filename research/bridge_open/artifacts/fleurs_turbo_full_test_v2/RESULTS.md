> **RESULT HOLD:** Input and native Large-v3 diagnostic checks are complete. The general model-performance comparison remains on hold for decoding-policy and model-specific acceptance work. See the [audit](../../FLEURS_DILIGENCE_REPORT.md).

# FLEURS comparison

Descriptive public FLEURS benchmark; not population, speaker, or fairness evidence.

## Per-language combined micro metrics

| Model | Language | WER | CER | Scored | Capped | Normalized-empty* | Empty-output status |
|---|---:|---:|---:|---:|---:|---:|---:|
| Turbo (openai/whisper-large-v3-turbo@41f01f3fe87f28c78e2fbf8b568835947dd65ed9) | as | 136.79% | 117.16% | 984 | 65 | 1 | 0 |
| Turbo (openai/whisper-large-v3-turbo@41f01f3fe87f28c78e2fbf8b568835947dd65ed9) | bn | 86.02% | 35.68% | 920 | 53 | 0 | 0 |
| Turbo (openai/whisper-large-v3-turbo@41f01f3fe87f28c78e2fbf8b568835947dd65ed9) | hi | 32.64% | 14.10% | 418 | 3 | 0 | 0 |

*Normalized-empty counts overlap capped outputs; they diagnose an empty scoring string after fixed normalization, not an exclusive terminal status.

## Route-specific micro metrics

| Model | Language | Route | WER | CER | Scored | Capped | Normalized-empty* | Empty-output status |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| Turbo (openai/whisper-large-v3-turbo@41f01f3fe87f28c78e2fbf8b568835947dd65ed9) | as | batched_shortform | 135.02% | 116.51% | 980 | 62 | 1 | 0 |
| Turbo (openai/whisper-large-v3-turbo@41f01f3fe87f28c78e2fbf8b568835947dd65ed9) | as | fixed_30s_complete_audio | 333.53% | 186.73% | 4 | 3 | 0 | 0 |
| Turbo (openai/whisper-large-v3-turbo@41f01f3fe87f28c78e2fbf8b568835947dd65ed9) | bn | batched_shortform | 85.79% | 35.10% | 911 | 46 | 0 | 0 |
| Turbo (openai/whisper-large-v3-turbo@41f01f3fe87f28c78e2fbf8b568835947dd65ed9) | bn | fixed_30s_complete_audio | 96.01% | 58.84% | 9 | 7 | 0 | 0 |
| Turbo (openai/whisper-large-v3-turbo@41f01f3fe87f28c78e2fbf8b568835947dd65ed9) | hi | batched_shortform | 31.02% | 13.36% | 417 | 2 | 0 | 0 |
| Turbo (openai/whisper-large-v3-turbo@41f01f3fe87f28c78e2fbf8b568835947dd65ed9) | hi | fixed_30s_complete_audio | 266.20% | 112.70% | 1 | 1 | 0 | 0 |

## Paired WER differences

Difference is left minus right in percentage points. Intervals use 2,000 seed-12 source-sentence cluster bootstrap draws within language.

| Left | Right | Language | Difference | 95% interval | Sentence clusters |
|---|---|---:|---:|---:|---:|

## Batch throughput

Batch throughput is aggregate audio seconds divided by aggregate timed batch wall seconds. It is not per-request latency.

| Model | Unique batches | Timed batches | Audio seconds | Batch wall seconds | Throughput |
|---|---:|---:|---:|---:|---:|
| Turbo (openai/whisper-large-v3-turbo@41f01f3fe87f28c78e2fbf8b568835947dd65ed9) | 290 | 290 | 29230.38 | 1669.04 | 17.51 audio s/wall s |

These descriptive public-benchmark summaries retain empty and generation-limited outputs. They do not support speaker, population, or fairness claims.
