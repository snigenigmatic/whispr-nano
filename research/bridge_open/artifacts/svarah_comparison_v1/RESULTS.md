# Svarah recovered-shard comparison

**Scope: 518 source-verified recordings from one recovered shard, not the complete 6,656-record corpus.** The other two cached shards failed transfer. All available valid recordings are included, including 16 clipping-flagged recordings. This availability-selected subset includes 209 previously exposed IDs. No speaker IDs were inferred. The 12-item acceptance cohort overlaps this table and is not additional evidence.

| Declared native system | Scored recordings | Micro WER | Micro CER | Empty output | Recordings with WER ≥ 100% |
|---|---:|---:|---:|---:|---:|
| Qwen3-ASR-1.7B | 518 | 13.60% | 9.21% | 0 | 41 |
| OmniASR-CTC-1B | 518 | 64.63% | 52.72% | 26 | 315 |

On paired recordings, Qwen has fewer word edits in 412, the counts are equal in 93, and Omni has fewer in 13. These are descriptive finite-cohort comparisons. WER ≥ 100% counts word edits; it is not a human-verified semantic failure category.

## Protocol and acceptance

The source WAV bytes are unchanged mono 16 kHz PCM16. Every reference/audio pair was independently compared to its original Arrow row and the pinned 6,656-row metadata. Every recording is decoded in full (0.1993125–27.8063125 seconds). No decoder, model revision or weights changed after the 12-item runtime check. Native BF16 Qwen uses explicit English, greedy batch 8; native BF16 Omni CTC uses batch 1 and verifies public-API/retained-token agreement on every output. Results compare these declared systems, not architecture in isolation.

Both runs completed all 518 predictions, passed source/model/code hash checks and independent rescoring, and retained tokens/termination evidence. Use [analysis.json](analysis.json) for exact counts, accepted job URLs and immutable source receipts; [paired counts](paired_counts.jsonl) for each source recording. The [ranked disagreements](largest_disagreements.csv) are a review aid, not completed human judgments. Worker runtime and reservations are not verified provider billing.

The frozen primary scorer applies Unicode NFC/casefold and punctuation-to-space normalization, preserving marks and making no numeral conversion. No post-hoc text normalization replaces this primary result.

## Limits and decision

This is an exploratory Indian-English extension of the existing system audit. It neither estimates all Svarah speakers nor isolates accent effects. Public-data training overlap is unknown. There are no speaker-level confidence intervals or p-values because reliable speaker mapping is unavailable. Reference/source checks are structural; fluent human listening is pending. This table does not establish a new recognition method, fairness, phone latency or publication novelty.

Next: examine paired disagreements with the audio, distinguish reference/normalization issues from recognition errors, and define any repair on development data before using an untouched confirmation set. Do not choose a repair and validate it on this same inspected subset.
