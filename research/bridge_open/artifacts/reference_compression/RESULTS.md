# Reference-text compression diagnostic

This CPU-only stress test uses released reference text, the pinned Whisper Turbo tokenizer, and the token-byte zlib statistic implemented by Transformers 4.57.6. It does not decode audio or change any ASR score. Full nonoverlapping windows exclude the remainder of each transcript.

| Language | Tokens per window | Windows | Median ratio | Above 1.35 | Above 2.4 |
|---|---:|---:|---:|---:|---:|
| as | 200 | 116 | 2.254 | 100.0% | 19.0% |
| as | 400 | 58 | 2.867 | 100.0% | 100.0% |
| bn | 200 | 301 | 2.299 | 100.0% | 28.9% |
| bn | 400 | 149 | 2.930 | 100.0% | 100.0% |
| gu | 200 | 353 | 2.817 | 100.0% | 100.0% |
| gu | 400 | 176 | 3.540 | 100.0% | 100.0% |
| hi | 200 | 145 | 1.481 | 94.5% | 0.0% |
| hi | 400 | 71 | 1.848 | 100.0% | 0.0% |
| kn | 200 | 174 | 2.174 | 100.0% | 4.0% |
| kn | 400 | 86 | 2.763 | 100.0% | 96.5% |
| ml | 200 | 310 | 2.469 | 100.0% | 73.5% |
| ml | 400 | 155 | 3.162 | 100.0% | 100.0% |
| mr | 200 | 62 | 1.441 | 100.0% | 0.0% |
| mr | 400 | 31 | 1.814 | 100.0% | 0.0% |
| ta | 200 | 127 | 1.439 | 86.6% | 0.0% |
| ta | 400 | 63 | 1.782 | 100.0% | 0.0% |
| te | 200 | 434 | 2.286 | 100.0% | 11.8% |
| te | 400 | 216 | 2.837 | 100.0% | 100.0% |

Ratios depend on token-window length and language. A common threshold can flag reference-derived text that is not an ASR repetition loop. These are arbitrary text windows rather than audio-aligned correct hypotheses, so this table is not a false-positive-rate estimate, a language fairness test, or proof that a different threshold improves WER. Source transcript quality and sampling also require independent review.

The 2.4 column applies a second numerical cutoff to the same token-byte statistic; it must not be confused with applying the original OpenAI Whisper text-compression detector, which uses a different representation. Thresholds were not tuned against these references for any completed decoding run.

Experiment a3bc8d94-81e1-44c9-98ba-351317cbb4b2; run a636d2b7-8cc8-48a9-820a-17b590ca8b5d; commit 2a6d706. Tokenizer file hashes and zlib version are in evidence.json; every window is in windows.csv.
