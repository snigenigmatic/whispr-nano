# Controlled decoding results

Same three already-exposed Hindi conversations; 6,503 reference words. All source hashes, audio coverage, token termination and scores independently checked. No model training. These are development results, not population estimates.

| Recipe | WER % | CER % | Substitution % | Deletion % | Insertion % | Capped generation calls |
|---|---:|---:|---:|---:|---:|---:|
| Fixed30s, beam5 | 52.02 | 43.23 | 24.93 | 25.33 | 1.77 | 4/89 |
| Fixed15s, beam5 | 56.79 | 46.60 | 27.76 | 20.31 | 8.72 | 8/177 |
| Native longform, beam5 | 46.09 | 34.34 | 26.31 | 13.19 | 6.58 | 6/104 |

WER is corpus-level word errors divided by reference words. Capped generations remain in scores. Each native generation call can yield several timestamp segments; count underlying calls, not output segments. Fixed windows retain standard zero-padding to the Whisper encoder length.

The native recipe uses batch1, timestamp-based sequential segmentation and no previous-token conditioning or temperature fallback; fixed recipes use batch8. This compares inference recipes, not a one-variable native-decoder ablation. Only the fixed15s versus fixed30s comparison changes one decoding setting. Runtime is descriptive and includes different batching. No significance claim or confidence interval is made from three conversations.

Native coverage is verified from actual input-frame windows and advancing seek offsets. Timestamp segment endpoints alone are not used to infer audio coverage. All text is scored without repetition cleanup, transliteration or numeral conversion.

The run configuration freezes each input artifact hash. Raw predictions, checkpoints identified by immutable revisions and weight hashes, native traces, resource receipts and provider billing evidence remain available in the surrounding evidence package. Resource receipts are estimates, not final provider bills.
