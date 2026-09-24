# Full Hindi comparison

All systems cover the same 418 FLEURS Hindi official-test recordings, pinned audio/references, and primary scorer. These are declared system configurations, not a controlled architecture comparison.

| System | n | Word errors / reference words | WER | CER |
|---|---:|---:|---:|---:|
| Qwen3-ASR-1.7B | 418 | 1352 / 10326 | 13.09% | 5.65% |
| Whisper Turbo | 418 | 3237 / 10326 | 31.35% | 13.62% |
| Whisper Large-v3 | 418 | 2913 / 10326 | 28.21% | 11.72% |

Qwen uses the pinned native API, bfloat16, greedy generation, batch size up to eight, explicit Hindi, and full waveforms. Its official output postprocessing is included; raw pre-parser output/tokens remain saved. Whisper uses float32 HF beam-5 short-form inference and native long-form inference for the single >30-second Hindi recording. Precision, decoding and API processing differ.

Qwen integrity: 418 records, 418 observed EOS terminations, 0 observed token caps, 0 empty outputs. Runtime failures are not excluded from a partial table: complete coverage is required.

Paired WER differences (Qwen minus comparator), sentence-cluster percentile bootstrap, 2,000 draws, seed 12:
- Whisper Turbo: -18.25 percentage points; 95% interval [-22.07, -15.76], 265 sentence clusters.
- Whisper Large-v3: -15.12 percentage points; 95% interval [-18.89, -12.73], 265 sentence clusters.

Intervals describe sentence resampling on this fixed test set. No speaker-population, demographic-fairness, Indian-English, noisy-conversation, mobile-device, or BRIDGE-equivalent conclusion follows. These post-result engineering checks are not an untouched method-development evaluation. No human semantic transcript review was completed.

Measured billing is unavailable. The shared HF guard records conservative reservations, not an invoice.

Raw pre-parser decoded text exactly matches the scored API transcript on 418/418 recordings (ignoring outer whitespace).
