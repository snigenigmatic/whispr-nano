# Controlled ASR pilot: evidence packet

Evidence status: **incomplete**. No scientific promotion decision is made.

Development-only pilot evidence; no final-test predictions read or scored

filtered10_gold minus random_matched_gold, final English L1-macro WER; positive means filtered worse

## All observed runs

| Arm | Seed | Normalizer | Baseline micro WER % | Final micro WER % | Final L1 macro WER % | Worst L1 WER % |
|---|---:|---|---:|---:|---:|---:|
| filtered10_gold | 0 | english | 28.818 | 24.690 | 24.898 | 37.013 |
| filtered10_gold | 0 | minimal | 29.663 | 24.972 | 25.170 | 37.241 |
| filtered10_gold | 1 | english | 28.818 | 25.085 | 25.315 | 36.429 |
| filtered10_gold | 1 | minimal | 29.663 | 25.444 | 25.663 | 36.788 |
| full_gold | 0 | english | 28.818 | 24.904 | 25.139 | 34.740 |
| full_gold | 0 | minimal | 29.663 | 25.264 | 25.493 | 34.974 |
| full_gold | 1 | english | 28.818 | 24.284 | 24.522 | 34.156 |
| full_gold | 1 | minimal | 29.663 | 24.702 | 24.934 | 34.391 |
| random_matched_gold | 0 | english | 28.818 | 24.758 | 25.019 | 35.065 |
| random_matched_gold | 0 | minimal | 29.663 | 25.107 | 25.362 | 35.363 |

## Every L1 and seed

| Arm | Seed | Normalizer | L1 | Final WER % | Clips | Speakers |
|---|---:|---|---|---:|---:|---:|
| filtered10_gold | 0 | english | Arabic | 28.753 | 147 | 1 |
| filtered10_gold | 0 | english | Hindi | 12.013 | 169 | 1 |
| filtered10_gold | 0 | english | Korean | 15.779 | 169 | 1 |
| filtered10_gold | 0 | english | Mandarin | 24.066 | 168 | 1 |
| filtered10_gold | 0 | english | Spanish | 31.766 | 152 | 1 |
| filtered10_gold | 0 | english | Vietnamese | 37.013 | 169 | 1 |
| filtered10_gold | 0 | minimal | Arabic | 28.816 | 147 | 1 |
| filtered10_gold | 0 | minimal | Hindi | 12.370 | 169 | 1 |
| filtered10_gold | 0 | minimal | Korean | 16.127 | 169 | 1 |
| filtered10_gold | 0 | minimal | Mandarin | 24.722 | 168 | 1 |
| filtered10_gold | 0 | minimal | Spanish | 31.746 | 152 | 1 |
| filtered10_gold | 0 | minimal | Vietnamese | 37.241 | 169 | 1 |
| filtered10_gold | 1 | english | Arabic | 30.993 | 147 | 1 |
| filtered10_gold | 1 | english | Hindi | 13.377 | 169 | 1 |
| filtered10_gold | 1 | english | Korean | 15.519 | 169 | 1 |
| filtered10_gold | 1 | english | Mandarin | 24.459 | 168 | 1 |
| filtered10_gold | 1 | english | Spanish | 31.114 | 152 | 1 |
| filtered10_gold | 1 | english | Vietnamese | 36.429 | 169 | 1 |
| filtered10_gold | 1 | minimal | Arabic | 31.050 | 147 | 1 |
| filtered10_gold | 1 | minimal | Hindi | 13.860 | 169 | 1 |
| filtered10_gold | 1 | minimal | Korean | 15.933 | 169 | 1 |
| filtered10_gold | 1 | minimal | Mandarin | 25.180 | 168 | 1 |
| filtered10_gold | 1 | minimal | Spanish | 31.169 | 152 | 1 |
| filtered10_gold | 1 | minimal | Vietnamese | 36.788 | 169 | 1 |
| full_gold | 0 | english | Arabic | 29.798 | 147 | 1 |
| full_gold | 0 | english | Hindi | 13.896 | 169 | 1 |
| full_gold | 0 | english | Korean | 14.481 | 169 | 1 |
| full_gold | 0 | english | Mandarin | 25.574 | 168 | 1 |
| full_gold | 0 | english | Spanish | 32.344 | 152 | 1 |
| full_gold | 0 | english | Vietnamese | 34.740 | 169 | 1 |
| full_gold | 0 | minimal | Arabic | 30.007 | 147 | 1 |
| full_gold | 0 | minimal | Hindi | 14.378 | 169 | 1 |
| full_gold | 0 | minimal | Korean | 14.832 | 169 | 1 |
| full_gold | 0 | minimal | Mandarin | 26.226 | 168 | 1 |
| full_gold | 0 | minimal | Spanish | 32.540 | 152 | 1 |
| full_gold | 0 | minimal | Vietnamese | 34.974 | 169 | 1 |
| full_gold | 1 | english | Arabic | 30.022 | 147 | 1 |
| full_gold | 1 | english | Hindi | 12.792 | 169 | 1 |
| full_gold | 1 | english | Korean | 13.961 | 169 | 1 |
| full_gold | 1 | english | Mandarin | 25.377 | 168 | 1 |
| full_gold | 1 | english | Spanish | 30.825 | 152 | 1 |
| full_gold | 1 | english | Vietnamese | 34.156 | 169 | 1 |
| full_gold | 1 | minimal | Arabic | 30.380 | 147 | 1 |
| full_gold | 1 | minimal | Hindi | 13.277 | 169 | 1 |
| full_gold | 1 | minimal | Korean | 14.443 | 169 | 1 |
| full_gold | 1 | minimal | Mandarin | 26.161 | 168 | 1 |
| full_gold | 1 | minimal | Spanish | 30.952 | 152 | 1 |
| full_gold | 1 | minimal | Vietnamese | 34.391 | 169 | 1 |
| random_matched_gold | 0 | english | Arabic | 31.143 | 147 | 1 |
| random_matched_gold | 0 | english | Hindi | 13.377 | 169 | 1 |
| random_matched_gold | 0 | english | Korean | 14.416 | 169 | 1 |
| random_matched_gold | 0 | english | Mandarin | 24.131 | 168 | 1 |
| random_matched_gold | 0 | english | Spanish | 31.983 | 152 | 1 |
| random_matched_gold | 0 | english | Vietnamese | 35.065 | 169 | 1 |
| random_matched_gold | 0 | minimal | Arabic | 31.348 | 147 | 1 |
| random_matched_gold | 0 | minimal | Hindi | 13.731 | 169 | 1 |
| random_matched_gold | 0 | minimal | Korean | 14.767 | 169 | 1 |
| random_matched_gold | 0 | minimal | Mandarin | 24.787 | 168 | 1 |
| random_matched_gold | 0 | minimal | Spanish | 32.179 | 152 | 1 |
| random_matched_gold | 0 | minimal | Vietnamese | 35.363 | 169 | 1 |

## Paired mechanism contrast

| Seed | Normalizer | Filtered minus random L1 macro WER (pp) |
|---:|---|---:|
| 0 | english | -0.121 |
| 0 | minimal | -0.192 |

## Exposure and overlap

| Arm | Seed | Updates | Unique seen | Total examples presented | Audio hours presented | Dev queries |
|---|---:|---:|---:|---:|---:|---:|
| filtered10_gold | 0 | 600 | 4800 | 4800 | 4.8269 | 2 |
| filtered10_gold | 1 | 600 | 4800 | 4800 | 4.8308 | 2 |
| full_gold | 0 | 600 | 4800 | 4800 | 4.8399 | 2 |
| full_gold | 1 | 600 | 4800 | 4800 | 4.8723 | 2 |
| random_matched_gold | 0 | 600 | 4800 | 4800 | 4.8523 | 2 |

Primary split overlap counts: `{"minimal_reference": {"dev:test": 0, "train:dev": 0, "train:test": 0}, "prompt_id": {"dev:test": 0, "train:dev": 0, "train:test": 0}, "speaker_id": {"dev:test": 0, "train:dev": 0, "train:test": 0}}`.

Training-set overlaps: `{"filtered10_gold:full_gold": 6731, "filtered10_gold:random_matched_gold": 5058, "full_gold:random_matched_gold": 6731}`.

Missing scheduled runs: `[{"arm_name": "random_matched_gold", "seed": 1}, {"arm_name": "full_gold", "seed": 2}, {"arm_name": "filtered10_gold", "seed": 2}, {"arm_name": "random_matched_gold", "seed": 2}]`.

Incomplete directories: `[]`.

## Limits

- Recorded weight fingerprints and encoder changes are checked; no checkpoint tensors were loaded by this auditor.
- Development evaluations are exploratory, including all logged intermediate queries.
- Provided transcripts are not newly audio-adjudicated; one development speaker per L1 limits population inference.

The accompanying JSON records exact source, evaluator and run-artifact hashes. A valid packet is an integrity result, not evidence that the hypothesis is true or the work is publishable.
