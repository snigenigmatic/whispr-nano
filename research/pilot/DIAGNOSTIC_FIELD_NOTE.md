# Invalid token-limit diagnostic field

Snapshot: 2026-09-09T08:19:17.103794+00:00. Review uses completed local development outputs only; smoke runs and test predictions are excluded. It does not establish a complete nine-run scientific result.

`training.py:397–408` decodes the returned tensor directly and computes WER from that text. Its separate `generation_reached_token_limit` field instead checks whether EOS is absent after slicing off four presumed prefix tokens. That assumption is false for these returned tensors: none of the inspected rows contains the decoder prefix.

The official Transformers 5.16.1 tag resolves to commit `93c8b7b485963a10800c91f55304db6be211c2bd`. Its Whisper implementation removes the decoder prefix, removes terminal EOS, then pads shorter returned text sequences to the longest sequence in the batch. This model uses EOS 50257 as its padding ID. Longest sequences, including ties, therefore contain no EOS padding and receive a false token-limit indication. This is a conclusion from official version-matched source and saved outputs; installed container source bytes were not independently hashed. [Prefix removal](https://github.com/huggingface/transformers/blob/93c8b7b485963a10800c91f55304db6be211c2bd/src/transformers/models/whisper/generation_whisper.py#L1149-L1153), [EOS removal](https://github.com/huggingface/transformers/blob/93c8b7b485963a10800c91f55304db6be211c2bd/src/transformers/models/whisper/generation_whisper.py#L1072-L1095), [final padding](https://github.com/huggingface/transformers/blob/93c8b7b485963a10800c91f55304db6be211c2bd/src/transformers/models/whisper/generation_whisper.py#L218-L223).

| Run | Evaluation | Rows | Flagged | Returned tensor widths | Flags exactly identify longest text in each batch |
|---|---|---:|---:|---:|---|
| filtered10_gold-seed0 | baseline | 974 | 158 | 12–25 | Yes |
| filtered10_gold-seed0 | final | 974 | 154 | 12–20 | Yes |
| filtered10_gold-seed1 | baseline | 974 | 158 | 12–25 | Yes |
| filtered10_gold-seed1 | final | 974 | 158 | 12–18 | Yes |
| full_gold-seed0 | baseline | 974 | 158 | 12–25 | Yes |
| full_gold-seed0 | final | 974 | 152 | 12–20 | Yes |
| full_gold-seed1 | baseline | 974 | 158 | 12–25 | Yes |
| full_gold-seed1 | final | 974 | 158 | 12–20 | Yes |
| random_matched_gold-seed0 | baseline | 974 | 158 | 12–25 | Yes |
| random_matched_gold-seed0 | final | 974 | 148 | 12–19 | Yes |

All saved configurations explicitly request `max_new_tokens=224`. Baseline outputs reach 25 tokens, ruling out a universal 20-token returned-output ceiling. Primary development references contain 3–13 whitespace-separated words; word counts are not tokenizer lengths. Final maximum widths of 18–20 are consistent with short utterances, and no inspected returned sequence approaches 224. These observations provide no evidence of a 20-token cap. The postprocessed outputs do not preserve per-sequence stopping reasons, so the invalid flag cannot establish a truncation rate or certify EOS termination. No new generation or reference-tokenization experiment was run for this note.

The defect is confined to this unused diagnostic: the hypothesis passed to WER is decoded before the flag is created, without the erroneous prefix slice. The independent auditor recomputes counts from every saved hypothesis and does not consult this field; neither does the decision gate. Thus this defect does not change the reported WER or registered contrast. This does not certify the correctness of every aspect of generation. Retain original artifacts, mark the field invalid, and do not report it as a truncation statistic. A future separately versioned diagnostic should capture raw generation output and explicit stopping evidence. Frozen training and evaluation code were not changed.
