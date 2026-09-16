# Frozen WER normalizer

`frozen_english_normalizer.py` is an unmodified copy of Hugging Face Transformers 4.57.6 `models/whisper/english_normalizer.py`, SHA-256 `1a6a81b72bc69eccf61d7a24b16015bb6c31249254725abe092747e348cfe758`. Copyright and Apache-2.0 license notice are retained in the file. Source: https://github.com/huggingface/transformers/blob/v4.57.6/src/transformers/models/whisper/english_normalizer.py

The same source and `paper/data/whisper_normalizer.json` mapping are used by selection, GPU evaluation, and the independent result auditor. This prevents an installed Transformers version difference from silently changing the reported metric. Runtime regex/Python/Unicode versions remain part of reproducibility context.
