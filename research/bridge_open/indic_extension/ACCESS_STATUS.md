# IndicConformer access status

Checked 2026-09-23 using the existing `HF_TOKEN` from `~/.openresearch/env` without printing it. The Hugging Face model API returned HTTP 200 and identified revision `e9b71b369c048e2c6b634d4c131061c34e441179`; the model metadata reports `gated: auto`. Resolving the model's `config.json` returned HTTP 403.

No terms were accepted, account permissions changed, model files downloaded, or further provider access attempts made. The runner and configs below are therefore contract-ready but unexecuted. A real run requires an approved access grant and the official AI4Bharat ONNX wrapper/runtime; the wrapper entry point is intentionally explicit rather than guessed.
