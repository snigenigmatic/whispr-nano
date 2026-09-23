# IndicConformer acceptance runner

This folder prepares an acceptance run for `ai4bharat/indic-conformer-600m-multilingual` at revision `e9b71b369c048e2c6b634d4c131061c34e441179` over the pinned FLEURS test manifest. The target languages are Assamese (`as`), Bengali (`bn`), and Hindi (`hi`), with 984, 920, and 418 recordings respectively.

`config_canary.json` contains one deterministic recording per language. `config_full.json` contains all 2,322 records. Both require native full-waveform 16 kHz mono input and preserve exact input/reference hashes.

The model is gated and its weight resolve returned HTTP 403 during preparation. No inference was run. Once access is granted, dispatch must provide an adapter implementing the explicit `IndicConformerAdapter` protocol in `run_indic_conformer.py`; the protocol avoids guessing an unpublished wrapper import. The adapter must return one plain text string per input and may expose an `evidence()` mapping for ONNX runtime/provider metadata. ONNX has no PyTorch `.eval()` state; the runner records `eval_state_observable=false` unless the adapter explicitly reports otherwise.
