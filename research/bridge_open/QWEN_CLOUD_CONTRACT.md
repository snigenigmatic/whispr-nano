# Qwen comparison, frozen before inference

Same pinned public Hindi cohort, mono16k audio, fixed30s nonoverlapping audio-clock windows and lexical WER/CER scorer as Whisper. No transcript context. Official Qwen-ASR Transformers backend0.0.6, model7278e1e70fe206f11671096ffdd38061171dd6e5. H100, BF16, SDPA, batch8, greedy max512 continuation tokens. Pinned model generation config EOS151643 or151645. No forced alignment. Hindi language hint and empty system context. Preserve continuation IDs and raw EOS evidence; primary text bypasses repetition cleanup; retain official parsed output as a secondary sensitivity result. References opened only after the conversation finishes.

This compares model-specific recipes; it does not isolate model architecture from numerical precision. Three public development conversations cannot establish population superiority or publication novelty. Freeze was chosen before viewing Qwen output.

Source: https://github.com/QwenLM/Qwen3-ASR/blob/main/qwen_asr/inference/qwen3_asr.py
Model: https://huggingface.co/Qwen/Qwen3-ASR-1.7B
PyPI qwen-asr0.0.6 wheel SHA256:b9c55a38413298f3a990a4475467399daec6e8f4172363053fc42e2166c2dfd3.

USD1.50 retained reservation; shared USD15 ceiling. Worker timeouts and retry limits unchanged.
