> **Historical diagnostic, followed by completed correction:** All 14 long recordings per model have since been rerun with native full-waveform transcription, and Turbo parity checks completed. See the [corrected comparison](artifacts/fleurs_whisper_corrected_v3/RESULTS.md) and [current disposition](FLEURS_RESULT_HOLD.md). The findings and original measurements below remain preserved.

# Independent FLEURS audit

No sample-rate, waveform-decoding, checkpoint or evaluation-state mismatch was found in the checks performed. Short-clip Assamese failures reproduced outside the study runner. A separate long-clip comparison exposed serious sensitivity to our fixed decoding recipe. The general full-study model comparison remains on hold.

## Input and model checks

- All 2,322 source WAVs and reference joins passed exact-byte/hash checks against the pinned source archives and TSVs. The files are already mono 16 kHz. SoundFile and SciPy agree on every waveform; nine FFmpeg examples agree exactly. These checks establish source fidelity, not human verification of every transcript.
- Native OpenAI Whisper and Transformers Large-v3 weights matched: 1,259 corresponding tensors, exact equality of learned parameters, complete HF model-state coverage, and tied output projection. Both models were in evaluation mode using float32 on an A100 80 GB.
- All 27 native/HF 128-bin feature comparisons passed; maximum absolute log-mel difference was 2.3842e-7. Assamese received the correct language token.
- All 126 diagnostic predictions were recovered with verified transport/source hashes and independently rescored. No original full-test prediction changed.

Evidence: [input audit](artifacts/fleurs_input_audit/CONCLUSIONS.md), [model/frontend checks](artifacts/fleurs_native_parity_audit/model_and_frontend_checks.json), [independent analysis](artifacts/fleurs_native_parity_audit/independent_analysis.json).

## Same short clips, independent inference

Eight original short-test batch members per language were selected deterministically. This post-result diagnostic is not a representative full-test estimate or a held-out improvement.

| Inference path | Hindi WER | Bengali WER | Assamese WER |
|---|---:|---:|---:|
| Original HF batch | 25.48% | 67.76% | 105.30% |
| HF single example | 25.48% | 67.76% | 105.30% |
| HF pipeline | 25.48% | 67.76% | 105.30% |
| Native decode, matching declared beam/language settings | 25.00% | 67.11% | 105.30% |
| Native default transcription policy, explicit language and float32 | 26.44% | 66.45% | 106.62% |

These are micro WERs on identical eight-clip sets. The original batch rerun reproduced all 24 saved texts exactly; HF single and pipeline paths also matched all 24. Native matched-setting decode matched 21/24 texts and 22/24 WERs. All eight Assamese texts matched exactly and were in Devanagari. Native default transcription still produced the wrong script on all nine Assamese diagnostic recordings, including the long example: eight predominantly Gurmukhi and one Devanagari.

The evidence supports an observed failure of this checkpoint/decoding setup on these files. It does not establish a limitation of every ASR model or every Whisper recipe. Native/HF search implementations are not identical. Turbo did not receive this independent native decoder comparison.

## Concrete long-clip problem

The 32.82-second Assamese recording `fleurs-as_in-test-10189020253469894562` exposes a weakness in the fixed 30-second split. Its separate 2.82-second tail reached the 440-new-token limit and repeatedly emitted “Satsang with Moojibaba.” Full-clip WER was **513.16%**, with 157 insertions. Native transcription on the complete waveform had WER **113.16%**, with five insertions.

The native output was still in the wrong script. This is not a usable transcription or an accuracy breakthrough. Native transcription changes segmentation, context and fallback policy together, so this comparison does not isolate one causal setting. It establishes that the fixed-split score is not equivalent to the native policy.

The other long examples did not improve in WER: Hindi 52.11% → 53.52%; Bengali 95.00% → 95.00%. No dataset-wide improvement follows from these three cases. All 14 long full-test recordings need a separately declared and validated long-form protocol before a replacement full-study table is accepted.

## Scoring

All 4,644 original WERs recomputed exactly. WER can exceed 100% because insertions add errors beyond the reference word count. Wrong-script output and repetition explain substantial observed errors.

OpenAI's BasicTextNormalizer removes combining marks, including meaningful Indic vowel signs: `किताबें` becomes `क त ब `; `বাংলা ভালো` becomes `ব ল ভ ল `. Lower WER after this transformation is not evidence of better recognition. Our scorer preserves those marks. The secondary implementation matched the pinned official source on all 2,322 references and 4,644 hypotheses. See the [scoring audit](artifacts/fleurs_scoring_audit/REPORT.md) and [official source](https://github.com/openai/whisper/blob/86098128c0b4f24f0e2aa2994de830614b474227/whisper/normalizers/basic.py).

## Disposition

Preserve original results as a fixed-recipe diagnostic. Keep general two-model full-study claims on hold: remaining concerns include a declared reference decoding policy, long-form handling, Turbo's native parity, and human linguistic spot-checking. Do not silently substitute selected reruns or favorable normalizers.

Further full-study dispatch must pass the [model acceptance checks](MODEL_ACCEPTANCE.md). A replacement benchmark needs a separately frozen, documented long-form policy. Poor performance itself is not a reason to exclude a correctly executed model; this diagnostic must not be presented as held-out improvement.

The [GPU job](https://huggingface.co/jobs/snigenigmatic/6ab2bbf751992417dfcd3d63) completed successfully. Script runtime: 822 seconds; ORX runtime including setup: 14m29s. The $1.50 reservation brings conservative HF holds to $34.50/$50; holds are not measured provider charges. No additional full-test GPU jobs were launched during this audit.
