# FLEURS comparison: corrected execution accepted within stated scope

The original fixed-split v2 results remain preserved as diagnostic evidence.
Use [the corrected v3 comparison](artifacts/fleurs_whisper_corrected_v3/RESULTS.md)
for the currently validated Whisper system results.

All 2,322 input files passed source-fidelity and audio checks. Independent
Large-v3 and Turbo checks verified model state, learned weights and frontend
features. Turbo batch/single/pipeline output reproduced all 24 short diagnostics;
the native route matched all 24 WER values. Large-v3's independent diagnostic is
in [the diligence report](FLEURS_DILIGENCE_REPORT.md).

Every recording longer than 30 seconds (14 per model) was rerun using pinned
native full-waveform transcription. The same duration rule applies to both
models and all languages. The other 2,308 recordings retain the audited HF
short-form recipe. The pooled table therefore describes a duration-routed
system, with separate short/long results; it is not a homogeneous decoder.
This post-result correction is not a held-out method improvement.

Severe Assamese errors reproduce in independent inference. They remain in the
results. Removing Indic vowel marks is not accepted as an accuracy improvement.
Known short-form caps remain in denominators; native Whisper EOS/cap status
is not observable in the saved API output. Human semantic transcript review
has not been completed.

Current disposition: source integrity, declared execution and arithmetic
accepted for this descriptive FLEURS benchmark. Population fairness, speaker
subgroup generalization, BRIDGE-equivalent conclusions, and publication
readiness are not established. Every additional model needs its own acceptance
checks before full dispatch.
