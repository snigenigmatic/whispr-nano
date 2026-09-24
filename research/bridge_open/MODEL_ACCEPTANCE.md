# Inference acceptance before comparison

Artifact validation and inference acceptance are distinct. The former establishes saved bytes, coverage and arithmetic; it must not be described as validation of model quality.

Poor performance alone is not a reason to exclude a correctly executed model or tune on its test references. Acceptance checks execution and interpretability, not whether a score is favorable.

Before expanding a model to the full corpus, save these checks on a frozen, small diagnostic cohort:

1. Source integrity: pinned revision, audio and transcript hashes, correct join, actual decoded sample rate, duration, channels, finite values, amplitude and non-silent signal. Decode independently on examples. Resample waveforms only when the model requires a different rate; record input/output rates, sample counts and the resampling method. Relabeling a sampling-rate argument is never resampling.
2. Model-specific input contract: official processor and checkpoint revisions, required feature dimensions, padding, attention masks, normalization and chunking. Compare frontend features to the documented native implementation when available. Do not reuse Whisper settings for other model families.
3. Model state and decoding: evaluation/inference mode, device, precision, explicit language/task, prompts, batch membership, seed, beam/sampling policy, fallback policy and length limits. Compare single-example, batched and official inference paths. Save raw tokens or equivalent termination evidence.
4. Output review: inspect low, median and high error cases, requested versus generated script, repetitions, empty output, clipped endings and decoder-limit cases. Keep failures in denominators; do not silently remove difficult clips.
5. Scoring: freeze language-appropriate normalization before the next benchmark. Preserve original references and hypotheses. Report micro WER with error counts, CER, sample counts and decoding failures. Secondary normalization sensitivity analyses must remain separate from the primary result.
6. Acceptance receipt: distinguish passed checks from untested assumptions and explain any discrepancy before full dispatch. Diagnostic examples taken from an already inspected test set cannot establish a held-out improvement.

Current Whisper FLEURS comparison: the corrected duration-routed v3 execution passed source, native parity and arithmetic checks. See [the disposition](FLEURS_RESULT_HOLD.md) for the exact scope and remaining limitations. Historical predictions remain preserved. Other model families require their own small-cohort acceptance checks.
