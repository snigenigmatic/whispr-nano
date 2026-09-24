# FLEURS official-test comparison v1

Frozen before acquiring test audio/references or running test inference on 2026-09-22. The fixed scope is all released official-test utterances in Hindi (`hi_in`), Bengali (`bn_in`) and Assamese (`as_in`) at Google FLEURS revision `70bb2e84b976b7e960aa89f1c648e09c59f894dd`. These languages were chosen before model results to replace the corresponding problematic source and retain Hindi as an existing comparison language.

## Primary model comparisons

- Whisper large-v3-turbo, revision `41f01f3fe87f28c78e2fbf8b568835947dd65ed9`.
- Whisper large-v3, revision `06f233fe06e710322aca913c1bc4249a0d71fce1`.
- Whisper small, revision `973afd24965f72e36ca33b3055d56a652f456b4d`, as the compact baseline when the bounded budget permits.

Additional model families are separate prespecified follow-ons after runtime verification, not substitutes chosen because they win a test comparison. Qwen's currently verified common-language comparison is Hindi only. English/Indian-English requires its own corpus and common cohort.

## Fixed conditions

Original mono 16 kHz audio, complete utterance, no reference-guided segmentation, no text/audio augmentation, no training. Deterministic beam search with 5 beams, no sampling, no timestamp prediction, max 440 new tokens, seed 12, float32, SDPA, no TF32. Use the existing fixed HF PyTorch 2.8.0 CUDA 12.8 container and Transformers 4.57.6. Same-language batches of up to 8 are allowed once validated on development data; their exact composition/order and final runner code hashes are recorded. Batch grouping must not use references or predictions. Do not change decoding based on test scores. A code bug invalidates the affected recipe and requires a disclosed revision/rerun.

Verify source archive LFS hashes, original clip/reference hashes and official metadata lengths. Keep all test candidates in the manifest. Flag missing/nonfinite/all-zero audio, >80% exact zero samples, wrong sample rate/channels, empty references, metadata sample-count mismatch or duration >30s before inference. Any flags pause primary evaluation for an explicit corpus revision and adjudication. Do not silently omit hard examples. Automated waveform checks are not human transcript adjudication.

## Scoring and comparisons

Use the released normalized `transcription` as the primary reference and preserve `raw_transcription`. Both reference and hypothesis receive the existing versioned scorer: NFC, casefold, Unicode punctuation to spaces, whitespace words, whitespace-free Unicode codepoint CER; no transliteration or number normalization. This is the project's open scoring protocol, not necessarily an author's leaderboard scorer.

Report pooled WER/CER separately for each language on identical complete cohorts, with empty/capped predictions retained in errors and generation/runtime failure rates reported. No aggregate mixing unequal language support. No BRIDGE score or direct ranking against commercial scores from another corpus.

For paired differences, resample source sentence-ID clusters within each language (seed 12; 2,000 replicates), keeping multiple recordings of the same sentence together. These intervals measure variability across sampled sentence clusters; speaker identities are absent from this export, so they are not speaker-cluster or population-fairness intervals. Public training/evaluation contamination remains unknown unless model documentation establishes otherwise.

Runtime reports distinguish batch throughput from single-request latency and include model load separately, GPU model, allocated peak memory, audio duration and batch composition. A100 throughput does not establish smartphone deployability.

## Execution and stopping

Start with the 36-clip development integration check. Freeze the accepted runner code and composition before test dispatch. Divide full test evaluation into bounded jobs with exact disjoint ID lists, preserving every official-test candidate once per model. Each HF job retains the existing 30-minute provider timeout and $1.50 reservation against the user-authorized $50 HF cap; existing reservations remain charged to the cap. Modal retains its separate $15 ceiling. Do not release reservations from runtime estimates. Do not launch follow-on model families before full common-cohort artifacts validate.

A first complete two-model comparison is the initial delivery milestone. It is a reproducible benchmark result, not by itself a claimed methodological novelty or assurance of publication.
