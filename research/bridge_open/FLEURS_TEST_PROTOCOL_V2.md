# FLEURS official-test comparison v2: complete audio coverage

This addendum supersedes only the input-length handling in FLEURS_TEST_PROTOCOL_V1.md. It is frozen after source acquisition/QC, before any official-test model inference. Version 1 and the acquisition receipt remain unchanged.

The official test TSVs contain 2,322 recordings (Hindi 418, Bengali 920, Assamese 984), totaling 29,684.16 seconds. Fourteen recordings have durations of 30.60–34.56 seconds (one Hindi, nine Bengali, four Assamese). Their source archive hashes, sample-count metadata and waveform checks pass. They are input-length routing cases, not evidence of incorrect references. Excluding them or truncating at 30 seconds would break the declared full-test scope.

Prespecified routing, identical for every Whisper model:

1. Recordings of at most 30 seconds use complete-utterance same-language batches of up to eight.
2. The 14 longer recordings use the existing verified fixed 30-second non-overlapping audio-clock decoder, covering the final remainder and joining chunk texts chronologically. No reference timestamps, text, VAD oracle, silence removal, overlap deduplication or outcome-based change is allowed.
3. Report the two routing conditions separately and their exhaustive full-test combination. The primary comparison uses all 2,322 recordings for each model. Report these preprocessing details; do not claim the resulting score is identical to a publisher's FLEURS leaderboard.

Manifest version 2 preserves every ID, exact original audio/reference byte hash, source member and source archive hash. It reclassifies only the 14 sole `over_30_seconds` warnings as `reference_qc=pass`, retains the warnings, and adds explicit `inference_route` metadata. All other QC flags still block primary inference. No test prediction has been seen when making this decision.

All other v1 model revisions, decoding, scoring, uncertainty and budget rules remain unchanged. Development scalar/batched differences must be documented. Results must not turn script changes into equivalences through transliteration after inspection.
