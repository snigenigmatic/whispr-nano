# Independent inference diagnostic

This post-result diagnostic checks suspicious FLEURS scores. It is not a held-out performance improvement or a new benchmark result.

The frozen cohort contains the first eight original short-test batch members in each of Hindi, Bengali and Assamese, and the first over-30-second example in each language. All 27 audio and reference files retain their original hashes. References are scored only after inference. Selection is deterministic, but not independent of the previously inspected test results.

Before interpreting any decoder result, verify source bytes, actual 16 kHz mono waveforms, model evaluation state, float32 precision, all corresponding native/HF model tensors, and independently computed 128-bin log-mel features. The prior CPU audit independently checked every source waveform and reference join; these checks add native implementation parity.

Compare the original HF batch, single-example HF direct generation, and HF pipeline with the same declared settings. Compare native decoding with matching declared beam/temperature/language knobs, while recognizing implementation and token-suppression differences. Finally run the official native transcription defaults, including its fallback and long-form behavior. Save actual tokens, segments and settings. The common short cohort is 24 examples; report the three long examples separately. Never compare aggregate arms with different cohort membership as an improvement.

The original full-test predictions remain immutable and on hold. A successful diagnostic does not authorize silent replacement, selection of a favorable test recipe, or an unsupported publication claim. Any further benchmark protocol must declare decoding and normalization explicitly.

Compute: one ORX Hugging Face A100 80 GB job, provider timeout 30 minutes, internal runtime guard 1,450 seconds, $1.50 reservation against the existing $50 HF ceiling. Reservations are conservative limits, not measured provider charges. No full-test rerun is authorized by this diagnostic itself.
