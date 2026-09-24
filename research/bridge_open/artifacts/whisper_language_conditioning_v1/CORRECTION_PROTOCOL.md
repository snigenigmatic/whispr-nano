# Full long-stratum correction and Turbo verification

This child builds on native Large-v3 weight/frontend/decoder parity. All source audio and reference bytes remain frozen. It evaluates all 14 test recordings longer than 30 seconds with the same native OpenAI policy for both Large-v3 and Turbo, and verifies Turbo on the original 24 short diagnostic clips. No sample is selected for replacement based on its error rate.

The corrected comparison has two explicit strata. At most 30 seconds: retain the original HF float32, beam-5, 440-token, language-specified condition after the independent implementation checks. Longer than 30 seconds: pass the complete waveform to openai-whisper 20250625 transcribe, using the committed full options in experiments/config.json. Both routes use the same Indic-preserving scorer. Report strata separately and label any pooled summary as a duration-routed system result, not a homogeneous decoder or universally optimal model score.

Record every native decode attempt, selected segment timestamps, complete waveform sample count, configured thresholds and fallback temperatures. The attempt token-count flag is a diagnostic heuristic, not authoritative EOS evidence: the public native decode result strips EOS. Do not drop empty, repetitive, wrong-script, suspicious timestamp or capped outputs from scores.

This is a post-result protocol correction. Preserve the prior full results and publish the route difference as sensitivity analysis. The diagnostic/full test has already been inspected; no held-out improvement claim is allowed. Numerical backend equality need not imply exact beam output equality, so report paired disagreements. Native long-form choices differ from short-form choices and must remain visible.

One guarded A100 80 GB HF job, 30-minute provider timeout, $1.50 reservation against the existing $50 ceiling. Do not launch further runs from this node after it produces an answer.
