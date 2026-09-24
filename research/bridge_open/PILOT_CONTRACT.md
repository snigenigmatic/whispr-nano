# Hindi pilot contract — frozen before full-conversation inference

This is a development pilot on the three publicly released Hindi recordings. The first 30 seconds of Hindi_1 were used for runtime debugging; no held-out claims are permitted. Only structural QC has been performed, not a human transcript audit.

The frozen model is Whisper-small, revision 973afd24965f72e36ca33b3055d56a652f456b4d. Inference uses isolated Transformers 4.57.6, torch 2.8.0, float32, MPS, greedy decoding, language hint Hindi, 30-second contiguous nonoverlapping audio windows, maximum 440 new tokens per window, and batch size 1. Boundaries never use reference timestamps. Join decoded windows with spaces; do not delete repetitions or repair script.

Scoring uses JiWER 4.0.0 after NFC, casefold, Unicode punctuation-to-space, and whitespace collapse. Preserve numerals, scripts and combining marks; no transliteration or number-word conversion. CER uses Unicode codepoints with whitespace removed. Report per-conversation WER/CER/WIL, pooled word/character edit ratios and conversation means.

All fully processed recordings remain in the scored cohort, INCLUDING empty or generation-capped text. Generation caps are model-quality failures and are explicitly counted as generation_limit, never clean success. Runtime failures are recorded separately and their missing scores are never replaced by zero. Report both scored count and clean-completion count. This pilot refinement prevents silently improving the average by dropping repetitions. It supersedes the generic draft protocol's conditional-success aggregation for this pilot only.

A smoke test showed repetition and generation-cap termination on the first Hindi window. It reproduced across CPU/GPU and in an isolated pinned runtime; the shared environment's version mismatch was real but was not the cause of that observed repetition. Preserve all smoke records. The full pilot measures this baseline honestly rather than removing its problematic first window.

Cost: local execution only, zero cloud charges. Existing Modal spending cap remains unchanged. Data manifest hash, runtime lock, model/config/tokenizer hashes, code hashes, raw tokens and chunk times are retained. Do not compare these metrics numerically with the BRIDGE leaderboard.

Pre-result repair: the first tracked launch was cancelled before completing a conversation. The default Transformers Whisper wrapper strips EOS from returned tokens, so the runner now requests raw generation output before applying termination flags. Batch size is one for bounded local memory and latency. The abandoned launch is preserved; it contributes no metrics.
