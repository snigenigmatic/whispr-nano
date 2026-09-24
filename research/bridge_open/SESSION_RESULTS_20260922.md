> **Audio QC correction (2026-09-22):** Five public files contain transcript text over exactly silent audio. Affected Assamese/Bengali quality comparisons and aggregates containing them are withdrawn pending adjudication. Original metrics below are preserved as historical records. See the [source-verified audit](artifacts/audio_content_audit/RESULTS.md).

# Current evidence: native decoding and Hugging Face execution

Prepared for C. Kaustubh. These results extend the six-model public Humyn pilot. They are development evidence and do not establish a publishable novelty claim or an exact BRIDGE replication.

## Completed controls

- Fixed 15-second Turbo windows worsened pooled Hindi WER from 52.02% to 56.79% on the same three conversations. This change was not promoted.
- Native long-form Turbo improved Hindi WER from 52.02% to 46.09%. The frozen recipe was then evaluated on the other 19 exposed conversations.
- Across all 22 conversations/nine languages, native decoding improved 17, worsened 4 and tied 1. Median paired change was −1.98 percentage points. Per-language pooled gains vary; Kannada WER regressed by 2.23 points. Native decoding still had 393/753 selected calls without EOS versus 245/648 for fixed windows. Lower WER did not eliminate repetition.
- HF A100 portability control reproduced the H100 Bengali transcript, selected token sequences and all metrics exactly on one 15.05-minute conversation. Complete remote artifacts were recovered and independently validated locally.
- Native compression/temperature fallback on that same A100 case worsened WER 98.28%→99.83% and CER 76.73%→89.51%. Selected calls without EOS fell 25/31→2/31, but inference time increased 95.0s→271.9s. All 176 attempts are retained. The pinned Transformers implementation carries beam 1 into later windows after its first sampling transition; this is part of the tested recipe.

Large-v3 completed: raw WER 97.4972%, CER 77.6333%. This case is affected by the audio/reference mismatch above and is withdrawn as model-quality evidence. All four HF jobs are terminal (three completed, one setup error); no study HF GPU remains running.

## Budgets

Modal remains capped at $15. Sixteen attempts currently hold $11.98370763: twelve have provider charges reconciled ($2.98370763 plus $3.00 retained margin), and four retain their original $1.50 reservations.

HF has a separate user-authorized $50 ceiling. Four bounded A100 dispatches currently hold $6.00, including the failed setup attempt. Each has a 30-minute provider timeout. These reservations are not verified spending or a claim about the dashboard's remaining balance.

## Evidence index

- [All 22 matched native/fixed results](artifacts/native_expansion_comparison/RESULTS.md) and [machine-readable comparison](artifacts/native_expansion_comparison/comparison.json).
- [Hindi decoder controls](artifacts/decoder_controls/RESULTS.md).
- [Matched HF fallback result](artifacts/hf_fallback_comparison/RESULTS.md) and [independent beam-state review](artifacts/hf_fallback_comparison/beam_state_review.md).
- [HF execution and source-transfer repair](HF_EXECUTION_20260922.md). The portable project is `/Users/prateekraj/whispr-bridge-hf`; fixed command and budget guard are in its README.
- [Earlier six-model pilot](EXPANDED_RESULTS_20260922.md). Its original budget paragraph is historical.

The original frozen ORX nodes preserve exact commits and run commands. This evidence snapshot collects reports for review; it does not replace those nodes. Recovered HF artifacts include the ORX commit and source-archive digest. No SPIRE held-out partition was opened. No demographic fairness, smartphone deployment, state-of-the-art performance, or publication acceptance is claimed.
