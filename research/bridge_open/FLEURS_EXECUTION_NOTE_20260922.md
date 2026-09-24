# FLEURS execution record

The scientific protocol remains v2. Jobs use the same pinned model, original
waveforms, 30-second routing, float32 beam-5 decoding, scoring, and batch size.
Only the distribution of already fixed IDs across provider jobs changes.

At 16:40 UTC on 2026-09-22, Large-v3 Bengali batches were taking approximately
40 seconds per eight recordings. To leave room inside the existing 30-minute
provider limit, the remaining Bengali range 608:911 is divided at 760 and the
Assamese short-recording cohort is divided at 192, 384, 576, and 768. All these
boundaries are multiples of eight, preserving batch membership and order. This
decision uses elapsed runtime, not transcription scores.

If a job reaches its wall-time guard, preserve the complete original attempt,
including runtime-failure records. A continuation may infer only recordings
that did not produce a scored prediction. Preserve their original batch
membership, model revision, decoder settings and source hashes. Never retry
empty, token-capped or inaccurate scored predictions to select a better answer.
Report first-attempt runtime failures separately from final cohort coverage,
with an explicit provenance map for each continuation. If an original complete
batch cannot be reconstructed, treat the resumed recipe as a disclosed new
condition instead of silently merging it into the original comparison.

All launches retain the $1.50 reservation and 30-minute provider limit. The
shared Hugging Face ceiling remains $50; reservations are not measured bills.
No Modal compute is launched by this FLEURS batch.
