# Open conversational-ASR extension: draft benchmark protocol

**Status:** pre-inference protocol. This specifies an independent open-model
evaluation inspired by BRIDGE ASR 2.0. It is not a replication, a re-score of
BRIDGE, or evidence about the published leaderboard.

## Scope and claim boundary

Humyn's BRIDGE page says it evaluates spontaneous dual-speaker conversations
with a common three-step cleanup, seven cohort dimensions, and a six-metric
stack. It names WER, CER, SemSim, CS F1, PIER, and WIL, and displays lwWER and
Humyn Score. Its linked detailed metrics reference and Humyn Score methodology
both returned 404 when checked on 2026-09-22. The public dataset page did not
expose an IndicBench manifest, scripts, or benchmark configuration.

The public `humyn-labs/Indic-High-Fidelity-MultiSpeaker-ASR` card exposes 37
records with `language`, `file_name`, `audio`, and `transcript_json`. It is an
acquisition lead, **not confirmed BRIDGE 2.0 scored data**. The card's CC BY
4.0 terms apply only to that public repository; file-local `Speaker 1` labels
are not stable person identities.

Every resulting table must be called an **open independent extension**. Never
concatenate, subtract, visually rank, or claim the same/different score with
BRIDGE until Humyn supplies an authoritative mapping of file IDs, reference
revision, normalizer, scorer, and model configuration and matched predictions
are rerun. Existing BRIDGE numerics are source-attributed background only.

Sources: [BRIDGE ASR 2.0](https://humynlabs.ai/bridge/ASR/2.0), [public
multi-speaker card](https://huggingface.co/datasets/humyn-labs/Indic-High-Fidelity-MultiSpeaker-ASR),
and `research/publication/bridge_access_audit_20260917.md`.

## Corpus and revisions

Before inference, create and freeze `manifests/corpus-v1.jsonl`. Every row
requires `corpus_id`, repository URL, immutable source revision or content
hash, split, source file name, audio SHA-256, reference SHA-256, released and
normalized language IDs, and license. A source revision, hash, inclusion rule,
or language-map change creates a new corpus revision; it cannot replace
`corpus-v1` silently.

The current public inventory is 37 conversations in 13 Indic labels: Assamese
(2), Bengali (3), Bhojpuri (3), Chhattisgarhi (3), Gujarati (3), Haryanvi (3),
Hindi (3), Kannada (3), Malayalam (2), Marathi (3), Punjabi (3), Tamil (3),
and Telugu (3). These are metadata counts, not claims that all pairs pass QC.
No Spanish, Portuguese, Vietnamese, BRIDGE cohort labels, or BRIDGE membership
is confirmed for this source.

Include only audio that decodes, references that parse, segments with finite
nonnegative `start`/`end` and `end >= start`, and frozen mapped language IDs.
Record exclusions before outputs are inspected. A reference-audio audit failure
gets `reference_suspect`, is withheld from primary lexical scoring pending a
new revision, and remains in the failure appendix.

`supported` means the frozen model route explicitly documents support for the
manifest language; `attempted` means it ran; `scored` needs a successful output
and QC-eligible reference. Report all three plus duration per language/model.
A language belongs to a comparison's **common cohort** only when every compared
model scores every included conversation. Write those exact IDs to
`manifests/common-<comparison-id>-v1.jsonl` before aggregates.

## Fixed inference conditions

Freeze before first non-smoke output the model repository and immutable
revision/weight digest; runtime and dependency-lock digest; device, precision,
decoding, language hint, prompt, and seed policy; and audio/channel policy. No
reference text/timestamps or manually chosen speech boundaries may enter normal
inference.

Allowed prespecified conditions are `whole_audio` and `audio_only_segmented`.
For chunks preserve each original full-audio time range, sequence number,
overlap/pad, segmentation algorithm and immutable settings, and deterministic
stitching rule. Reference-timestamp segmentation is prohibited from primary
results. If diagnostic use is ever justified, call it
`oracle_reference_boundaries`, report it separately, and never select a model
with it.

## Prediction and failure records

Write append-only UTF-8 JSONL: one `conversation_prediction` for every
manifest conversation/model/configuration, including failures. Required fields:

```json
{
  "record_type": "conversation_prediction", "schema_version": "1.0",
  "run_id": "...", "corpus_revision": "corpus-v1", "model_id": "...",
  "model_revision": "...", "condition_id": "whole_audio",
  "conversation_id": "...", "language": "...",
  "status": "success|unsupported|decode_error|inference_error|timeout|empty_output|invalid_output",
  "attempt": 1, "audio_sha256": "...", "started_at": "RFC-3339",
  "ended_at": "RFC-3339", "text_raw": "...", "text_for_scoring": "...",
  "chunks": [], "failure": null
}
```

Segmented output has ordered original-time chunk intervals, raw chunk text, and
stitching provenance. Failures contain a stable code, non-secret diagnostic,
and retry policy. Retries create separate attempts; select the first success by
the predeclared rule. Unsupported, empty, and failed output remains in coverage
and failure reporting; never turn it into zero WER or silently drop it. Report
conditional-on-success lexical scores with completion/coverage rate.

## Text, code switching, metrics, and uncertainty

Publish a versioned normalizer and fixtures before scoring. It must declare
Unicode, whitespace/punctuation, case, language tokenization, digits/numbers,
and transliteration handling. Preserve raw reference/model text; derive scored
text only through it. Any normalizer change creates a new score revision.

Score under the released reference language. Do not translate, use model-detected
language for cohorting, or substitute Latin transliterations. Code-switched
spans require separately frozen, auditable reference annotations. Until those
and a scorer exist, CS F1 and PIER are blocked, never guessed from English-token
heuristics. `metrics_contract.json` is authoritative: WER, CER, and standard
WIL may be reported as independently defined open metrics after freezing the
normalizer, tokenizer, and pinned scorer release; label them `open_protocol_v1`.
SemSim is preparable as an independent metric once its encoder, immutable
revision, input policy, similarity function, and aggregation are designed and
frozen. None of those four metrics is BRIDGE-matched from public material. CS
F1, PIER, lwWER, and HumynScore remain unavailable for a comparable BRIDGE
claim. Do not reverse engineer unavailable Humyn formulas.

Primary results are per-language, model, and condition, with conversation count,
reference token/character totals, duration, status counts, WER, and CER. Do not
publish an all-language leaderboard or rank aggregates with unequal language or
successful-file sets. A macro mean averages per-language conversation means on
the named common cohort. A micro measure pools numerator/denominator only over
the same named cohort and is supplemental.

For each prespecified pair, bootstrap **complete conversation clusters**,
stratified by language, then recompute the stated macro/micro estimand. Report
two-sided percentile 95% interval, replicate count, seed, and cluster count.
Never treat chunks, segments, words, or file-local speaker labels as independent
units. With two or three conversations per language, intervals are descriptive,
not fairness or population evidence.

## Required deliverables and unblock conditions

- [ ] immutable corpus/source/model/normalizer manifests and hashes
- [ ] raw prediction, chunk-map, stitched prediction, and complete failure records
- [ ] pinned scorer version and fixtures for every reported metric; pin JiWER's
      `process_words` output if independently reporting standard WIL
- [ ] per-language coverage plus named common-cohort manifest per comparison
- [ ] per-conversation rows, aggregate tables, bootstrap config and intervals
- [ ] BRIDGE numerics physically separate and source-attributed as background

BRIDGE matching remains blocked until an authoritative versioned package has
scored IDs/audio/reference hashes, reference revision, cleanup and metric code,
code-switch/cohort annotations, endpoint settings/versions, per-file
predictions or reproduction authorization, and Humyn Score formula/weights.
Preserve this open protocol and add a new matched revision when supplied.
