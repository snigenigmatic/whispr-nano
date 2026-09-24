# Finite core study — 24 September 2026

Owner: C. Kaustubh. Authorized by “continue on all of these” after the scope review. This document replaces open-ended model discovery as the execution scope; earlier accepted evidence remains frozen.

## Question and contribution test

Which of six previously selected open ASR systems should a practitioner choose for the specified Indian-speech tracks, given corpus-level error, severe per-recording failures, and measured GPU inference resources? Do average word errors conceal failures that change that choice?

The intended output is a reproducible systems audit and bounded error analysis. A new model, new dataset, universal model ranking, first Indian-ASR benchmark, or new fairness method is not the proposed contribution. The literature assessment must establish a narrower useful distinction; coverage alone does not establish novelty. If prior work already answers the same practical question, report this as a replication/system audit and do not manufacture a method claim.

## Fixed coverage

Six recipes: Whisper Small, Turbo, Large-v3; Qwen3-ASR 0.6B and 1.7B; OmniASR CTC 1B. Four tracks: FLEURS Hindi 418, Bengali 920, Assamese 984; separately, recovered Svarah English 518. Qwen Bengali/Assamese remain outside documented support in this plan. There are 20 supported quality cells: 12 previously accepted and 8 to complete. `completion_matrix.json` is the live status authority, not the discovery roster.

Retain every cohort row, failure, empty output and existing clipping flag. FLEURS uses the pinned complete selected test cohorts. Svarah is an availability-selected 518/6656 shard, with prior exposure and no verified speaker IDs. Its 12-item acceptance panel overlaps the 518, and adds no independent recordings. SPIRE held-out data remains unopened.

## Quality conditions and acceptance

Reuse pinned accepted preprocessing, full-waveform routing, scorer and decoding recipes. Whisper: float32 beam 5 HF short-form at <=30 seconds, accepted official native full-waveform route above 30 seconds. Qwen: native BF16 greedy batch 8, explicit supported language. Omni: native BF16 greedy CTC batch 1. These are declared system recipes, not a controlled causal comparison of architectures. Changing batch or precision defines a separate condition.

Accept results only after original source byte/hash and sample-rate/duration checks; exact unique cohort coverage; pinned model/revision and actual model-asset hashes; committed code/config transport; retained raw prediction/termination evidence; and independent rescoring. Empty or failed outputs remain in coverage accounting, and failed inference is never silently dropped. Technical integration failures cannot support a model-quality conclusion.

Primary tables report per-track micro WER/CER, mean utterance WIL, reference denominators and failure coverage. WER>=100% is an edit-count tail marker, not a human semantic-error label. Do not average unrelated tracks into a winner. Source sentence IDs are not speaker IDs. Optional utterance bootstrap intervals are descriptive and cannot imply independent speaker-level inference.

## Controlled resource measurement

Use the existing exact 12-item Svarah acceptance panel for every model, one full warmup and three full timed passes, batch one, same A100 80GB class. Preload original arrays before timing. Include features, transfer, decoding and text postprocessing; exclude model load, file I/O, scoring and secondary validation calls. Record GPU/runtime, complete outputs, failures, residency, allocated/reserved peaks, per-clip latency and aggregate audio-seconds/wall-seconds. See RESOURCE_PROTOCOL.md.

Score each pass separately on 12 unique references; do not count repetitions as more data. This small, pre-exposed English panel provides bounded hardware measurements, not deployment throughput guarantees or phone feasibility. Qwen primary batch-eight quality is distinct from its batch-one profile; any combined quality/resource plot must use matched panel quality, or explicitly qualify parity evidence.

## Bounded reference and error review

Retain the frozen 50-clip FLEURS packet and anonymous model cards. Provide reviewer instructions, a validated import path, and a separate bounded English extension once missing predictions exist. A language-competent reviewer must listen to the source and assess reference fidelity before model meaning. Record unjudgeable audio and uncertainty; do not force a label. Human judgments remain pending until actual completed responses are received. Do not manufacture ratings from WER or Unicode scripts.

## Paper and completion boundary

Deliver: (1) full supported coverage table with accepted receipts, (2) matched resource-panel table, (3) bounded error examples with review status, (4) literature-grounded manuscript draft and claim-to-evidence map, (5) offline inspectable predictions/audio review artifact.

No further model families, new datasets, training, script masks, pseudo-label methods, or extra decoder searches are part of this completion. Exact Humyn/BRIDGE replication, demographic fairness, semantic/code-switch annotation metrics and mobile latency remain outside the supported claims. Human review and a defensible contribution remain manuscript gates even if all GPU jobs finish. Submission and acceptance are separate from a completed experiment.
