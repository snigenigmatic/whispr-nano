# Full open-model conversational ASR study

Prepared for C. Kaustubh, 22 September 2026. Status: implementation and data access plan, with existing predictions reused for local diagnostics. The full study is not complete.

## Study decision

Build an independent, reproducible counterpart on the six selected open-model recipes. Match Humyn exactly only if its authoritative benchmark assets become available. A broader analysis of the current public sample is achievable; replacing unavailable metadata or metrics with guesses would not be a replication.

Humyn describes 23 commercial systems, 21 languages, six headline metrics and seven cohort dimensions. The listed dimensions are Region, Gender, Age, Density, Duration, Overlap and Gap. Its linked methodology pages returned 404 in the current access check, and no authoritative file-to-benchmark/cohort mapping was verified. [Official study](https://humynlabs.ai/bridge/ASR/2.0); [current access audit](../publication/bridge_full_study_access_20260922.md).

The public [Indic release](https://huggingface.co/datasets/humyn-labs/Indic-High-Fidelity-MultiSpeaker-ASR) supplies 37 files, of which our frozen structural checks admitted 30. Public [LATAM](https://huggingface.co/datasets/humyn-labs/LATAM-High-Fidelity-ASR) and [Asian-language](https://huggingface.co/datasets/humyn-labs/Asian-High-Fidelity-ASR-Dataset) cards offer possible external extensions; none is verified as the complete scored BRIDGE corpus. Keep the Indian-language study primary. Adding unrelated languages is not necessary to resolve its current weaknesses.

## Analysis matrix

| Study component | Current state | Work needed for the full open study |
|---|---|---|
| Per-language WER and CER | Implemented, validated on 83 model–conversation pairs | Complete supported cells under frozen production-reasonable decoder recipes; obtain more independent conversations |
| WIL and error decomposition | Implemented locally from existing aligned counts | Report pooled and mean-conversation definitions explicitly; retain capped outputs |
| Script failures | Unicode diagnostic implemented | Bilingual review of a fixed sample; distinguish spelling/script variants from wrong words |
| Semantic similarity | Not implemented | Choose and pin a multilingual encoder with verified language coverage, chunk/alignment and pooling policy; test controlled omissions, numbers and negation before interpreting it |
| Code-switch precision/recall/F1 | Reference annotations missing | Label English spans, borrowing policy, exact-token versus transliteration equivalence and disagreements; freeze a scorer before scoring |
| Phoneme-informed or transliteration-aware errors | Definitions/runtime not established | Obtain Humyn implementation or publish an independently specified language-aware scorer; validate lexicon coverage and conversion errors |
| Region, gender, age cohorts | Row metadata unavailable | Obtain consented source metadata; do not infer these labels from voices or names |
| Duration, density, gaps, overlap | Audio and segment annotations exist; authoritative cohort labels/bins unavailable | Define continuous open measurements and inspect annotation reliability; reference timestamp gaps are not automatically acoustic silence, and segment overlap is not automatically overlapping speakers |
| Shared-file difficulty and model correlation | CPU analysis implemented | Use exact common cohorts; correlations remain descriptive and confounded by language; do not attribute causality to recording quality |
| Routing benefit | Reference-oracle bounds implemented | Learn/select routing on separate development data and evaluate on held-out conversations before claiming a deployable benefit |
| Cost, latency, memory | Inference telemetry and estimates retained | Match hardware, precision, batching and timing scope; verify provider bills; measure phone performance separately if claimed |
| HumynScore / direct closed-model comparison | Unavailable | Require official formula, weights, missing-data rules, exact corpus/reference mapping, normalizer and predictions/configurations |

All supplementary measures must have distinct open-protocol names until matching is established. Do not present Unicode-script overlap as code-switch F1, embedding similarity as transcript completeness, or a reference oracle as a trained router.

## Execution order and acceptance gates

1. **Correct the comparison conditions.** Preserve the current 30-second results as a bounded-decoder condition. On already exposed development files, compare a native Whisper decoding route and a shorter audio-only window against it. Freeze decoding, language hints, stitching, precision and failure handling before further evaluation. A long-form implementation must not use reference boundaries. Include a language-hint policy difference when comparing language-conditioned AR models with unconditioned CTC.
2. **Audit references and script errors.** Prepare a fixed language-stratified listening sample, reviewed by speakers of the relevant languages. Record audible words, reference mistakes, script equivalents and uncertainty separately. Any corrected reference set becomes a new revision and all affected models are rescored; never repair references selectively to favor a model.
3. **Complete the supported matrix.** Small, Turbo, Large-v3 and Omni form the broad-language comparison, subject to each model's verified support. Qwen 0.6B/1.7B belong in the matched Hindi track on this Indic corpus. Unsupported languages are coverage limitations, not zero-error cells. Large-v3 currently has only Hindi predictions; its other supported cells remain undone.
4. **Build the annotation-dependent metrics.** Obtain official definitions or create openly documented independent ones. Review annotation agreement and scorer fixtures before reporting code-switch/phoneme/semantic results. Demographic cohorts remain absent unless real metadata arrives.
5. **Freeze and confirm.** The current recordings have been inspected, so future tuning on them makes them development data. Use a separately frozen confirmation corpus and disclose prior exposure. SPIRE remains a separate Indian-English track with its own split and access terms; adding it does not restore held-out status to the Humyn sample.
6. **Release the evidence.** Ship permitted manifests and references with attribution, prediction files, exact model revisions, scorer and inference code, failure records, per-language/common-cohort tables, resource receipts and limitations. Release only material permitted by each dataset/model license.

Primary claim candidates are conditional on evidence: how decoding limits, output scripts and language support affect open-model conversational ASR comparisons. Model count alone is not a contribution. With one to three recordings per language, the existing sample cannot sustain demographic fairness or population-superiority claims.

## Budget

No new cloud calls were made for the local diagnostic extension. The existing ledger retains USD15 in reservations; completed-run estimates are smaller but are not provider-verified charges. Before launching any new GPU job, reconcile the actual bill, failed-run charges and overhead against the original USD15 ceiling, then reserve bounded jobs from verified remaining room. This is a billing task, not an automatic request for more money. If no room remains, keep GPU work queued and continue local annotation/scoring work.

Estimate the remaining inference workload from the measured seconds per audio hour for each exact recipe, including startup and headroom. Do not extrapolate Qwen timing to Large-v3 or assume an H100 eliminates annotation and access requirements. No full-study price or completion date is claimed without the remaining corpus size and validated recipe throughput.

## Exact request for Humyn

The following is a draft only; it has not been sent.

> Hello Saksham and team,
>
> I’m C. Kaustubh from PES University. We are building an independent open-model extension of BRIDGE ASR 2.0 using your public conversational samples and will attribute the data and benchmark appropriately.
>
> Could you share the versioned evaluation manifest (or a mapping of public files to the scored set), exact reference revision, three-step cleanup and scoring scripts, and per-file labels/definitions for Region, Gender, Age, Density, Duration, Overlap and Gap? We also need the definitions or implementations for SemSim, CS precision/recall/F1, PIER, lwWER and HumynScore, together with reuse terms.
>
> The linked methodology pages currently return 404 for us. A public repository or an academic-access package would let us test the open models under the same protocol. Until then, we will describe our results as an independent public-sample study and avoid direct leaderboard comparisons.
>
> Thank you,
> C. Kaustubh
