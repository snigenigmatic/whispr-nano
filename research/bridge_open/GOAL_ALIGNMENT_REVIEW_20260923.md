# Goal alignment review — 23 September 2026

Prepared for C. Kaustubh after the request to examine scope drift. This is an assessment and proposed correction, not authorization for a new experiment. No GPU job was launched for this review. Existing results and frozen experiment nodes remain unchanged.

## What the user asked for

The enduring outcome is a demonstrable, useful ASR research project and a defensible publication submission for the capstone. The original technical motivation was better Indian-English recognition with compact open models under device/resource constraints.

The user subsequently chose an explicit different route: evaluate strong, recent open ASR systems as an independent counterpart to Humyn's commercial-system audit. They also explicitly approved trying other open datasets after problems with the public Humyn files. That benchmark pivot and corpus substitution were authorized; reverting to correction-transfer training, numeric rescoring or on-device training is not an automatic correction of the current drift.

The latest agreed audit direction is represented by [FULL_STUDY_SPEC.md](FULL_STUDY_SPEC.md), [MODEL_ROSTER.md](MODEL_ROSTER.md), [PROTOCOL.md](PROTOCOL.md) and [metrics_contract.json](metrics_contract.json). Their historical budget limits are superseded by the user's later spending authorization. Broad spending authorization did not change the scientific objective.

## What actually happened

| Agreed component | Accepted work now | Gap / interpretation |
|---|---|---|
| Open-model comparison with explicit language support | Four declared systems in the retained FLEURS comparison: Turbo, Large-v3, Qwen 1.7B and Omni CTC 1B; Qwen covers Hindi only in that comparison. Two systems on the recovered Svarah subset. | The 27-repository discovery roster is not a completed evaluation. Even the earlier core shortlist is incomplete on the replacement benchmark. |
| Indian speech robustness, inspired by a conversational audit | Full selected FLEURS test cohorts for Hindi, Bengali and Assamese; 518 Svarah records from one recovered shard. | FLEURS read speech cannot establish conversational behavior. The Svarah subset cannot establish all-corpus or speaker-population performance. Different datasets must stay distinct. |
| Multi-dimensional evaluation | WER/CER/WIL and error counts, source/runtime evidence, diagnostic script counts and failure tails. | No completed semantic or code-switch annotation-based evaluation; no defensible demographic comparisons; no controlled accuracy/resource comparison or measured phone performance. |
| Credible research contribution | Reproducible system differences and a negative language-selection diagnostic. | No newly trained improvement and no established novel contribution. Script observations alone do not settle novelty, language correctness or meaning. |
| Demonstrable artifact and manuscript | Working local viewers, prediction exports, source archives and blinded review packet. | Useful engineering artifacts; not a completed human-reviewed study or submission-ready manuscript. |

The six-recipe comparison in the earlier full-study plan is still incomplete on the replacement cohorts:

| Core recipe from the earlier plan | FLEURS Hindi (418) | FLEURS Bengali (920) | FLEURS Assamese (984) | Recovered Svarah (518) |
|---|---|---|---|---|
| Whisper Small | Comparable full result pending | Pending | Pending | Pending |
| Whisper Turbo | Accepted | Accepted | Accepted | Pending |
| Whisper Large-v3 | Accepted | Accepted | Accepted | Pending |
| Qwen3-ASR 0.6B | Comparable full result pending | Outside documented support in this plan | Outside documented support in this plan | Pending |
| Qwen3-ASR 1.7B | Accepted | Outside documented support in this plan | Outside documented support in this plan | Accepted |
| OmniASR CTC 1B | Accepted | Accepted | Accepted | Accepted |

“Pending” means no accepted complete result in these current comparison tables; it does not erase historical pilot runs on other data. This is an accounting of gaps, not a decision to run every cell. Corpus relevance and the paper question must determine the final scope before dispatch.

The current [status](CURRENT_STATUS.md) and [paper evidence map](PAPER_EVIDENCE_MAP.md) already disclose these limitations. Repeating the disclosures without resolving the scoped gaps became part of the drift.

## Where the assistant lost alignment

1. **A recovery benchmark became the main project.** The FLEURS integration check appropriately replaced invalid source examples, then expanded into the central workstream without closing the broader audit matrix.
2. **Implementation completion became the progress signal.** Passing hashes, native-API parity and receipt checks is necessary to trust measurements. It does not, by itself, answer the scientific question or complete promised coverage.
3. **Anomalies began selecting the agenda.** Whisper failures led to decoder controls; Omni output failures led to a proposed script-repair direction. Bounded diagnosis was warranted. Promoting each anomaly into the next research project was not.
4. **The constraints stopped shaping a single deliverable.** The original compact-model motivation, later audit scope, numeric-error proposal and correction-transfer study remained in different documents. This made it too easy to describe another locally useful result as progress toward an unspecified paper.
5. **Human-dependent and access-dependent work was left unresolved while inference continued.** A larger prediction archive cannot replace verified speaker metadata, reviewed reference quality, annotation-dependent metrics or clear publication positioning.

The most recent suggestion to test script-constrained decoding should therefore remain a diagnostic idea. It is not the default next project. Neither Svarah WER number demonstrates that we improved an ASR model.

## Proposed correction

Keep the user's latest explicit audit direction as the active scope:

> Evaluate a defined set of open ASR systems on verified Indian-speech cohorts, measure recognition quality, failures and resource use under declared conditions, and establish what this adds to existing evaluations.

Use one completion matrix for that scope. Each model/language/corpus/metric cell must be marked complete, not run, unsupported, or blocked. Freeze the core shortlist and corpus conditions before additional inference; do not select them because of observed rankings. Complete or explicitly justify exclusions from the already selected core comparison before adding new model families.

Keep FLEURS as the controlled read-speech track and Svarah as a separately qualified Indian-English extension. Conversational claims require verified conversational data; if that is unavailable, state the reduced scope plainly and evaluate whether the resulting paper still has a contribution. Exact BRIDGE replication remains contingent on its authoritative assets and definitions. Do not manufacture unavailable demographic or code-switch labels, and do not use the published closed-model leaderboard as a matched comparator.

Retain primary WER/CER and failure coverage; consolidate existing WIL/error counts rather than inventing metrics. Add a controlled resource-use comparison to connect the audit to the original practical motivation, with explicit hardware/batch/precision and timing boundaries. GPU measurements cannot stand in for phone measurements. Complete a bounded reference/error review; record any human-dependent work as an assigned deliverable rather than endlessly expanding inference around it.

Write the paper's question, expected tables, artifact and contribution test before another model expansion. A contribution test should identify what existing work does not answer and what finding would change a user's or researcher's conclusion; more model rows alone do not pass it. A poor benchmark result does not create a need to train or repair that model.

## Preserve, stop, and next action

- **Preserve:** accepted predictions, source audits, repaired runtimes, scoring, paired diagnostics, viewers and reserved SPIRE separation.
- **Stop as default work:** further decoder fixes after acceptance, opportunistic method proposals, unrelated dataset expansion, and progress reports centered on job/artifact counts.
- **Do next if continuing the agreed audit:** reconcile the original model/metric/corpus commitments with the accepted evidence in one finite completion matrix, choose the smallest coherent remaining study, and tie each remaining experiment to its paper table or practical conclusion.

The appropriate current description is **a partial open-ASR audit with substantial validated infrastructure**. Calling it either a finished publishable study or no work at all would be inaccurate. No claim about a new contribution or publication acceptance is established by this alignment review.
