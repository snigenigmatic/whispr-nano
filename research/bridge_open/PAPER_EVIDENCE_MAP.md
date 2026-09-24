# Paper evidence map

Working topic: an auditable comparison of open ASR system accuracy and failure profiles on Indian-language and Indian-English speech. This is a candidate paper scope, not a claim of novelty or acceptance.

| Candidate statement | Current evidence | Status / restriction |
|---|---|---|
| Evaluated systems differ substantially on the three FLEURS language cohorts. | [Complete comparison](EXPANDED_STUDY_20260923.md), frozen input/model/code hashes, independent rescoring, complete outputs. | Supported for these declared configurations. No architecture-only causal interpretation. |
| The lowest mean WER does not always imply the fewest very high-error recordings. | [Failure-tail counts](artifacts/failure_tails_v1/RESULTS.md): Omni Hindi has lower micro WER than Large-v3 but more recordings with WER ≥ 100%. | Supported descriptive observation. The threshold is an edit-count condition, not a verified semantic harm category. Post-hoc analysis. |
| Automatic Whisper language selection is a general fix. | [Paired native diagnostic](artifacts/language_conditioning_analysis_v1/RESULTS.md). | Not supported. Hindi worsened, Bengali was unchanged, Assamese remained poor. The 44-recording selection is diagnostic, not a full-corpus estimate. |
| Script diagnostics establish incorrect language or meaning. | Unicode counts, retained transcripts, pending human review. | Not supported. Shared scripts and untracked scripts limit the heuristic. Wrong-script failure itself already has prior art. |
| Native system performance differs on recovered Indian-English audio. | [Svarah 518-record comparison](artifacts/svarah_comparison_v1/RESULTS.md): Qwen 13.60% WER, Omni 64.63%; complete paired source-verified outputs. | Supported only for this availability-selected, partly preexposed subset. No accent-causality or full-Svarah claim; human reference review pending. |
| Script mixing is visible in the native system's English-corpus outputs. | [Post-hoc text profile](artifacts/svarah_error_diagnostics_v1/RESULTS.md): 309 Omni outputs have non-Latin letters; 82 mix Arabic and Latin letters. | Supported as a Unicode text observation. It does not identify spoken/output language, explain the mechanism, or establish novelty over prior script-collapse research. |
| Results establish speaker fairness, phone suitability, or BRIDGE equivalence. | Required matched metadata, device measurements and authoritative BRIDGE assets are missing. | Exclude these claims. |
| A new repair improves recognition. | No new model or repair has been evaluated on an untouched confirmation set. | Exclude until a frozen method, proper comparator and independent confirmation exist. |

## Suggested manuscript structure

1. **Question and scope:** explain what aggregate ASR accuracy does and does not reveal across the selected language/corpus conditions. State a precise incremental question after the external-corpus results and human review are available.
2. **Protocol:** provenance, dataset exposure, supported-language coverage, complete-waveform routing, exact native runtimes and mark-preserving normalization. Explain the source-audio QC correction transparently without mixing withdrawn and accepted results.
3. **Results:** complete-cohort WER/CER, paired source-sentence uncertainty where available, and error-distribution counts. Report native language-conditioning results as a separate diagnostic with its selection rule.
4. **Human analysis:** blinded language/meaning/reference judgments, reviewer fluency, number of reviewers and adjudication process. Separate random and targeted samples; report disagreement rather than inventing consensus. This section is currently empty of human findings.
5. **External corpus and limitations:** Svarah input checks, exposure and mapping limits; no pooling with FLEURS. Discuss training-data contamination uncertainty, differing decoding/precision, unsupported-language caveats and the absent phone evaluation.

## Submission gate

Before calling the work submission-ready, complete fluent human review, retain the independently validated external-corpus results with their subset limitations, and identify an explicit contribution beyond existing script-collapse/evaluation literature. A benchmark release may be a useful contribution only when its question, source quality, reproducibility and incremental findings justify it. Additional GPU runs alone do not satisfy this gate.

The existing artifacts already support a capstone demonstration and an honest progress report. They should be presented as completed empirical work with open scientific questions, not as a promised publication or a trained breakthrough.
