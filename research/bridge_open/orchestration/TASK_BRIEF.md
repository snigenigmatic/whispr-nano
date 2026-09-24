# Bounded ASR research handoff

Project owner: C. Kaustubh. This is a project-specific task brief for a future adapter or an agent in the existing workspace. It is not directly loadable as upstream freephdlabor configuration, and it grants no additional execution or spending authority.

## Immediate objective

Use the existing six-model evidence to identify the smallest experiment that resolves a material threat to the open conversational-ASR study. The immediate question is whether Whisper's observed failure rate is dominated by the selected 30-second decoding recipe. Preserve the script-mismatch observation as a separate hypothesis; do not mix it into this experiment.

Existing evidence: 83 model–conversation evaluations, 30 unique public recordings, 12 language labels. Broad comparisons cover only the common supported cohort; Qwen is Hindi-only in this Indic sample. Hindi's three recordings have been used for development. No sample already inspected becomes held-out again through renaming or resplitting.

## Roles and concrete outputs

1. **Coordinator:** maintain the current question, tried configurations, exact ORX node/run IDs, cost reservations, and why the next action addresses an unresolved issue. Avoid proposing a fresh research direction unless evidence rules out the present question.
2. **Evidence reviewer:** independently trace each material claim to the indexed config, manifest, prediction, metric and run log. Return issues with file references, severity, observed evidence, and a falsification or repair check. Review the scientific design before any manuscript wording.
3. **Experiment engineer:** specify one controlled comparison of the preserved 30-second condition with a documented native/short-window route on the exposed development cohort. Fix model revision, data, scorer, precision and decoder policy; record unavoidable recipe differences. Supply runnable changes and meaningful checks, not a prediction of the desired result.
4. **Writer:** draft only after the coordinator records which claims survive evidence review. Every numeric claim needs a source file and cohort. Unresolved issues and negative findings remain visible. An AI review score is not external peer review or publication acceptance.

Use the existing workspace agents for bounded work. A standalone provider-backed framework requires a separate LLM cost limit and verified provider access; Modal credits do not authorize external LLM calls. Keep retrieval/formatting work small; use independent scientific review where it can change the decision.

## Experiment contract before a launch

Record the hypothesis, comparison conditions, exact development IDs, primary WER/CER and completion measures, chunk/stitching policy, permitted language hints, runtime ceiling, cost reservation, and failure handling. A negative or unchanged outcome is valid. Do not tune a condition on a confirmation set, drop difficult outputs, or change normalization after seeing the result.

The current stage stops at a review and experiment specification. All GPU runs must later go through the existing ORX fixed entrypoint and external budget ledger. The fully reserved USD15 ledger requires provider-bill reconciliation before another reservation. No bypass through direct Modal calls, shell-generated cloud jobs, or the upstream AI-Scientist experiment tool is part of this task.

## Evidence rules

- Start with `review_packet.json`; verify indexed SHA-256 values before using claims. Retain full prediction records, including truncation and wrong-script output.
- The local packet contains paths and hashes, not a license to upload project files or private audio to third parties. SPIRE held-out data stays unopened.
- Keep WER/CER/WIL separate from unavailable BRIDGE-specific metrics. No surrogate is relabelled as HumynScore, CS F1 or PIER.
- Report descriptive uncertainty with its actual sampling unit. Three Hindi conversations do not support population-level superiority.
- Source checks and metric replay are necessary but do not certify transcript truth. Request a defined bilingual audit where the claim depends on that truth.
- Stop after one proposal and one independent review/revision pass, or earlier for a concrete blocker. Record the blocker. Do not iterate until an LLM awards an acceptance score.

## Completion

Return: (1) supported claims and their evidence; (2) the three most consequential unresolved issues; (3) one executable experiment specification with cost status; and (4) a decision of ready-for-controlled-run, needs-reference-review, or blocked-by-access/budget. This decision concerns the next experiment, not publication readiness.
