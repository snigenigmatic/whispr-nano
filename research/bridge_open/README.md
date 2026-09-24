> **Current study:** Start with [CURRENT_STATUS.md](CURRENT_STATUS.md) for the accepted Whisper, Qwen and OmniASR results, evidence commit, limitations and active Indian-English continuation. The original notes below remain historical.

> **Audio QC correction (2026-09-22):** Five public files contain transcript text over exactly silent audio. Affected Assamese/Bengali quality comparisons and aggregates containing them are withdrawn pending adjudication. Original metrics below are preserved as historical records. See the [source-verified audit](artifacts/audio_content_audit/RESULTS.md).

Current accepted execution: [corrected FLEURS Whisper comparison](artifacts/fleurs_whisper_corrected_v3/RESULTS.md), with [acceptance scope](FLEURS_RESULT_HOLD.md). The earlier [six-model public Humyn pilot](EXPANDED_RESULTS_20260922.md) and [Hindi development results](RESULTS_20260922.md) are historical; affected source-audio comparisons were withdrawn and must not be used as the current benchmark.

Full-study follow-up: [scope, execution gates and access requirements](FULL_STUDY_SPEC.md), with [completed CPU error and oracle analyses](artifacts/study_panels/RESULTS.md).

Research-loop tooling: [freephdlabor assessment and prepared evidence handoff](orchestration/FREEPHDLABOR_ASSESSMENT.md).

# BRIDGE open extension

This folder contains the protocol and executed pilot for an independent open-model
conversational-ASR study. Read [PROTOCOL.md](PROTOCOL.md); the binding metric
availability record is [metrics_contract.json](metrics_contract.json).

The live Hugging Face discovery on 22 September 2026 produced a
[27-model roster](MODEL_ROSTER.md), an immutable-revision
[machine-readable roster](model_roster.json), a [coverage matrix](coverage_matrix.csv),
and a [107-cell model/language plan](benchmark_plan.json). The plan is not
launchable until the corpus and runtime configurations are frozen. It does not
launch cloud jobs or change the authorized spending cap. Raw publisher metadata
is retained in `artifacts/`; the corrected Kyutai repository and pinned Omni
language registry are in the supplement. Model cards are cached under the
ignored `.cache/bridge_open/` directory and hashed in the snapshots.

Before a run, freeze source/model/normalizer revisions and scorer versions,
create a hash manifest, retain all full-audio or chunked predictions and
failures, publish per-language coverage and a comparison-specific common cohort,
and bootstrap by complete conversation. WER, CER, and standard JiWER WIL are
independently reportable after that freeze; SemSim is independently preparable
once its encoder and aggregation are frozen. Do not treat the public 37-file
Humyn card as confirmed BRIDGE data or claim the same/different score without a
matched asset and scorer package.

## Execution started

All 37 public recordings have been acquired and hashed. Read [CORPUS_QC.md](CORPUS_QC.md) for seven timestamp warnings and [PILOT_CONTRACT.md](PILOT_CONTRACT.md) for the three-Hindi-conversation development pilot. [expansion_runs.json](artifacts/expansion_runs.json) records the expansion and evidence-replay run IDs. Six model recipes produced 83 conversation predictions on 30 distinct eligible files. The isolated `.venv-bridge` runtime avoids a mixed-version shared environment; dependencies are pinned in `requirements.lock.txt`. Modal inference jobs are complete and stopped. The full USD15 reservation cap is retained; reserved SPIRE data remains unopened.
