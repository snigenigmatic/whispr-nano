# Current study status — 24 September 2026

Owner: C. Kaustubh. The active scope is the [finite six-system core study](core_study/CORE_STUDY_20260924.md); its [completion matrix](core_study/completion_matrix.json) replaces open-ended expansion and repair proposals. Historical details follow below.

- **All 20 supported quality cells are accepted:** 13,232 primary predictions over 2,840 unique recordings. Four Qwen Bengali/Assamese conditions remain explicitly unsupported. All GPU jobs in this finite batch have finished. [Complete quality results](core_study/QUALITY_RESULTS.md).
- The first Qwen 0.6B Hindi dispatch failed a native-API configuration check. Its repaired rerun passed independent acceptance: **18.50% WER on all 418 Hindi recordings**. Both attempts are preserved separately.
- On the **518 available Svarah recordings**, Small achieved **9.69% WER**, Turbo **12.47%**, and Large-v3 **8.24%**. This partial, previously exposed shard supports a descriptive comparison, not a representative full-Svarah or untouched confirmation claim.
- Small's Bengali **194.58%** and Assamese **214.50%** WER include severe repetition and termination failures in the declared decoder recipe. Source audio, frontend and weight checks passed; these scores remain visible with insertion, normalized/raw-empty and observed-EOS diagnostics. Acceptance verifies artifacts, not decoder optimality.
- **All six resource conditions are accepted** on the same pre-exposed 12-item A100 80GB panel. [Matched resource results](core_study/RESOURCE_RESULTS.md) report panel quality, timing and memory. This small panel reverses Small/Large-v3's quality ordering relative to the 518-item shard; it cannot establish a corpus-wide or phone deployment ranking.
- The [working manuscript](../../core_audit.tex) and [PDF](../../core_audit.pdf) contain the completed tables and limitations. **Actual human review and a defensible contribution remain submission gates.** [Review readiness](core_study/REVIEW_READINESS.md) records these limits.
- The [FLEURS handoff](core_study/REVIEW_HANDOFF.md) is ready for 50 clips / 168 anonymous cards. The [separate English handoff](listening_review_core_en_v1/REVIEW_HANDOFF.md) contains 12 pre-exposed resource clips / 72 cards. Both remain unreviewed; tests use synthetic ratings only. No reviewers have been contacted.
- [Literature positioning](core_study/LITERATURE_POSITIONING.md) rules out a first/comprehensive Indian-ASR benchmark claim. No new ASR method, demographic fairness, matched BRIDGE comparison or phone measurement is claimed. The reserved SPIRE test set remains unopened.
- [Local evidence viewer](artifacts/core_study_v1/index.html) and [reproduction instructions](core_study/REPRODUCTION.md) provide inspection and provenance. Job reservations are not measured charges. No further GPU dispatch is required to complete the frozen matrix.

## Preserved prior evidence (23 September)

## Accepted evidence

Evidence commit `1468f7b42a75c91254823fe0dc7a22999de748e8` on `codex/native-hf-evidence-20260922`, worktree `/Users/prateekraj/whispr-bridge-pilot`, preserves the completed expansion. It builds on the corrected Whisper/Qwen evidence in `cf75c36`.

| Declared system | Hindi, 418 recordings | Bengali, 920 | Assamese, 984 |
|---|---:|---:|---:|
| Whisper Turbo | 31.35% | 85.86% | 134.71% |
| Whisper Large-v3 | 28.21% | 74.70% | 113.30% |
| Qwen3-ASR-1.7B | 13.09% | Not evaluated | Not evaluated |
| OmniASR CTC-1B | 21.50% | 24.52% | 33.76% |

Entries are micro WER with the frozen Indic-mark-preserving scorer. The dataset is FLEURS official test: 2,322 recordings and 7,384 primary system predictions. Different precision and decoding recipes make this a system comparison. No new weights were trained.

Five expansion jobs completed: one paired Whisper language diagnostic, one Omni implementation acceptance check, and three complete Omni language cohorts. Native API agreement passed on all nine acceptance recordings; full cohort/source-hash checks and independent rescoring passed. Eight corrupted-evidence cases were rejected. Automatic language selection did not provide a general Whisper repair. The diagnostic's 44 recordings overlap the benchmark and are not additional independent observations.

- [Complete results, run links and limitations](EXPANDED_STUDY_20260923.md)
- [Evidence manifest and reproduction guide](artifacts/expanded_study_v1/REPRODUCTION.md)
- [Audio/transcript viewer](artifacts/benchmark_review_v2/index.html)
- [Prior-art assessment](artifacts/expanded_study_v1/PRIOR_ART.md)
- [Descriptive failure tails and paired recording counts](artifacts/failure_tails_v1/RESULTS.md)
- [Research decisions and next experiment](CONTINUATION_PROTOCOL_20260923.md)

## Research interpretation

The accepted evidence establishes performance differences on these fixed read-speech cohorts and a negative result for automatic Whisper language selection. It does not establish a new recognition method, demographic fairness, Indian-English accent robustness, phone performance, or full reproduction of Humyn BRIDGE. Training-data overlap with public benchmarks is not independently ruled out. Wrong-script decoding already has direct prior art.

Post-hoc failure-tail analysis shows why averages are insufficient: on Hindi, Omni has 18 recordings with WER ≥ 100%, versus 3 for Large-v3 and 0 for Qwen, despite Omni's better mean than Large-v3. Omni has no such recordings in the Bengali or Assamese cohorts. This threshold counts word edits, not verified semantic failures; the observation requires transcript review before an explanation or repair is claimed.

Human transcript adjudication is pending. The [expanded blinded packet](listening_review_v2/index.html) retains the same 50 utterances and now includes 168 anonymous transcript cards across all available systems. The frozen original remains unchanged. Merely browsing outputs does not complete a review; script matching alone is not a language or meaning judgment.

## Active continuation

Recovered 518 original Svarah audio/reference pairs from one complete cached shard. Both native 12-record implementation checks passed independent acceptance. Both completed all 518 available records, including 16 clipping flags and 209 previously exposed IDs. Independent source/model/code checks and rescoring passed. Qwen achieved **13.60% WER** (733/5,389 word edits); Omni achieved **64.63%** (3,483/5,389). There were 0 and 26 empty outputs, respectively, all retained in scoring. Qwen had fewer edits on 412 recordings, equal edits on 93, and more on 13. [Complete Svarah comparison](artifacts/svarah_comparison_v1/RESULTS.md). This is an availability-selected subset of 6,656, not full Svarah or an untouched confirmation set. No speaker identities are inferred. The reserved 198-speaker SPIRE final set stays unopened. See the [source correction and execution record](SVARAH_EXECUTION_20260923.md).

The user removed the total compute budget cap on 23 September and authorized continuation. Per-job timeouts, duplicate-launch prevention and provenance checks remain. Reservations are estimates, not provider billing. No new run is implied by this status document; run receipts record actual dispatches.

The [post-hoc Svarah text diagnostics](artifacts/svarah_error_diagnostics_v1/RESULTS.md) give a concrete follow-up: 309 Omni outputs contain non-Latin letters and 82 mix Arabic and Latin letters, versus 2 and 0 for Qwen. The WER gap also appears among recordings at least five seconds long. These are Unicode/output observations, not human language judgments or proof of a cause. These observations remain part of the bounded error review; script repair is not in the active core-study scope.

## Holds and historical material

- Humyn public-file comparisons affected by silent audio remain withdrawn; see [audio QC audit](artifacts/audio_content_audit/RESULTS.md).
- IndicConformer has no completed baseline: the existing account's gated file request returned HTTP 403. No gate terms were accepted.
- Prior SFT/RL/correction-transfer results do not demonstrate the current benchmark's contribution.
- Paper acceptance and publication remain unestablished; submission requires defensible positioning and completed evidence, not only a larger model table.
