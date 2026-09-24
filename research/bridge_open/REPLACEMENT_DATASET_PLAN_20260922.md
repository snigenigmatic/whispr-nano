# Replacement benchmark plan — 22 September 2026

Prepared for C. Kaustubh. The public Humyn files remain preserved with the [audio QC correction](artifacts/audio_content_audit/RESULTS.md). Changing the corpus creates a new benchmark; it does not validate the withdrawn results or reproduce BRIDGE's private evaluation.

## Immediate choice: FLEURS development pilot

Use Google's official [FLEURS release](https://huggingface.co/datasets/google/fleurs), pinned to `70bb2e84b976b7e960aa89f1c648e09c59f894dd`, under CC BY 4.0. The original [paper](https://www.alphaxiv.org/abs/2205.12446), Section 2, describes human recordings, a second-worker audio/sentence check, read speech and official train/development/test partitions. Hindi, Bengali and Assamese are available in the same source. This gives a practical reference benchmark for the languages affected by the current Humyn audio issue.

The new acquisition script selects 12 development recordings per language by a fixed filename hash, without consulting reference content or ASR scores. It verifies complete archive hashes against the pinned repository's LFS SHA256, preserves original audio bytes, records raw and official normalized references, and checks sample count/rate, finite values, silence fraction, clipping and duration. All suspicious rows are retained and flagged. Automated checks do not certify semantic transcript alignment. No test audio or references are opened during this preparation.

This 36-utterance set is only for integration checks. Do not publish it as a benchmark, tune methods and report its result as held out, or infer population fairness from it. The TSV's sentence ID is not a speaker ID. A full study must explicitly document the uncertainty unit and avoid claiming independent speakers from this export.

Preparation completed: **36/36 clips passed automated checks, totaling 447.12 seconds (7.452 minutes)**. All three complete source archives matched the pinned upstream LFS hashes. An independent scipy WAV decode also passed for every clip. See the [receipt](artifacts/fleurs_dev_v1/receipt.json) and [independent validation](artifacts/fleurs_dev_v1/independent_validation.json). No inference or GPU job was launched.

## Two complementary study tracks

| Track | Candidate | Purpose and limits |
|---|---|---|
| Multilingual read speech | FLEURS official test, after recipe freeze | Consistent multilingual accuracy/failure comparison. Short read clips do not replace spontaneous conversations or assess long-form decoding. Public evaluation exposure in model training is unknown unless authors document it. |
| Indian English | [AI4Bharat Svarah](https://github.com/AI4Bharat/Svarah) | Direct fit to the original accent question: read and spontaneous Indian English, 117 speakers per the release. Prior local 2,500-row evaluations are already exposed. Recover verified row/speaker mapping and audit exposure before subgroup or held-out claims. Current HF mirror is gated; existing access/cache must be reused legitimately. |
| Optional independent Indic replication | [Kathbath / IndicSUPERB](https://github.com/AI4Bharat/IndicSUPERB) | Twelve languages, clean/noisy and Test Known/Test Unknown conditions. No Assamese. Audit source speaker IDs. These split names concern Kathbath's own training partition; they do not establish that a pretrained model has never seen a speaker. |
| Later conversational extension | [IndicVoices](https://huggingface.co/datasets/ai4bharat/IndicVoices) | Richer spontaneous speech and demographics, but gated. Current release exposes train/valid, not an official test in the card. Verify access and speaker separation before choosing an evaluation subset. |

See the [bounded multilingual access review](../publication/replacement_multilingual_dataset_check_20260922.md) for exact access and license qualifications. Svarah covers Indian-accented English; FLEURS English is not a substitute for that population.

## Execution sequence

1. Finish FLEURS development acquisition and waveform/metadata QC; preserve the frozen manifest and receipt.
2. Create a new ORX child for one short-utterance smoke inference using an already pinned model/runtime. Inspect exact failures, normalized scoring and representative audio/reference pairs before scaling. Preserve no-EOS/empty/error outputs. The existing Humyn hold is not lifted by this proposal; a new corpus-specific QC decision is required.
3. Freeze the inference recipe and primary common-language model comparisons. Use only documented supported languages for each model; do not rank models over unequal language sets. Existing Qwen routing covers Hindi in this three-language pilot, not Bengali/Assamese.
4. Freeze official-test manifests, record public benchmark exposure and model revision, then evaluate the selected model set with common conditions. Report WER/CER and failure coverage per language; benchmark runtime and memory separately under controlled hardware.
5. Add Svarah as a separate domain, not a pooled WER mixture. Examine whether model rankings and the accuracy/resource trade-off persist across datasets. A reproducible ranking table alone is not a guaranteed publication contribution; the paper needs a supported finding, uncertainty, error analysis and released evaluation artifacts.

The approved ceilings remain $15 Modal and a separate $50 HF. This preparation uses local CPU and downloads only; it launches no GPU and makes no claim about the current billed balance. Any next HF dispatch retains the existing per-job timeout and budget reservation guard.
