# Continuation protocol — Indian-English external-corpus check

Prepared for C. Kaustubh before new Indian-English inference. Status: source audit and 12-record native acceptance completed; full available-shard extension completed and independently validated. This is a benchmark extension, not a proposed trained method.

## Decision record

1. Preserve the complete FLEURS comparison and paired language diagnostic. Do not overwrite outputs, rerun selectively because scores look bad, or relabel the diagnostic as a general repair.
2. Keep the historical Humyn source-audio hold in effect. More models cannot repair incorrect audio/reference pairs.
3. Return to Indian English using existing legitimately cached Svarah data where possible. Verify current bytes and row mapping; old result CSVs are exposure evidence, not trustworthy replacements for source audio or a new inference run.
4. Carry forward the accepted native OmniASR CTC-1B and Qwen3-ASR-1.7B implementations. Extend the corpus before adding further model families. English must be verified as supported in the pinned official recipes.
5. Expand human review with Omni outputs while retaining the same 50-recording selection. Do not manufacture fluency judgments or infer error prevalence from the deliberately targeted examples.

## Question and scope

How do the two accepted native systems perform on available Indian-English Svarah recordings, and what errors remain? FLEURS Hindi rankings are not predictions about English; agreement or reversal is descriptive cross-corpus evidence, not proof of accent causality. Do not pool WER across unrelated languages or datasets.

The completed audit of located historical Svarah prediction files found 2,794 exposed IDs, 209 in the recovered 518-record shard. A fresh inference run does not make a previously inspected dataset held out. Preserve corpus speaker IDs only if original metadata establishes their mapping; row numbers, filenames guessed from conventions and demographic tuples are not substitutes.

## Input acceptance

- Recover original cached records, source/dataset identifiers, cache fingerprints, metadata, reference text and audio bytes. Record hashes before transformation. Distinguish an HF cache fingerprint from an upstream repository revision.
- Check duplicate IDs, exact audio and reference duplicates, durations, channels, declared and decoded sample rates, finite/range validity, silence and clipping. Preserve flagged rows and document exclusion reasons before inference.
- Decode with two independent audio readers where practical. If resampling is necessary, record source and target rate, exact algorithm/version and original/derived hashes. Never relabel a waveform's rate without resampling its samples.
- Use the complete waveform. Omni's accepted pipeline has a 40-second limit; longer items require a separately frozen, validated chunking recipe or an explicit common-cohort limitation. No silent truncation.
- Keep reference text out of model prompts and decoding decisions. Read references for scoring after prediction; provenance hashing is permitted beforehand.
- Do not open reserved SPIRE final audio or references as a substitute when Svarah has a blocker.

## First execution decision

Select a deterministic implementation cohort using audio IDs/duration and a fixed hash seed, without reference content or ASR scores. Freeze its manifest and selection rule before dispatch. Include the longest eligible waveform to exercise the runtime boundary. This cohort establishes input/runtime correctness only; its small size is not publication evidence.

Create an ORX child of each accepted native model node. Keep its fixed launch command, precision, model/tokenizer revisions and decoder configuration. The child changes corpus staging and declared language to English. Retain the native API/retained-token agreement check where supported. A real input/runtime failure requires diagnosis before scaling; it does not justify dropping a recording.

After the implementation cohort passes, freeze complete eligible corpus IDs and assess measured execution throughput before full dispatch. If source provenance or access remains unresolved, record the concrete blocker and continue documentation/review preparation rather than inventing a dataset mapping.

## Required outputs and analysis

Each condition records immutable code/config, original and staged source hashes, model assets, effective dtype/eval state, token/termination evidence, complete predictions, failures, runtime and provider job receipt. Cost is null until provider billing is verified.

Use the existing primary normalization (Unicode NFC, casefold, punctuation to spaces, meaningful marks retained, no numeral conversion). Report WER/CER, coverage, empty outputs, termination/cap flags and raw transcript examples. Scoring alternatives, if explored, must be separately labelled rather than replacing the primary result. Bootstrap by verified speaker only when identifiers and their mapping are established; otherwise withhold speaker confidence intervals and state the actual available dependence unit.

The human review packet must remain blinded, save each reviewer judgment explicitly, and distinguish random and targeted selection. A human must verify audio/reference quality and transcript language/meaning. Automated checks support this process; they do not count as completed human adjudication.

## Publication decision

A possible contribution is a reproducible audit of system failures and evaluation sensitivity across source-verified corpora. This remains a candidate, not an established novelty claim. Compare the final findings against the documented script-collapse and Indian-language evaluation prior work. Any new repair must be frozen before an independent confirmation evaluation. The current benchmark can support a capstone demonstration now; it cannot promise conference acceptance.

## Execution addendum

The complete third cache shard is the only successfully recovered source audio. The full extension includes all 518 records, including 16 clipping flags; its manifest is `dbddbaeef3a1f78ff85d0cd741e85d86daf7c868b39ee8d4c48f67a21bfafedd`. All waveforms are already mono 16 kHz, so no resampling or amplitude conversion is applied. The 12-record checks are nested in this cohort. [Execution and correction details](SVARAH_EXECUTION_20260923.md) distinguish cancelled dispatches, accepted runs and pending analysis.

The completed [518-record comparison](artifacts/svarah_comparison_v1/RESULTS.md) is descriptive. Next work is audio/reference and paired-error review; no new recognition method has been established.
