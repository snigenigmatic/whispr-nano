# Core audit evidence — 24 September 2026

Owner: C. Kaustubh. All 20 supported quality comparisons and six resource profiles are accepted. There are 13,232 primary predictions over 2,840 unique recordings. Human review remains incomplete.

- [Quality table](../../core_study/QUALITY_RESULTS.md) and [resource table](../../core_study/RESOURCE_RESULTS.md).
- [PDF working draft](../../../../core_audit.pdf) and [LaTeX source](../../../../core_audit.tex).
- [Local inspection page](index.html); serve the repository root so original audio paths resolve.
- [Reproduction guide](../../core_study/REPRODUCTION.md) and [completion matrix](../../core_study/completion_matrix.json).
- [Review handoff](../../core_study/REVIEW_HANDOFF.md) and [English panel handoff](../../listening_review_core_en_v1/REVIEW_HANDOFF.md).

## New GPU dispatches

The failed configuration attempt is retained separately. No further GPU jobs are part of this batch. Earlier accepted quality runs are referenced in `quality_results.json`, not counted as new dispatches here.

| Artifact | Provider status | Run | Job |
|---|---|---|---|
| [core_qwen06_hindi_failed_v1](../core_qwen06_hindi_failed_v1/orx_source_receipt.json) | failed | `61219347-d8a4-438c-9cbc-9d0058d096b4` | [HF job](https://huggingface.co/jobs/snigenigmatic/6ab4c4826b030d633f68d6f3) |
| [core_qwen06_hindi_v1](../core_qwen06_hindi_v1/orx_source_receipt.json) | done | `04965a3b-7e62-46f4-a6ee-e361b18eca20` | [HF job](https://huggingface.co/jobs/snigenigmatic/6ab4c55c52d0dbd7f1d8988d) |
| [core_qwen06_svarah_v1](../core_qwen06_svarah_v1/orx_source_receipt.json) | done | `a640aca5-5300-458f-802a-5ea285aa0554` | [HF job](https://huggingface.co/jobs/snigenigmatic/6ab4c38c52d0dbd7f1d89832) |
| [core_resource_omni_v1](../core_resource_omni_v1/orx_source_receipt.json) | done | `ae6ce853-45ab-42e3-8f57-ce5aea1a7f1b` | [HF job](https://huggingface.co/jobs/snigenigmatic/6ab4c4776b030d633f68d6f1) |
| [core_resource_whisper_qwen_v1](../core_resource_whisper_qwen_v1/orx_source_receipt.json) | done | `ab478b7f-56cf-4629-8c0e-04e0f8c82511` | [HF job](https://huggingface.co/jobs/snigenigmatic/6ab4c5ae52d0dbd7f1d89895) |
| [core_whisper_large_svarah_v1](../core_whisper_large_svarah_v1/orx_source_receipt.json) | done | `ae2398b9-8b45-408e-8f91-6ac13751102f` | [HF job](https://huggingface.co/jobs/snigenigmatic/6ab4c9b352d0dbd7f1d8996c) |
| [core_whisper_small_as_v1](../core_whisper_small_as_v1/orx_source_receipt.json) | done | `5a91f1a3-1a17-4538-8900-865eccb07fd0` | [HF job](https://huggingface.co/jobs/snigenigmatic/6ab4c98452d0dbd7f1d89966) |
| [core_whisper_small_bn_v1](../core_whisper_small_bn_v1/orx_source_receipt.json) | done | `db4d854f-acef-4b4b-a476-f26e05a59b33` | [HF job](https://huggingface.co/jobs/snigenigmatic/6ab4c9726b030d633f68d793) |
| [core_whisper_small_hi_v1](../core_whisper_small_hi_v1/orx_source_receipt.json) | done | `d54ce732-3bdd-4619-9bea-b7686e6cdbde` | [HF job](https://huggingface.co/jobs/snigenigmatic/6ab4c6586b030d633f68d741) |
| [core_whisper_small_svarah_v1](../core_whisper_small_svarah_v1/orx_source_receipt.json) | done | `be042b3b-90de-4ecc-a50c-8ac333206e34` | [HF job](https://huggingface.co/jobs/snigenigmatic/6ab4c9986b030d633f68d799) |
| [core_whisper_turbo_svarah_v1](../core_whisper_turbo_svarah_v1/orx_source_receipt.json) | done | `9d3e509c-6b69-4acf-90d9-0bf018f2ebd6` | [HF job](https://huggingface.co/jobs/snigenigmatic/6ab4c86e6b030d633f68d766) |

## Preservation boundary

`evidence_manifest.json` lists copied file hashes and sizes; it excludes itself. The snapshot is a local evidence commit, not a public dataset release. Original audio outside executed source archives remains at its verified local source paths. Provider costs remain unverified; job reservation estimates must not be reported as charges.

The public interpretation remains limited by the partial pre-exposed Svarah shard, recipe differences, a twelve-record resource panel, and unfinished human listening review. Large Bengali/Assamese Whisper error rates include repetition, missing observed EOS and punctuation-only outputs; independent acceptance verifies evidence integrity rather than optimal decoding.
