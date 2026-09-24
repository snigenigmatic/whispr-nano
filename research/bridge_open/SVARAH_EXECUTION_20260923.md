# Svarah source recovery and execution — 23 September 2026

Owner: C. Kaustubh. Scope: Indian-English extension of the accepted native Qwen and Omni comparison. This document distinguishes implementation acceptance, actual benchmark coverage and unfinished human adjudication.

## Recovered data and correction

The existing authorized Modal cache `ast-asr-cache` contains three Arrow shards for `ai4bharat/Svarah`, revision `ebbf7777fe771490696a3f7b007097606fa8c924`. Transfers of the first two shards failed with upstream-storage errors; incomplete copies were not used. The third shard was recovered in full: 86,494,400 bytes, SHA-256 `7f8ec2d10d827bb2329ff96164b332bb09352b62a5e9c8fa059415c6b1ffad11`, 518 records.

Records were joined to the pinned 6,656-row metadata by exact audio basename/reference pair, then independently checked against original Arrow bytes. The matched global indices are 6138–6655. A first preparer version erroneously used shard-local indices for reference lookup. The independent check detected this before any GPU launch. Those v1 inputs are quarantined; the corrected v2 implementation bundle and full-v1 cohort both pass original-source checks. [Invalidation record](artifacts/svarah_input_audit_shard02_v1/INVALIDATION.md).

All 518 source files are finite, nonsilent, mono 16 kHz PCM16 WAV; bytes are preserved. SoundFile and SciPy decoding agree. Duration ranges from 0.1993125 to 27.8063125 seconds. There are 16 clipping-flagged recordings. Clipping is a retained source-quality flag in the benchmark, not an exclusion. Reference validation is structural, not human confirmation of what was spoken.

An audit of prior prediction files found 2,794 exposed IDs across Svarah; 209 occur in the recovered shard. No verified speaker mapping is available. This cohort is availability-selected and exploratory. The reserved 198-speaker SPIRE final set remains unopened.

## Frozen cohorts

| Cohort | Selection | Manifest SHA-256 | Role |
|---|---|---|---|
| 12-record v2 | Longest eligible ≤40 s, then 11 IDs with lowest fixed-seed hash; clipping-flagged rows excluded from this runtime check only | `14c08d00b5e585de77ee3af6b486b05980e0cc9e8a31139654437b2ed8082dac` | Input/native-runtime acceptance; no ranking claim |
| All recovered 518 | Every valid record in the complete recovered shard; all 16 clipping flags retained | `dbddbaeef3a1f78ff85d0cd741e85d86daf7c868b39ee8d4c48f67a21bfafedd` | Descriptive external-corpus benchmark; not full Svarah |

The 12 records are included in the 518, so they are not extra independent evidence. Both models receive exactly the same frozen IDs, complete waveforms and references. Neither model sees references in its recognition prompts.

## Executions

| Experiment | ORX experiment | Run / provider | Result |
|---|---|---|---|
| Omni implementation acceptance | `9e4bcbe5-8422-41eb-9881-aad42062f89a` | `b647d90b-5cb2-4edd-9061-92347f769fd0`; [HF job](https://huggingface.co/jobs/snigenigmatic/6ab394d651992417dfcd682b) | DONE, independently accepted 12/12; public native API and retained tokens agree on all outputs |
| Qwen implementation acceptance | `6d9f77db-dbab-43c9-bcd3-129c95819311` | `f83430e4-2b69-4bad-8e2a-8d85a698e617`; [HF job](https://huggingface.co/jobs/snigenigmatic/6ab394e651992417dfcd682f) | DONE, independently accepted 12/12; all outputs terminate with EOS, no observed token cap |
| Omni recovered-shard benchmark | `edcf01bd-f947-4d6b-a439-1fe352fd90d6` | `c086b84f-e5ed-4ef8-91a1-397b1c155676`; [HF job](https://huggingface.co/jobs/snigenigmatic/6ab3971052d0dbd7f1d83bb2) | DONE, independently accepted 518/518; API agreement on all outputs |
| Qwen recovered-shard benchmark | `6105909c-cd5e-4fc7-97f8-c3f6d0ef6c29` | `e5f19c1e-4479-4334-a087-5229eb433b80`; [HF job](https://huggingface.co/jobs/snigenigmatic/6ab397f751992417dfcd6930) | DONE, independently accepted 518/518; EOS present and no token-cap hits for all outputs |

One initial Qwen dispatch (`bea416b1-6a05-486e-862a-464ab536688f`, commit `4dc9350`) was cancelled during dependency installation because staging had retained the 12-item audit receipts alongside the correct 518-item manifest. No experiment result was produced. The provisional node was repaired by requiring full-cohort receipt counts and exact manifest hashes, tested to reject the stale receipts, and committed as `80ec0e8` before relaunch. [Cancelled dispatch record](artifacts/svarah_cancelled_dispatch_v1/receipt.json).

Answered experiment nodes are frozen. The 518-record experiments are children of their successful input-acceptance nodes. All use the same project command, `bash experiments/hf_entrypoint.sh`. Model assets and native inference settings stay fixed: Omni BF16 CTC batch 1 with official API agreement enabled; Qwen BF16 greedy batch 8 with explicit English and retained termination traces. There is no training or decoder tuning between acceptance and benchmark.

## Evidence and reproduction

- [Input audit for 12 records](artifacts/svarah_input_audit_shard02_v2/receipt.json) and [independent comparison](artifacts/svarah_input_audit_shard02_v2/independent_validation.json).
- [Input audit for all 518](artifacts/svarah_input_audit_shard02_full_v1/receipt.json) and [independent comparison](artifacts/svarah_input_audit_shard02_full_v1/independent_validation.json).
- [Qwen acceptance](artifacts/qwen_svarah_acceptance_v2/local_independent_acceptance.json), [Omni acceptance](artifacts/omni_svarah_acceptance_v2/local_independent_acceptance.json). These directories retain exact executed source archives, predictions, configs, environment hashes and run receipts.
- [Producer and exact preparation modes](svarah_extension/PROTOCOL.md), [independent source validator](validate_svarah_inputs.py), [independent run validator](accept_svarah_runs.py).

For full input preparation, with the complete source shard locally present:

```sh
.venv/bin/python research/bridge_open/svarah_extension/prepare.py --available-shard svarah-test-00002-of-00003.arrow --all-available
.venv/bin/python research/bridge_open/validate_svarah_inputs.py data/bridge_open/svarah_native_shard02_full_v1/bundle --expected-count 518
```

Never launch from quarantined v1 inputs. Do not reinterpret missing source shards as zero-error or zero-failure recordings. Accepted performance tables include empty outputs and all source-quality flags. No speaker confidence intervals, full-corpus claims, accent-causality conclusions or human error judgments are justified by this source audit alone. Provider billing remains unverified; budget reservations are not measured cost.

## Accepted result and decision

[Final comparison](artifacts/svarah_comparison_v1/RESULTS.md): Qwen 13.60% WER and 9.21% CER; Omni 64.63% WER and 52.72% CER. All 518 recordings are scored for each system, including Omni's 26 empty outputs (Qwen has zero). Qwen has fewer word edits on 412 paired recordings, equal edits on 93, and more on 13. These finite-cohort facts do not establish an accent mechanism or speaker fairness.

[Independent evidence negative checks](artifacts/svarah_evidence_checks_v1/negative_checks.json) reject corrupted summary, incomplete provider status, archive path escape and changed runner hashes, including under optimized Python. [Input negative checks](artifacts/svarah_evidence_checks_v1/input_negative_checks.json) reject missing recordings, incorrect global IDs/source rows, altered audio hashes and incorrect references. The run acceptor additionally compares the archived independent validation receipt to a newly recomputed source check, preventing stale-cohort receipts from being accepted.

Stop this GPU batch after the two complete benchmark runs. Next, review raw audio and transcripts for paired failures and reference/normalization issues. The statistical dependence and availability-selection limitations remain; more model executions would not resolve those gaps.
