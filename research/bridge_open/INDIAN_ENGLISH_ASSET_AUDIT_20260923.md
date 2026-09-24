# Indian-English asset audit — 23 September 2026

This is the initial metadata inventory, made before source recovery and inference in this session. The [subsequent execution record](SVARAH_EXECUTION_20260923.md) supersedes its availability blockers: 518 original Svarah records were recovered and both native system runs completed. The reserved 198-speaker SPIRE validation audio/references remain unopened. Asset paths outside this evidence package refer to the main workspace.

## Decision

Do not use Svarah as a speaker-disjoint or untouched-confirmation set now. A benchmark of frozen pretrained systems may deliberately reuse its exposed rows if it declares that scope, but it cannot present them as an untouched confirmation. A newly selected, utterance-disjoint exploratory follow-up is also possible because the pinned release metadata has no verified row-to-speaker identifier and 2,794 Svarah IDs are already exposed by prior local evaluations.

The viable future confirmation corpus is SPIRE-SIES, using only its already inventoried development side until the study protocol is frozen. The 198-speaker validation set remains sealed. Before any evaluation expansion, reconcile the exact authorized source release and license, then make a hash-checked development manifest and stage a small, deterministic speaker-disjoint development slice. No current artifact establishes a usable SPIRE license.

## Independently checked in this workspace

### Svarah evidence and availability

- [paper/data/svarah_metadata_raw.jsonl](../../paper/data/svarah_metadata_raw.jsonl) contains 6,656 metadata rows and has SHA-256 `ae875faf9c12fecff64c765e2a116d413e897fc63c416f60f28f39adc9793d47`.
- Its paired provenance file records `ai4bharat/Svarah`, test split, revision `ebbf7777fe771490696a3f7b007097606fa8c924`, dataset fingerprint `ac19cf30d32b3aae`, and a metadata-only extraction (`audio_bytes_exported: false`). The recorded feature schema contains an audio path, transcript, duration, and demographic fields, but no `speaker_id`.
- [paper/data/original_svarah_manifest.jsonl](../../paper/data/original_svarah_manifest.jsonl) has 6,656 rows and SHA-256 `0e859ea406e8129db84a6ad90b5a3aad9261417976253e7a6fe172bbaa7c63e4`.
- The local result CSVs have one shared 2,500-ID panel for Tiny, Small, Large-v3, Distil, and Qwen. The 450-row Saaras panel overlaps that panel in 156 IDs. Their checked union is therefore **2,794 exposed Svarah IDs**, leaving at most 3,862 rows before any additional exposure audit.
- No directory named `datasets--ai4bharat--Svarah` or `ai4bharat___svarah` exists in the local Hugging Face cache. The only visible local dataset-cache directories are FLEURS and Humyn. This does not disprove the documented remote cache.
- Original Svarah sample rates were not established in this audit: the local metadata has no sample-rate field and no source audio was opened. Existing fetchers request/resample 16 kHz mono output; that is an output contract, not evidence of the original rate.

### SPIRE development metadata

- [data/numeric_asr/spire_20260917/metadata.jsonl](/Users/prateekraj/whispr-nano/data/numeric_asr/spire_20260917/metadata.jsonl) has 35,299 metadata-only rows (SHA-256 `95bc6e2f9478a929765f02c21a4beb7e187779ba46db02b8609d5f2b81b97d4d`) with fields `uid`, `speaker_id`, `accent`, `split`, `duration`, `src_sr`, `source_file`, and `source_row`. No reference or audio field was opened.
- Its [split map](/Users/prateekraj/whispr-nano/data/numeric_asr/spire_20260917/splits.json) has SHA-256 `3d0404574b1e886989211be9b4e52782f0be644fa1e7083e31368aeefa5bb991`. Checked counts are 29,796 train rows / 1,126 train speakers, 5,214 validation rows / 198 validation speakers, and 289 unassigned rows / 87 speakers. The train and validation speaker-ID sets have zero intersection.
- Development-side source-rate metadata is 29,009 rows at 48 kHz, 261 at 44.1 kHz, and 526 at 16 kHz. Train durations range from 1.08 to 30.0 seconds across 16 accent labels. These are source metadata values, not a waveform inspection.
- The local preparation receipt pins the source as `VectorSigma389/spire-sies` revision `02af0c4d65a2f7f22784154caa1a5aa22128dca0`; it records audio columns were not projected for the inventory. Its source-license terms are not recorded in the checked local files.
- [research/numeric_asr/prepare_spire.py](/Users/prateekraj/whispr-nano/research/numeric_asr/prepare_spire.py) explicitly reads only metadata first, filters references to `split == train` before materializing them, and assigns `audio_path: null`. The local full training manifest is not inspected here. [research/numeric_asr/candidates.py](/Users/prateekraj/whispr-nano/research/numeric_asr/candidates.py) also rejects validation/test/sealed split labels and every dataset name containing `SPIRE198speakers` before audio is opened.

## Older claims, retained only as provenance

[paper/metadata/source_evidence.json](../../paper/metadata/source_evidence.json) records a prior first-party/API check that Svarah revision `ebbf…` was a three-Parquet-file, 6,656-row test split, with a 1,094,998,590-byte download estimate and a CC BY 4.0 card statement. It also records that cached recovered metadata lacked a speaker column. I did not re-query Hugging Face or accept a gate, so these are historical evidence rather than newly verified access or license facts.

[research/publication/numeric_dataset_inventory_20260917.md](/Users/prateekraj/whispr-nano/research/publication/numeric_dataset_inventory_20260917.md) records an earlier read-only control-plane listing of the remote Modal volume `ast-asr-cache:/ai4bharat___svarah/default/0.0.0/ebbf7777fe771490696a3f7b007097606fa8c924`: three Arrow shards of 495.0 MiB, 480.5 MiB, and 82.5 MiB plus `dataset_info.json` (about 1.03 GiB). The initial inventory did not find `modal` on PATH. Subsequent execution located it at `/Users/prateekraj/whispr-nano/.venv/bin/modal`, rechecked the volume listing, and attempted retrieval. Only the complete third shard was recovered; incomplete first/second shards were excluded.

[paper/metadata/README.md](../../paper/metadata/README.md) reports that the Svarah paper describes 117 speakers, but it also documents the missing filename-to-speaker mapping. The 117 figure must not be converted into per-row speaker labels or used for speaker-cluster intervals.

## Initial route proposals (before source recovery)

1. **Preferred: SPIRE development-only acquisition.** With authorized source access, verify the source repository revision and license first. Then use the existing metadata-only admission predicate to select a small deterministic train-only development slice, download only its audio bytes, resample each verified waveform to a declared 16 kHz mono evaluation form, and record source/audio/reference hashes. Keep all 198 validation speakers sealed. This route has real corpus-provided speaker IDs but is not ready until its license/source release is reconciled.
2. **Svarah frozen-system benchmark or exploratory slice.** Mount the existing remote Arrow cache read-only; do not re-download the 1.03 GiB dataset. A benchmark may reuse the known 2,794-ID exposed union only if its reporting calls that pre-exposed evaluation. A newly selected subset must subtract that union and be labeled *utterance-disjoint exploratory*, never speaker-disjoint; retain only sentence/recording-tag sensitivity analyses.

## Remaining limitations and resolved blockers

- Svarah lacks verified stable speaker IDs for row-level resampling or speaker-disjoint selection.
- Resolved for the recovered third shard: original audio hashes, references and mono 16 kHz source waveforms were independently inspected for all 518 records. Source audio for the other two shards remains unavailable after transfer errors.
- The prior Svarah records are exposed; a new subset requires an explicit ID-level exposure ledger before references or audio are staged.
- SPIRE has promising corpus-provided speaker IDs, but its exact authorized release, current terms, and permitted use need reconciliation before staging. Its final 198-speaker validation cohort must remain unopened until the evaluation protocol is frozen.
