# Reproduce the finite core audit

Start with `QUALITY_RESULTS.md`, `RESOURCE_RESULTS.md` and `completion_matrix.json`. The manuscript is `core_audit.tex` / `core_audit.pdf` at the repository root. Raw run directories are under `research/bridge_open/artifacts/core_*`; the preceding accepted FLEURS and Svarah run directories remain unchanged.

Each remote job retains its committed source snapshot, provider/source receipt, configuration, model receipts and raw predictions. The receipt's `sourceDigest` identifies the executed snapshot. Do not substitute a later checkout for the code that ran. The failed Qwen Hindi configuration attempt is retained separately from its accepted rerun.

## Local validation

From `/Users/prateekraj/whispr-nano`, use the existing `.venv/bin/python` environment. Validators require the frozen original references and audio assets; a copied manuscript is not the full reproducibility package.

```sh
.venv/bin/python -O -m research.bridge_open.core_study.accept_qwen06 research/bridge_open/artifacts/core_qwen06_hindi_v1 --track hindi
.venv/bin/python -O -m research.bridge_open.core_study.accept_qwen06 research/bridge_open/artifacts/core_qwen06_svarah_v1 --track svarah
.venv/bin/python -O -m research.bridge_open.core_study.accept_whisper_core research/bridge_open/artifacts/core_whisper_small_hi_v1
.venv/bin/python -O -m research.bridge_open.core_study.summarize_quality
```

Apply the Whisper validator separately to each completed core Whisper directory; partial or unaccepted cells must remain pending. `accept_resources.py` independently checks both resource run directories. See its CLI help for the two positional paths. The summary generators overwrite only derived outputs, not raw predictions or executed source archives.

Source manifests:

- FLEURS: `data/bridge_open/fleurs_test_v1/corpus-v2.jsonl`, SHA-256 `967d8dca02f9f729c1ed7548dce7283cf9dfc62623815a7c131f71acb13ce2c7`.
- Full available Svarah shard: `data/bridge_open/svarah_native_shard02_full_v1/bundle/manifest.jsonl`, SHA-256 `dbddbaeef3a1f78ff85d0cd741e85d86daf7c868b39ee8d4c48f67a21bfafedd`.
- Resource panel: `data/bridge_open/svarah_native_shard02_v2/bundle/manifest.jsonl`, SHA-256 `14c08d00b5e585de77ee3af6b486b05980e0cc9e8a31139654437b2ed8082dac`.

FLEURS manifest source paths refer to the original local dataset location. Svarah bundles resolve reference filenames under their `references/` directories. Relocation requires preserving source bytes and fixing path resolution explicitly; replacing an unavailable asset silently would invalidate the comparison.

## Review and inspection

Serve the repository root, not only a review subdirectory:

```sh
.venv/bin/python -m http.server 8768 --bind 127.0.0.1
```

Open `/research/bridge_open/listening_review_v2/index.html` for the frozen FLEURS review or `/research/bridge_open/listening_review_core_en_v1/index.html` for the separate English resource panel. The evidence viewer is `/research/bridge_open/artifacts/core_study_v1/index.html`. These pages use verified local audio paths. Do not distribute sealed identity mappings to blinded reviewers. Import only actual exports; synthetic tests are not review evidence.

## Compute and preservation

Every GPU run used the fixed ORX command `bash experiments/hf_entrypoint.sh`, an immutable committed condition, HF A100-large and a 30-minute provider timeout. These were bounded job reservations, not measured provider charges. No additional job is launched by the local validators or report generators.

The selected evidence snapshot includes an inventory of file SHA-256 hashes and sizes at `artifacts/core_study_v1/evidence_manifest.json`. Original dataset audio remains at its source paths; portable executed archives retain the committed input definitions. Human review and submission are separate completion gates from GPU coverage.
