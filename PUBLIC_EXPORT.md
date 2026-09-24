# What this GitHub branch contains

This is a public export of the final tracked study tree at local evidence commit **7aa169b**, based on the existing GitHub `main` commit `351bd993a37ff681150a384fdedc02e515f2c952`. The original local evidence branch remains unchanged. No force-push, rewrite of the original evidence history, or merge into `main` is involved.

Included: experiment and analysis code, executed text/code copies, configurations, model/data/code digests, complete retained predictions, metrics, resource traces, acceptance receipts, failed-run evidence, review packets/importers, the manuscript, and the owner task checklist.

## Deliberate exclusions

- **24 original `source_snapshot.tar.gz` files** remain local. Many embed original dataset audio; six also exceed GitHub's ordinary per-file limit. This export does not republish that audio.
- **Four `sealed_mapping.json` files** remain local to support blinded human review. Share isolated packets and their authorized audio with reviewers, not this repository or result tables before rating.
- The three frozen text manifests and twelve resource-panel reference JSON files are included with their original hashes. Other original audio/reference asset directories and model weights are not included, and a fresh clone does not download them automatically. The pre-existing practice audio on `main` is unchanged.

[The public export manifest](research/bridge_open/artifacts/core_study_v1/public_export_manifest.json) lists every omitted file with its original SHA-256 and size. It also records each derived archive's digest, included files and excluded binary entries.

## Source-code archive copies

Each original archive has a **`source_code_snapshot.tar.gz`** alongside its original receipt. These derived archives contain the original code/text bytes, including input definitions and references, while excluding binary audio. They are clearly named derivatives and **are not the exact executed source snapshot**. Original execution digests, acceptance receipts and result hashes have not been rewritten to make them appear interchangeable.

The original local inventory, `evidence_manifest.json`, describes the original evidence snapshot, including excluded files. It is not a promise that every original file is in this public branch. Use the public export manifest to understand the distribution boundary.

## Reproduction and review

The retained metric tables and raw predictions are inspectable in a clone. Full source-byte acceptance, listening playback and GPU replay additionally require the corresponding original audio assets, source path resolution and, for archive identity checks, the original execution archives. See [the reproduction guide](research/bridge_open/core_study/REPRODUCTION.md) for pinned manifests and hashes. Missing assets must not be silently replaced or accepted.

The authoritative original archive set remains at `/Users/prateekraj/whispr-bridge-pilot` on `codex/native-hf-evidence-20260922`, commit `7aa169b`. Keep that local evidence safe. If a future public data release is planned, handle its attribution, permitted redistribution and packaging explicitly; this push is a code/results publication.

Reviewer packet HTML must be served with its verified audio. All actual review ratings remain pending. A public export is neither a completed human review nor a paper submission.

## Checks in a clone without the original assets

The original complete local study passed **52 focused tests**. A public clone can run **50** of them without the original audio and exact execution archives. The two integration tests `test_real_completed_svarah` and `test_preload_reads_the_verified_bundle_not_the_source_relative_path` deliberately require those assets and fail if they are absent; their checks have not been weakened. The public-package validation receipt records the separate outcome.

Use the same eight test paths listed in `artifacts/core_study_v1/validation_receipt.json` (relative to `research/bridge_open`), with this explicit selection when working without original assets:

```sh
python -m pytest -q research/bridge_open/core_study/test_accept_whisper_core.py research/bridge_open/core_study/test_accept_qwen06.py research/bridge_open/core_study/test_summarize_quality.py tests/test_core_review_import.py tests/test_core_resource_profile.py tests/test_core_resources_acceptance.py tests/test_resource_results_provenance.py research/bridge_open/listening_review_core_en_v1/test_review_import.py -k 'not test_real_completed_svarah and not test_preload_reads_the_verified_bundle_not_the_source_relative_path'
```

This is a code/fixture test selection, not a new acceptance of a complete experiment from missing inputs. To run full source-byte validation, restore the original assets and run all tests without deselection.
