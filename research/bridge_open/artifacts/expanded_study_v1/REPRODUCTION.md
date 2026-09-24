# Evidence handoff

Start with `RESULTS.md` and `study_summary.json`. The five new run directories retain predictions, configuration, environment evidence, provider/source receipts and the exact portable source archive. The archive in each receipt is the executed source; later preparation files are not a substitute for it.

`evidence_manifest.json` lists the SHA-256 and byte size of every file selected for this expansion, using repository-relative paths. It excludes itself to avoid a circular hash. The preceding accepted benchmark is preserved by commit `cf75c36` on `codex/native-hf-evidence-20260922` in `/Users/prateekraj/whispr-bridge-pilot`.

Local independent validation requires the original, hash-matched dataset. Its frozen manifest is `data/bridge_open/fleurs_test_v1/corpus-v2.jsonl`, with SHA-256 `967d8dca02f9f729c1ed7548dce7283cf9dfc62623815a7c131f71acb13ce2c7`. Recorded source paths point into `/Users/prateekraj/whispr-nano`. Raw dataset audio is not duplicated into this evidence commit. The source archives contain the committed acquisition/staging workflow and frozen input definitions.

From `/Users/prateekraj/whispr-nano`, use the existing `.venv/bin/python` environment:

```sh
.venv/bin/python -m research.bridge_open.analyze_omni_extension /tmp/omni-recheck research/bridge_open/artifacts/omni_fleurs_full_hi_v1 research/bridge_open/artifacts/omni_fleurs_full_bn_v1 research/bridge_open/artifacts/omni_fleurs_full_as_v1
.venv/bin/python -m research.bridge_open.check_extension_negative_cases
.venv/bin/python -m research.bridge_open.plot_expanded_benchmark
```

The comparison destination must not already exist. Rejection checks intentionally corrupt temporary copies and preserve raw evidence. Plot generation replaces only generated figures.

Serve the main workspace to use the audio viewers:

```sh
.venv/bin/python -m http.server 8768 --bind 127.0.0.1
```

Open `/research/bridge_open/artifacts/benchmark_review_v2/index.html` or `/research/bridge_open/listening_review/index.html`. A copied viewer includes transcripts but needs the dataset paths for audio. The listening packet is unreviewed and contains the prior Whisper/Qwen outputs; keep its `sealed_mapping.json` away from reviewers until annotations are frozen. Browser annotations must be exported before handing off the work.

All five expansion jobs are complete. The ledger receipt records conservative reservations, not measured provider charges. No publication claim, human transcript judgment, new trained method, or demographic inference follows automatically from these benchmark scores.
