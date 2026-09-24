# Continuation evidence snapshot

This package adds documentation, post-hoc FLEURS error-tail counts, the unreviewed blinded listening packet, and four independently accepted native Svarah executions (two implementation checks and two 518-record comparisons) to evidence commit `1468f7b42a75c91254823fe0dc7a22999de748e8`.

[File manifest](manifest.json) gives repository-relative paths and SHA-256 values. Source archives for all four accepted jobs contain the exact executed configs, runtime code and committed WAV/reference bundles. Model weights are represented by pinned upstream revisions and observed asset hashes, not duplicated model files. The cancelled Qwen dispatch is preserved as a status/reason receipt, not an accepted result.

Start with [current status](../../CURRENT_STATUS.md), [Svarah results](../svarah_comparison_v1/RESULTS.md), and [execution/source correction record](../../SVARAH_EXECUTION_20260923.md). The 12-record cohorts overlap the 518 and must not be double-counted. The complete FLEURS comparison remains in the earlier snapshot.

## Local reproduction

Use the existing project Python environment. No GPU is required to rescore accepted predictions. The original source shard is retained at `data/bridge_open/svarah_cache_v1/svarah-test-00002-of-00003.arrow` in the main workspace; its hash is pinned by the input auditor. It is an external local prerequisite and is not copied into this evidence commit. The main workspace also retains the prepared native bundles. The exact selected WAVs/references are additionally preserved inside each executed source archive. This package does not claim a public redistribution release or public source access.

From the main workspace root:

```sh
.venv/bin/python -m research.bridge_open.analyze_failure_tails
.venv/bin/python -m research.bridge_open.analyze_svarah
.venv/bin/python -m research.bridge_open.check_svarah_evidence
.venv/bin/python -m research.bridge_open.analyze_svarah_errors
```

The Svarah analysis reruns source/receipt/model/code acceptance and independent scoring before writing its table. It requires both source cache and prepared full bundle, which the source-preparation script can reconstruct from the complete cached shard and pinned metadata. The accepted 12-item source bundles were prepared before full selection; their manifests remain unchanged.

[Input checks](../svarah_evidence_checks_v1/input_negative_checks.json) retain the explicit source-corruption rejection outcomes. The evidence-check script validates that corrupted summaries, uncompleted jobs, escaping archive paths and changed runner hashes are rejected under Python optimization. Runtime receipt guards also reject stale 12-item cohort receipts when staging the full 518-record cohort.

The [blinded packet](../../listening_review_v2/README.md) is ready for fluent reviewers. Its 50 recordings and 168 transcript cards have no completed human judgments. Use the main-workspace local HTTP server so linked audio resolves; sealed mapping and metadata are evaluator-only files. Local cached audio and access rights are needed for replay. Source verification does not substitute for human transcript adjudication.

No new model weights, fairness finding, mobile measurement, untouched confirmation or publication acceptance is claimed. Primary scoring rules remain frozen. Provider billing has not been verified.
