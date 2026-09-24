# Core audit: claim-to-evidence map

This map applies to the 24 September 2026 working manuscript, `core_audit.tex` at the repository root. The live coverage authority is `completion_matrix.json`. A completed GPU job alone is not an accepted result.

| Claim | Evidence and acceptance | Scope |
|---|---|---|
| Corpus error differs across the six declared inference recipes | `quality_results.json`, `QUALITY_RESULTS.md`, per-cell prediction hashes and independent acceptance receipts; `summarize_quality.py` rejoins exact source IDs and re-scores the frozen reference bytes | Four separately reported tracks; no pooled winner or causal architecture comparison |
| Smaller Qwen has more errors on the measured Hindi and Svarah cohorts | `core_qwen06_hindi_v1` and `core_qwen06_svarah_v1`, compared with the accepted 1.7B predictions on identical recordings | Size comparison under native BF16 batch-eight inference; no adaptation gain |
| Whisper Small uses less allocated GPU memory than Large-v3 on the measured panel | `resource_results.json`, both `core_resource_*` run directories, raw synchronized timed calls and `accept_resources.py` | Same 12 pre-exposed recordings, batch one, A100 80GB class; not phone feasibility or full-corpus quality |
| Qwen batching changes some outputs | Resource-panel pass-three texts compared with primary batch-eight predictions by exact audio ID; `RESOURCE_RESULTS.md` | Batch-one quality remains attached to batch-one timing; no blending of conditions |
| Some outputs contain non-Latin characters or have WER at least 100% | `svarah_error_diagnostics_v1`, per-cell tails in `quality_results.json`, original retained transcripts | Unicode and edit-count observations; neither language identification nor a semantic judgment |
| Small's Bengali and Assamese recipes produce severe repetition and termination failures | `core_whisper_small_bn_v1` and `core_whisper_small_as_v1` source/weight/frontend acceptance, retained output IDs, insertions and observed-EOS fields in `quality_results.json`; `whisper_small_native_spotchecks.json` | Fixed forced-language beam-five recipe with a 440-new-token short-form limit; three native spotchecks per language do not establish full-cohort native/HF equivalence or optimal decoding |
| The failed Qwen Hindi dispatch did not yield a usable baseline | Failed run `61219347-d8a4-438c-9cbc-9d0058d096b4`, retained in `core_qwen06_hindi_failed_v1`; corrected run `04965a3b-7e62-46f4-a6ee-e361b18eca20` and its accepted source snapshot | Native-API configuration repair before usable predictions; failure is retained, not hidden |
| Review materials are ready | Frozen FLEURS packet: 50 clips / 168 cards; separate English resource-panel packet: 12 clips / 72 cards; validated importers | All actual ratings remain pending. Packet construction and test fixtures are not completed human review |
| This is narrower than prior Indian-ASR benchmark work | `LITERATURE_POSITIONING.md`, full-text reading of Vimarsha, OpenWER and Responsible ASR | No first/comprehensive benchmark, new metric, demographic fairness or new recognition-method claim |

Publication readiness still depends on actual review, a clear practitioner finding and venue fit. These engineering checks do not guarantee acceptance. Source-model training overlap with public evaluation data has not been ruled out.
