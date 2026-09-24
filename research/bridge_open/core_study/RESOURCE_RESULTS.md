# Matched resource-panel results

All six systems completed the fixed 12-item (144 reference words) Svarah panel on an A100-SXM4-80GB. One warmup and three measured passes; quality below uses pass 3 only. No failed calls.

Every row is the fixed batch-one resource condition. Qwen's primary full-quality condition uses batch eight, so its full-batch WER and these batch-one timings are not a matched latency/quality Pareto measurement. Measured provider billing is unknown: elapsed time and memory observations do not establish a bill. This study contains no phone, edge, or mobile-device measurement or claim.

| Recipe | Panel WER | Median pass corpus RTF | Peak allocated GiB | Peak reserved GiB |
|---|---:|---:|---:|---:|
| Whisper Small | 9.03% | 0.03615 | 1.494 | 1.678 |
| Whisper Turbo | 12.50% | 0.04040 | 3.491 | 3.727 |
| Whisper Large-v3 | 15.97% | 0.10947 | 8.346 | 8.750 |
| Qwen 0.6B | 26.39% | 0.08026 | 1.678 | 1.828 |
| Qwen 1.7B | 27.08% | 0.09862 | 4.018 | 4.449 |
| Omni CTC 1B | 62.50% | 0.00377 | 2.449 | 2.730 |

Whisper Small is favorable on this particular small panel, but this is not evidence that it beats the larger models on the full tracks. The completed 518-record Svarah comparison reverses Small versus Large-v3: 9.69% versus 8.24% WER, whereas this panel favors Small. All three Whisper recipes have identical raw transcripts between primary and resource runs on the exact panel IDs (`primary_panel_parity.json`). The observed ordering therefore depends on the evaluated subset; it is not an effect of batch-dependent Whisper transcript changes on these twelve items. Omni has the smallest measured RTF but high panel WER. Precision and search remain model-specific.

The Qwen batch-one profile outputs differ from primary batch-eight output on 1/12 clips (0.6B) and 5/12 clips (1.7B). Therefore the resource table reports its own batch-one panel quality; attaching the primary full-batch-eight WER to these timings as one matched measurement would be wrong.

Original arrays were loaded before timing; preprocessing, transfer, decoding and text postprocessing were inside the synchronized clock. Cold start and audio I/O were excluded. Peak allocated and reserved memory are different allocator quantities.

Warmup records lack individual audio hashes. Their source binding is limited to the verified shared preloaded panel, ordered IDs/durations, and archived harness; every timed call carries the original audio hash. This qualification is retained in resource_results.json.

Evidence: core_resource_omni_v1 (run ae6ce853-45ab-42e3-8f57-ce5aea1a7f1b) and core_resource_whisper_qwen_v1 (run ab478b7f-56cf-4629-8c0e-04e0f8c82511). Independent acceptance is in resource_results.json.

## Timing repetitions

These are repeat measurements on the same 12 inputs, not independent samples or statistical confidence intervals.

| Recipe | Pass 1 RTF | Pass 2 RTF | Pass 3 RTF | Observed range |
|---|---:|---:|---:|---:|
| Whisper Small | 0.035137 | 0.036151 | 0.041427 | 0.035137–0.041427 |
| Whisper Turbo | 0.040404 | 0.040475 | 0.040387 | 0.040387–0.040475 |
| Whisper Large-v3 | 0.109515 | 0.109468 | 0.109404 | 0.109404–0.109515 |
| Qwen 0.6B | 0.080223 | 0.080625 | 0.080262 | 0.080223–0.080625 |
| Qwen 1.7B | 0.090079 | 0.106911 | 0.098622 | 0.090079–0.106911 |
| Omni CTC 1B | 0.003723 | 0.003766 | 0.003824 | 0.003723–0.003824 |
