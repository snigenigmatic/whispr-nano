> **Audio QC correction (2026-09-22):** Five public files contain transcript text over exactly silent audio. Affected Assamese/Bengali quality comparisons and aggregates containing them are withdrawn pending adjudication. Original metrics below are preserved as historical records. See the [source-verified audit](artifacts/audio_content_audit/RESULTS.md).

# Hugging Face GPU execution

C. Kaustubh authorized a separate $50 HF compute budget on 2026-09-22. Modal retains its existing $15 cap. ORX verified the configured HF account and Jobs write permission.

A portable ORX project was created at `/Users/prateekraj/whispr-bridge-hf` because the existing immutable experiment command contains a Mac-specific Python path. The old project and all completed nodes remain unchanged.

- HF ORX project: f8f80981-f6a4-4f34-af55-f292bfdd8708
- Baseline experiment: a142465c-b45c-43a8-9929-1aae62c38fe2
- Fixed command: `bash experiments/hf_entrypoint.sh`
- Image: `pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime`
- GPU: one A100 80 GB, 30-minute timeout
- Reserve: $1.50 per dispatch, cumulative $50 ceiling, no release based on runtime estimates
- GPU list price: $0.041667/minute (official Jobs hardware API, checked 2026-09-22)

First dispatch 21d2ee9d-9675-484f-9a57-9dfd6a0c0f32 created HF job6ab2909a52d0dbd7f1d7efa0, although ORX reported submission failure. The HF SDK printed sync status before its JSON response. The local source-transfer SDK now passes supported `quiet=True`; original file and patch hashes are preserved in HF_PORTABILITY_REPAIR.json.

That job stopped at the audio file hash check before model inference. Regenerated FLOAT WAV files have timestamp-bearing PEAK metadata. The corrected runner transfers the exact frozen WAV/reference bytes, preserving the original hashes and manifest. The failed attempt remains charged/reserved in the ledger.

Corrected dispatch 735b3a94-b034-44e9-8fca-d2d7d6c553db / job6ab2914852d0dbd7f1d7efcb completed. Complete artifacts were recovered and independently validated. Transcript, selected raw token sequences and all metrics exactly match the Modal H100 result for this single exposed Bengali conversation (WER 98.2759%, CER 76.7348%). This is a portability check, not proof of universal hardware equivalence.

The child fallback experiment b01b8562-1615-436d-b347-68d14f6075a9 changes only the committed temperature schedule and compression threshold (plus identifying metadata). Run c1bed397-0d94-45c4-a0ad-ba4793c0821a / job6ab2921351992417dfcd366a completed and independently validated. Fallback worsened WER from 98.2759% to 99.8331% and CER from 76.7348% to 89.5141%; inference time rose from 95.0 to 271.9 seconds. Selected capped calls fell from 25/31 to 2/31, demonstrating that fewer cap flags do not imply better transcripts. There were 176 attempts and 24/31 windows exhausted the schedule. The implementation used one beam after the first sampling transition (175/176 attempts), recorded explicitly in the trace. This recipe will not be scaled up. The next child 6c158ead-dcb3-4b56-8e9b-e38f2b97f679 replaces only Turbo with full Large-v3 under the validated no-fallback recipe. Run e0c595d5-bfae-496a-a4d5-6a6503be97aa / job6ab293c852d0dbd7f1d7f04c completed. Raw WER 97.4972%, CER 77.6333%; model-quality interpretation is withdrawn following the audio QC finding above. All four HF jobs are terminal; no study HF GPU remains running. All four dispatches retain their full $1.50 reservations: $6.00 held of the $50 HF ceiling, not $6.00 verified spending.
