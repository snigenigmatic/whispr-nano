# Full-waveform FLEURS OmniASR CTC extension

Completed evidence: the nine-item acceptance and all three full language cohorts passed. The executed configurations and source archives in `../artifacts/omni_fleurs_*_v1/` are authoritative. The four runtime modules in this directory match the accepted full-run source. Other entrypoints and configuration scaffolding here describe preparation; the jobs used separate frozen ORX branches, each with a fixed command and configuration. See `../EXPANDED_STUDY_20260923.md` for run links and results.

This directory prepares the pinned official OmniASR CTC pipeline for all 2,322 frozen FLEURS v2 recordings. It uses `facebook/omniASR-CTC-1B` revision `8c22e3ffdaa4aab6431b128b84b991a7d9c2515c`, preserving the verified model and tokenizer asset hashes from the previously successful public Humyn pilot.

Unlike the earlier fixed-30-second pilot, every FLEURS recording is passed to the official pipeline as one full mono 16 kHz waveform. The longest frozen input is 34.56 seconds, within the native 40-second cap. Each artifact has one complete-audio chunk with absolute `[0, duration]` bounds; no clipping, overlap, or 30-second split is permitted.

`acceptance_config.json` fixes a deterministic nine-recording public-API/retained-token check: two sorted recordings and the longest waveform for Assamese, Bengali, and Hindi. The retained argmax IDs and repeat-collapsed IDs are recorded, and the official `transcribe()` string must match the retained-token decode for every acceptance item. It contains the 33.96 s Assamese, 34.56 s Bengali, and 31.44 s Hindi examples.

`full_config.json` uses the same code path and all 2,322 manifest IDs, but avoids a duplicate public API forward after the acceptance cohort has passed. Both configurations hash every selected audio and reference source before model work; reference text is opened only after its waveform's inference result is complete and is then independently rescored.

The HF entrypoint pins torch/torchaudio 2.8.0, fairseq2 0.6.0/fairseq2n 0.6, NumPy 1.26.4, SciPy 1.15.3, SoundFile 0.13.1, JiWER 4.0.0, Hugging Face Hub 0.36.2, and official `omnilingual-asr` source revision `81f51e224ce9e74b02cc2a3eaf21b2d91d743455`. It requires the frozen FLEURS manifest paths to be mounted at their recorded absolute locations. No cost is inferred from elapsed time.

Limits: CTC outputs have no EOS or generation-length semantics. The preserved evidence is raw argmax IDs and adjacent-repeat collapse before the official tokenizer decoder. Public API agreement establishes only this deterministic implementation equivalence; it does not establish transcript correctness, language fairness, or generalization.
