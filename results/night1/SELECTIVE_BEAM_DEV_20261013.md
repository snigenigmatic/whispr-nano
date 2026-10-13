# Selective five-beam fallback: development control

**Status, 13 October 2026:** completed exploratory comparison on the existing 104-clip, 26-speaker Lahaja development sample. This is not a held-out intervention result. The detector was chosen after this cohort had been examined for loops.

The proposed inference policy checks only the greedy Whisper-small hypothesis: if it contains at least five identical adjacent normalized words, replace the whole hypothesis with a five-beam decode of the same audio; otherwise keep the greedy output. References enter scoring, not the trigger. The pinned dataset revision, model revision, Hindi transcription setting, and word normalizer match the prior runs.

| Decoder or policy | Word errors / 1,992 reference words | WER |
| --- | ---: | ---: |
| Greedy | 2,403 | 120.63% |
| Full five-beam | 1,583 | 79.47% |
| Text-only loop repair | 1,458 | 73.19% |
| Selective five-beam fallback | 1,660 | 83.33% |

The fallback triggered on **25/104 clips (24.04%)**. Relative to greedy, it helped 21 triggered clips and harmed 2. Its paired, speaker-bootstrap improvement over greedy was 37.30 WER points (95% interval 22.56–53.56 points). The fallback was **3.87 points worse than full five-beam** and **10.14 points worse than text-only repair** on this development cohort. The paired interval for fallback minus full five-beam was −2.18 to +9.73 points. No runtime or battery saving was measured, and there is no genuine-repetition safety test. This control does not justify a publication claim or a recommendation to deploy the fallback.

The ORX run is `fd8001a8-f598-4614-8699-b6e6d0dfe5cd` at commit `0a95125` on `orx/lahaja-development-selective-beam-fallback`. The local aggregate receipt at `~/.local/share/openresearch/lahaja-selective-beam-dev-v1/summary.json` records source and output SHA-256 hashes; gated row-level predictions remain local. A separate 405-clip held-out *full five-beam* control is running as ORX experiment `32871ad9-02cf-47ed-996e-6910a532c192`, run `36217bd3-5caa-48ea-8521-0660ca15d7f0`. Its result must be inspected and packaged before it is quoted.
