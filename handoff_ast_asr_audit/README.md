# Handoff: Svarah audit results → belongs in `ast-asr`, not here

These files were produced running against the `ast-asr` repo's own code
(`ast_asr.taxonomy`, `ast_asr.metrics`) on Modal + the Sarvam API. They're
staged here only because this is the one repo this session has write
access to — **this whole folder should end up in `ast-asr`'s `audit/`
directory, not in whispr-nano.**

## To install into your `ast-asr` clone

```bash
cp handoff_ast_asr_audit/*.py    <ast-asr>/audit/
cp handoff_ast_asr_audit/*.csv   <ast-asr>/audit/
cp handoff_ast_asr_audit/*.json  <ast-asr>/audit/
cd <ast-asr>
git add audit/
git commit -m "Add Whisper-family + Saaras V3 Svarah audit results"
git push
```

## Results: every system tested shows the same structural pattern

| System | Overall WER | ΔDP | Poisson p | Verdict | n | Cost |
|---|---|---|---|---|---|---|
| whisper-tiny | 20.60% | 14.19pp | 3.3e-24 | structural | 2500 | $0.034 |
| whisper-small | 10.88% | 3.18pp | 0.0045 | structural | 2500 | $0.095 |
| whisper-large-v3 | 7.04% | 2.38pp | 0.0152 | structural | 2500 | $0.411 |
| distil-large-v3 | 10.03% | 4.02pp | 0.00025 | structural | 2500 | $0.285 |
| **Saaras V3** | **5.81%** (best overall) | 3.45pp | 0.00023 | structural | 450 | $0.226 |

**Every single system — every open Whisper variant and Sarvam's own flagship
Saaras V3 — shows a statistically significant per-family WER gap on Svarah,
and Sino-Tibetan is the worst-performing family in every single case,
including in Saaras V3, the model explicitly marketed as solving exactly
this problem.** Saaras V3 has the best overall WER of anything tested (5.81%)
but does not close the fairness gap — direct, real-system evidence for
"aggregate WER hides subgroup movement."

Total cost across all five systems: **$1.051**.

## Notes on the Saaras V3 run specifically

- 450 clips, stratified 150/family (not the full ~2500 the Whisper systems
  used) — chosen for time/rate-limit reasons; still gives solid per-family
  power, especially for the otherwise-thin Sino-Tibetan group.
- Scored with the exact same `ast_asr.metrics.normalize_for_wer` /
  `word_edit_counts` / `ast_asr.taxonomy.SVARAH_LANGUAGE_FAMILIES` convention
  as the Whisper-family systems, so numbers are directly comparable.
- Raw Saaras API responses are plain, well-punctuated, well-capitalized
  transcripts (`{"transcript": "...", "language_code": "en-IN"}`) — no
  extraction step was needed.
- Reproduction: `fetch_stratified.py` (Modal, fetches audio + fills
  `audit/saaras_audit/manifest.json`) then `run_saaras_audit.py` (plain
  Python, needs `SARVAM_API_KEY` in the environment, calls the Saaras REST
  API directly, caches every response to disk so a re-run after any
  interruption costs nothing).
- **Do not commit the `saaras_audit/response_cache/` or `saaras_audit/wavs/`
  directories** if you re-run this — they contain raw audio and cached API
  responses, not something that needs to live in git history. Only the
  `results_saarasv3_svarah_clean.csv` / `summary_saarasv3_svarah.json` in
  this folder are meant to be committed.

## Reproducing the Whisper-family systems

Script (`modal_whisper_family_audit.py`) is the exact one already validated
and described in `docs/TEAM_AGENT_TASKS_20260828.md` in this repo.
