# Svarah fairness audit — six systems, six structural gaps

This is the canonical home for the PW25_BJD_05 Phase 3 pivot's audit
deliverable: **"Whose Indian accent? A significance-tested subgroup audit of
Indian and frontier ASR systems."** It started life reusing the `ast-asr`
capstone repo's own scoring convention (`ast_asr.taxonomy.SVARAH_LANGUAGE_FAMILIES`,
`ast_asr.metrics.normalize_for_wer` / `word_edit_counts`) so its numbers stay
comparable to every other Svarah number in that project — but the results,
scripts, assembler, and verification below live **here**, in `whispr-nano`,
which is the repo this project actually pushes to and presents from. (An
identical copy was also committed to `ast-asr`'s `fair-cispo-work` branch,
locally, but that repo has had push-credential issues in this environment;
treat *this* copy as authoritative.)

`assemble.py` (below) is the independent verification layer: it reglobs
every `results_*_svarah_clean.csv`, hard-validates the schema, **recomputes
every number in the table below directly from the per-utterance rows**, and
cross-checks the result against each `summary_*.json`. Run it yourself:

```bash
cd audit/svarah_fairness_audit
python assemble.py                # writes master_audit_table.{csv,md}
python -m pytest test_assemble.py -v   # 9 tests: schema drift + math regression
```

## Results: every system tested shows the same structural pattern

| System | Overall WER | ΔDP | Poisson p | Verdict | n | Cost |
|---|---|---|---|---|---|---|
| whisper-tiny | 20.60% | 14.19pp | 3.3e-24 | structural | 2500 | $0.034 |
| whisper-small | 10.88% | 3.18pp | 0.0045 | structural | 2500 | $0.095 |
| whisper-large-v3 | 7.04% | 2.38pp | 0.0152 | structural | 2500 | $0.411 |
| distil-large-v3 | 10.03% | 4.02pp | 0.00025 | structural | 2500 | $0.285 |
| **Saaras V3** | **5.81%** (best overall) | 3.45pp | 0.00023 | structural | 450 | $0.226 |
| Qwen3-ASR-0.6B | 15.49% | 3.57pp | 0.0013 | structural | 2500 | $0.142 |

**Six systems, six statistically significant per-family gaps, and
Sino-Tibetan is the worst-performing family in every single one** — three
sizes of open Whisper, a distilled Whisper variant, Sarvam's own flagship
Saaras V3 (the model explicitly marketed as solving exactly this problem),
and Alibaba's Qwen3-ASR (state-of-the-art on Western ASR leaderboards, but
notably *worse* overall on this Indian-accented set than several Whisper
variants — a second finding worth a slide on its own: general-benchmark SOTA
does not predict Indian-accent performance). Saaras V3 has the best overall
WER of anything tested (5.81%) but does not close the fairness gap — direct,
real-system evidence for "aggregate WER hides subgroup movement."

Total cost across all six systems: **$1.193**.

### Qwen3-ASR notes

- `Qwen/Qwen3-ASR-0.6B-hf`, native Transformers API
  (`processor.apply_transcription_request`), language forced to `"English"`
  per clip (Svarah is Indian-*accented English*, not a different language;
  forcing avoids any language-ID misfire).
- One systematic, expected scoring quirk: Qwen3-ASR tends to write numbers
  as words ("fifty") where Whisper/Saaras more often keep digits ("50") —
  this is scored as-is (no per-vendor digit normalization), consistent with
  every other system in this table, but worth a one-line caveat if this
  number gets quoted.

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

## Files in this folder

- `modal_whisper_family_audit.py` / `modal_qwen3_asr_audit.py` /
  `fetch_stratified.py` / `run_saaras_audit.py` — the four scripts that
  produced every row below.
- `results_<system>_svarah_clean.csv` — per-utterance results, one row per
  clip, for all six systems.
- `summary_<system>_svarah.json` — per-system summary (overall/per-family
  WER, ΔDP, Poisson test) as written by the run itself.
- `assemble.py` / `test_assemble.py` — the independent assembler + its
  pytest suite (schema hard-fail, math regression, summary cross-check).
- `master_audit_table.csv` / `.md` — the assembler's output; regenerate
  any time with `python assemble.py`.
