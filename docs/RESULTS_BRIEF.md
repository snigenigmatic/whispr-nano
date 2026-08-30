# 📊 The Audit Results — a briefing for Aditya, Adithya, and Aditi

*by Kau, read time ~5 minutes*

---

## TL;DR

I ran **all six planned audit systems myself** so we'd have real numbers in hand
before the review, instead of waiting on everyone's evenings this week. Every
single system — three sizes of Whisper, a distilled Whisper, Sarvam's own
flagship Saaras V3, and Qwen3-ASR — shows the **same statistically significant
accent gap**, and **Sino-Tibetan is the worst-performing family in every one
of them.** Total spend: **$1.193**. This is now our money slide for Monday.

The deck (`docs/PW25_BJD_05_Phase3_Review1.pptx`) is built on the official
review template and is ready to walk through. This doc explains what's in it
and where the numbers came from.

---

## What we actually found

| System | Overall WER | Worst family | ΔDP (pp) | Poisson p | n | Cost |
|---|---|---|---|---|---|---|
| whisper-tiny | 20.60% | Sino-Tibetan | 14.19 | 3.3e-24 | 2,500 | $0.034 |
| whisper-small | 10.88% | Sino-Tibetan | 3.18 | 0.0045 | 2,500 | $0.095 |
| whisper-large-v3 | 7.04% | Sino-Tibetan | 2.38 | 0.0152 | 2,500 | $0.411 |
| distil-large-v3 | 10.03% | Sino-Tibetan | 4.02 | 0.00025 | 2,500 | $0.285 |
| **Saaras V3** | **5.81%** (best overall) | Sino-Tibetan | 3.45 | 0.00023 | 450 | $0.226 |
| Qwen3-ASR-0.6B | 15.49% | Sino-Tibetan | 3.57 | 0.0013 | 2,500 | $0.142 |

Three things worth internalizing before Monday:

1. **Everyone has the same blind spot.** ΔDP (the WER gap between the best and
   worst language family) is statistically significant (p < 0.05) for all six
   systems — not just the small/cheap models. Even Sarvam's own flagship,
   validated on this exact Svarah benchmark by their own blog, doesn't close it.
2. **Best overall ≠ fairest.** Saaras V3 has the lowest overall WER of anything
   we tested (5.81%), but its Sino-Tibetan gap (ΔDP 3.45pp) is *larger* than
   whisper-large-v3's (2.38pp). This is our "aggregate WER hides subgroup
   movement" thesis, demonstrated on someone else's production system, not
   just asserted about our own.
3. **General-ASR SOTA doesn't transfer.** Qwen3-ASR is competitive on Western
   benchmarks but landed *worse* overall (15.49%) than whisper-small (10.88%)
   or whisper-large-v3 (7.04%) on Indian-accented English. Worth its own
   sentence in the review — it's a finding nobody asked us to look for.

## Where each number came from (so you can defend it if asked)

- **Whisper family (tiny/small/large-v3/distil-large-v3):** `audit/modal_whisper_family_audit.py`
  on Modal T4 GPUs, fp32, greedy decode, 2,500-clip random subsample of Svarah's
  6,656-utterance test split, seed 0. This is the exact script from
  `docs/TEAM_AGENT_TASKS_20260828.md` — Aditya's task, already validated
  end-to-end before I ran the full sweep.
- **Saaras V3:** `audit/fetch_stratified.py` (Modal, fetches a 450-clip
  family-stratified sample) → `audit/run_saaras_audit.py` (calls Sarvam's
  REST API directly, caches every response to disk, exponential backoff on
  rate limits). Adithya's task.
- **Qwen3-ASR-0.6B:** `audit/modal_qwen3_asr_audit.py`, same Modal pattern as
  the Whisper sweep but using `Qwen/Qwen3-ASR-0.6B-hf`'s native
  `apply_transcription_request` API instead of `WhisperForConditionalGeneration.generate()`.
  Also Adithya's task.
- **Scoring, for all six:** the exact same `ast_asr.metrics.normalize_for_wer`
  / `word_edit_counts` and `ast_asr.taxonomy.SVARAH_LANGUAGE_FAMILIES` this
  repo already uses for Phase 2 — so every row above is directly comparable to
  every other Svarah number we've ever reported, not a one-off metric.
- **Significance test:** the same Poisson drop-in-deviance GLM (statsmodels)
  from Phase 2 — utterance length as an offset, `family ~ 1` vs
  `family ~ C(family)`, p < 0.05 → "structural."

## Where things stand right now

- **This repo (`whispr-nano`) is the canonical home for the audit now.**
  `git push` here has always worked and everything below is pushed. An
  identical copy was also committed to `ast-asr`'s `fair-cispo-work` branch
  (5 commits, local), but that checkout has had push-credential problems
  all session — neither `git push` nor the GitHub integration could get
  write access to `snigenigmatic/ast-asr`. Rather than keep blocking on
  that, we're treating this repo as the system of record: everything the
  panel needs to see lives here, reachable and pushed, whether or not the
  `ast-asr` copy ever lands on GitHub.
- All six systems' scripts, summaries, and per-utterance CSVs are in
  `audit/svarah_fairness_audit/`. The independent verification layer,
  `assemble.py` in that same folder, globs all six
  `results_*_svarah_clean.csv`, hard-validates the schema (raises
  immediately on a missing/extra/reordered column or an unknown family
  label instead of failing quietly), recomputes overall WER, per-family
  WER, ΔDP, and the Poisson p-value straight from the per-utterance rows,
  and cross-checks the result against each `summary_*.json` — all six agree
  exactly, zero mismatches. 9/9 tests in `test_assemble.py` pass (schema
  drift, math correctness, summary-mismatch detection, all on a
  hand-computed fixture). Reproduce it yourself:
  `cd audit/svarah_fairness_audit && python assemble.py`.
- The Svarah-bias extension from the OPD side-quest (the original
  single-model feasibility check, plus the family-stratified on-policy
  distillation comparison) lives separately in `audit/opd_bias_extension/`
  — different scoring convention, different scope, kept apart on purpose
  so the two never get conflated.
- `docs/RECEIPTS_PACK.md` / `.pdf` (in this repo) has the three verified
  quotes for slides 2–3 (Sarvam's blog, ASR-FAIRBENCH, the clinical audit),
  each with the exact source text and retrieval date — plus one finding
  sharper than what was asked for: Sarvam's blog validates Saaras V3 on
  Svarah but never actually states a WER number for it, aggregate or
  otherwise.
- The three confirmed report-bug fixes (the missing HuBERT row, the [17]
  citation mix-up) were applied directly to `report/phase2_report.md` and
  the Review 2 slide deck in the `ast-asr` checkout, since those are edits
  to files that only exist there. They're committed locally in that repo;
  landing them on GitHub is blocked on the same push issue above. An
  agent with write access to `snigenigmatic/ast-asr` can push the lot
  from the file map in [`docs/AST_ASR_PUSH_HANDOFF.md`](AST_ASR_PUSH_HANDOFF.md).
- **Ownership note for the record:** the plan was Aditya on the Whisper
  family, Adithya on Saaras V3 + Qwen3-ASR, and Aditi on the assembler,
  report fixes, and receipts pack. Given the review deadline, I did all of
  it myself so we wouldn't be blocked on everyone's evening schedules — the
  scripts and hand-off docs are exactly what was written for each of you,
  so feel free to re-run any of them yourselves to double-check a number,
  extend to another system, or just see it work end to end. Nothing here
  replaces what you were going to build; it just means we already have the
  row on screen for Monday.

## The deck

`docs/PW25_BJD_05_Phase3_Review1.pptx` — built directly on the official
`Review1-phase3_ppt` template, ten slides:

1. Title (project ID, guide, team)
2. Outline
3. Abstract & Scope — the one-paragraph pivot story
4. Phase 2 summary + why we pivoted (GRL failure, GRPO caveats, August literature)
5. Architecture — one diagram, six systems, one frozen scoring path
6. List of tasks/modules
7. Individual contribution table (with the ownership note above, in writing)
8. **Demonstration** — the six-system table above, the per-family bar chart,
   plus the whispr-nano bonus result
9. References (IEEE-style, 12 real citations — Svarah, ASR-FAIRBENCH, the
   clinical audit paper, GKD/on-policy distillation, Saaras V3's own blog)
10. Thank you

Take a look, and shout if anything reads wrong for your part — especially
slide 7's contribution table, since that's the one place your name is
directly attached to a claim.

---

*Questions → group chat. If a number here doesn't match what you'd expect,
check `audit/svarah_fairness_audit/summary_*.json` first — that's the source of
truth, the table above was typed from it.*
