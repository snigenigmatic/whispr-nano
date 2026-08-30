# Agent brief: push Phase 3 audit work into `snigenigmatic/ast-asr`

Give this file to an agent that already has **write access** to
[`snigenigmatic/ast-asr`](https://github.com/snigenigmatic/ast-asr).
This repo (`whispr-nano`) is the source of truth. The destination checkout
already has the same five commits locally on `fair-cispo-work` in one
environment, but they never landed on GitHub (push credentials failed).
Do **not** re-run any audit. Copy files, apply two small report patches,
commit, push.

## Constraints

- Branch: **`fair-cispo-work` only**. It is a strict superset of every
  other branch. Do not create a new branch.
- Do **not** touch `src/` or `configs/` (hash-frozen).
- Do **not** invent new numbers, re-score CSVs, or call Modal / Sarvam.
- Do not open a PR unless the human asks.

## 1. Checkout

```bash
git clone https://github.com/snigenigmatic/ast-asr.git
cd ast-asr
git checkout fair-cispo-work
git pull origin fair-cispo-work
```

If that checkout is already **5 commits ahead** of `origin/fair-cispo-work`
and the tips are `f8f7b00`, `f42d69e`, `190c375`, `872387b`, `fcb90e1`,
just `git push -u origin fair-cispo-work` and stop. The rest of this
brief is only for a clean remote that does not have those commits.

## 2. Copy files from `whispr-nano`

Clone or otherwise obtain the current `whispr-nano` tree (the repo this
handoff lives in). Then copy **into the `ast-asr` tree**, flattening the
folder — destination is `audit/`, not `audit/svarah_fairness_audit/`.

| From `whispr-nano` | To `ast-asr` |
|---|---|
| `audit/svarah_fairness_audit/modal_whisper_family_audit.py` | `audit/modal_whisper_family_audit.py` |
| `audit/svarah_fairness_audit/modal_qwen3_asr_audit.py` | `audit/modal_qwen3_asr_audit.py` |
| `audit/svarah_fairness_audit/fetch_stratified.py` | `audit/fetch_stratified.py` |
| `audit/svarah_fairness_audit/run_saaras_audit.py` | `audit/run_saaras_audit.py` |
| `audit/svarah_fairness_audit/assemble.py` | `audit/assemble.py` |
| `audit/svarah_fairness_audit/test_assemble.py` | `audit/test_assemble.py` |
| `audit/svarah_fairness_audit/master_audit_table.csv` | `audit/master_audit_table.csv` |
| `audit/svarah_fairness_audit/master_audit_table.md` | `audit/master_audit_table.md` |
| `audit/svarah_fairness_audit/results_whisper-tiny_svarah_clean.csv` | `audit/results_whisper-tiny_svarah_clean.csv` |
| `audit/svarah_fairness_audit/results_whisper-small_svarah_clean.csv` | `audit/results_whisper-small_svarah_clean.csv` |
| `audit/svarah_fairness_audit/results_whisper-large-v3_svarah_clean.csv` | `audit/results_whisper-large-v3_svarah_clean.csv` |
| `audit/svarah_fairness_audit/results_distil-large-v3_svarah_clean.csv` | `audit/results_distil-large-v3_svarah_clean.csv` |
| `audit/svarah_fairness_audit/results_saarasv3_svarah_clean.csv` | `audit/results_saarasv3_svarah_clean.csv` |
| `audit/svarah_fairness_audit/results_Qwen3-ASR-0.6B-hf_svarah_clean.csv` | `audit/results_Qwen3-ASR-0.6B-hf_svarah_clean.csv` |
| `audit/svarah_fairness_audit/summary_whisper-tiny_svarah.json` | `audit/summary_whisper-tiny_svarah.json` |
| `audit/svarah_fairness_audit/summary_whisper-small_svarah.json` | `audit/summary_whisper-small_svarah.json` |
| `audit/svarah_fairness_audit/summary_whisper-large-v3_svarah.json` | `audit/summary_whisper-large-v3_svarah.json` |
| `audit/svarah_fairness_audit/summary_distil-large-v3_svarah.json` | `audit/summary_distil-large-v3_svarah.json` |
| `audit/svarah_fairness_audit/summary_saarasv3_svarah.json` | `audit/summary_saarasv3_svarah.json` |
| `audit/svarah_fairness_audit/summary_Qwen3-ASR-0.6B-hf_svarah.json` | `audit/summary_Qwen3-ASR-0.6B-hf_svarah.json` |
| `docs/RECEIPTS_PACK.md` | `docs/RECEIPTS_PACK.md` |
| `docs/RECEIPTS_PACK.pdf` | `docs/RECEIPTS_PACK.pdf` (force-add if `*.pdf` is gitignored) |

Do **not** copy `audit/svarah_fairness_audit/README.md` as-is — that README
is written for `whispr-nano`. A one-paragraph pointer in `ast-asr/audit/`
is enough if you want a local note.

Do **not** copy `audit/opd_bias_extension/` — that is the OPD side-quest,
not the six-system Phase 3 audit.

## 3. Apply the two report-bug patches (not in `whispr-nano`)

These edits only exist on files that live in `ast-asr`. Apply them exactly.

### 3a. `report/phase2_report.md`

1. In §2.2, change the GRPO-on-ASR citation from
   `Radhakrishnan et al. [17]` to `Shivakumar et al. [17]`.
2. In the §4.2 experiment list, replace the HuBERT claim with:

   > 1. Zero-shot baselines (Table 6.1). Whisper-tiny, Whisper-small, and
   > Wav2Vec2-base, evaluated against Svarah-eval clean. HuBERT was planned
   > for this cluster but never run — no checkpoint, log, or result CSV
   > exists for it anywhere in this repo; Table 6.1 records that honestly
   > with a "not run" row rather than silently dropping it.

3. In Table 6.1, add a row after Whisper-small:

   `| HuBERT | *not run* | — | — | — | — | — |`

   and immediately after the table:

   > HuBERT is listed as planned in §4.2 and Appendix references [8], but no
   > zero-shot HuBERT run was ever executed — there is no checkpoint, log, or
   > result file for it in this repo. We record the gap rather than silently
   > omit the row.

4. Bibliography entry [17] becomes:

   > [17] P. G. Shivakumar, Y. Gu, A. Gandhe, and I. Bulyko (Amazon),
   > "Group Relative Policy Optimization for Speech Recognition,"
   > arXiv:2509.01939, 2025. *(Corrected 2026-08-29: the author list
   > previously attached to this arXiv ID — "V. S. Unni, A. Mittal, P. Jyothi,
   > S. Sarawagi et al." — does not match the paper; verified against the
   > arXiv abstract page.)*

### 3b. `PW25_BJD_05_Phase2_Review2.html`

Two string replacements only:

- `Radhakrishnan et al. — first GRPO-RLHF` → `Shivakumar et al. — first GRPO-RLHF`
- `Radhakrishnan, Unni, Mittal, Jyothi, Sarawagi et al. <em>Group Relative Policy Optimization for Speech Recognition.</em>`
  → `Shivakumar, Gu, Gandhe, Bulyko. <em>Group Relative Policy Optimization for Speech Recognition.</em>`

The §7.2 "Honest limitations" half-sentence bug is already gone in the
current `phase2_report.md`. Do not invent a third edit.

## 4. Commit and push

Suggested commits (or one commit if you prefer; keep the message honest):

```text
Add six-system Svarah fairness audit: Whisper family, Saaras V3, Qwen3-ASR
Add Saaras V3 Svarah audit results (Sarvam API, 450 clips)
Fix three confirmed report bugs (Aditi AT2)
Add results assembler with schema hard-fail (Aditi AT1)
Add receipts pack for deck slides 2-3 (Aditi AT3)
```

Then:

```bash
git push -u origin fair-cispo-work
```

## 5. Done when

- `origin/fair-cispo-work` contains the six `results_*_svarah_clean.csv`
  files, the six `summary_*.json` files, `audit/assemble.py`,
  `docs/RECEIPTS_PACK.md`, and the two report-bug edits.
- `git status` is clean. `src/` and `configs/` are untouched.

Monday's deck and numbers already live in `whispr-nano`. This push is
bookkeeping so the course repo matches. If push fails, say so and stop —
do not open a PR to a different repo or rewrite history.
