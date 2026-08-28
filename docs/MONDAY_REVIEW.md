# Monday Capstone Review — Kau's personal task list

Scoped-down from [WEEKEND_PLAN.md](WEEKEND_PLAN.md) for one constraint: review is Monday, no full
team run is possible, and this needs to be executable solo. Goal is not the full audit — it's enough
*real, new, checkable* evidence to move a skeptical panel toward approving the pivot.

Context: BJD's "Regrading the Capstone Review" email is explicit — *"many teams haven't done much
implementation... if you have done anything, please showcase it. Similarly, if you have submitted or
published already, please showcase that."* Lead with volume of real work: the submitted ACL SRW
paper, the Phase 2 pipeline, the traced GRL failure, the GRPO result, and whispr-nano — a real $0.16
GPU experiment built and run this week.

## Friday night (~1.5 hrs) — Blockers A & B

No new GPU spend, no teammates needed. Pure CPU stats on existing Phase 2 CSVs. This gates what you're
allowed to claim about the GRPO result in the review.

- **Blocker A** — GLM coefficient autopsy: is "gap became non-structural" actually bias concentrating
  on Bodo/ST while IA≈Dravidian? Agent brief: `WEEKEND_PLAN.md` §K1.
- **Blocker B** — train/eval text overlap: does the 2-point WER drop survive removing
  memorized-transcript utterances? Agent brief: `WEEKEND_PLAN.md` §K2.

Either outcome is presentable. Broken claim → rigor story ("we re-audited our own result and caught
a subgroup power problem — which is why we're pivoting to auditing"). Holds → one sentence, move on.

## Saturday — one real new audit row

Skip the full multi-system sweep. Produce **one** significance-tested per-family WER number, on a
system Phase 2 never touched, reusing infrastructure that's already paid for:

- `whisper-large-v3` is already cached in this repo's Modal Volume (whispr-nano's teacher) — zero
  re-download cost.
- Svarah (`ai4bharat/svarah`) is public, 1,998 eval utterances, ~$1–2 of T4 time for a clean pass.
- Per-family WER / ΔDP / Poisson test machinery already exists in Phase 2's `ast-asr/pipeline.py` —
  reuse, not new code.

Output: one slide that says "here is the missing row — we already produced it, for $2" instead of
"we plan to."

## Sunday — assemble, don't write from scratch

Six slides, mostly condensed from existing docs:

1. Phase 2 recap + the August encroachment (memo / `TEAM_BRIEF.md`)
2. Blocker A/B outcome, stated plainly
3. The pivot proposal + coverage table (memo §3)
4. **Money slide:** the one live audit row from Saturday
5. whispr-nano showcase: real numbers, $0.16, on/off-policy (`README.md` results table)
6. The ask: approve the pivot; ACL SRW unaffected; scale-up already has a Modal credits application
   drafted (`PROPOSAL.md`)

## Monday morning

Every number must trace to a file. Bring the Modal dashboard open — "here's the run, here's the $2
receipt" answers "how do we know this scales" better than any slide.
