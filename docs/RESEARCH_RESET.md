# Two-week research reset

The goal of these two weeks is a **specific, reproducible problem that existing methods do not adequately solve**. A manuscript comes after that, if at all.

Candidate question (hypothesis, not a novelty claim):

> When do corrections learned from a teacher’s mistakes transfer to Indian-English speakers, and when do they fail?

Indian English stays central. Two reviewers do the audio check. Existing audit and OPD numbers stay **exploratory**. Test results do not trigger further tuning. We do **not** invent a new loss.

Advisor one-pager: [`docs/ADVISOR_ONE_PAGER.md`](ADVISOR_ONE_PAGER.md).  
Listening protocol: [`docs/LISTENING_PROTOCOL.md`](LISTENING_PROTOCOL.md).  
Label-source design (frozen before launch): [`data/research_reset/design.json`](../data/research_reset/design.json).

## Days 1–3 — decide whether the mechanism is worth pursuing

| Step | Status in this repo |
|---|---|
| One page with the advisor | Written. Print `docs/ADVISOR_ONE_PAGER.md`. |
| 120-clip dual transcription | Sample frozen (`data/research_reset/listening_set.json`). Reviewers use the listening lab or the CSV worksheets. Audio fetch still needs a Hugging Face token for gated Svarah. |
| 12-run label-source experiment | Specs frozen (`run_specs.json`). **Not launched.** Launch only after reconciliation and the stop-rule designation already in `design.json`. |
| Stop rule | Implemented and tested in `research_reset.stop_rule`. |

### Six L1 groups

Hindi, Bengali, Marathi (Indo-Aryan); Tamil, Telugu (Dravidian); Bodo (Sino-Tibetan, Svarah’s only ST language, **descriptive** later). Ten accepted and ten rejected clips each = 120. Accepted means whisper-large-v3 matches the reference under the **capstone** normalizer (the Phase 2 / audit convention). Rejected means it does not.

### Label-source gate

Cross **filtered** (teacher–reference match) vs **matched-random** recordings with **reference** vs **teacher** transcripts. Three paired seeds `(12, 34, 56)` → 12 runs. Within a `(pool, seed)` pair, init, recording order, greedy decode, and AdamW are identical. Checkpoints at **600** and **2,400** updates; **2,400 is primary** (designated in `design.json` before any run).

Continue to Days 4–10 only if teacher targets worsen matched-random **macro WER** (unweighted mean over the six L1s) by **≥ 0.5 percentage points on average**, with deterioration in **at least two seeds**, and **no sign reversal** under the Whisper English normalizer.

### What this split is not

The 2,500 scored Svarah IDs are a **gate-only** labeled pool. The 120 listening clips are held out; the rest split 80/20 train/eval. This is **not** the paper test set. **SPIRE-SIES** and **L2-ARCTIC** stay unused until Days 11–14. Days 4–10 must rebuild train/dev with verified speaker identities.

## Days 4–10 — challenge existing solutions

Proceed only if the gate passes.

- Indian-English train/dev with verified speaker IDs.
- Arms: ordinary pseudo-label SFT; uDistil-Whisper-style filtering ([Waheed et al., NAACL 2025](https://aclanthology.org/2025.naacl-long.296/)); Pseudo2Real ([Lin et al., ACL Findings 2026](https://aclanthology.org/2026.findings-acl.59/)); reference SFT as a **privileged diagnostic**.
- Match init, exposure, and tuning. Count source vs target annotations separately.
- Report every supported L1; small groups descriptive.
- If an existing method closes the gap, **that is the finding**. Any proposed extension must address a remaining, demonstrated failure.

Code sketch of the arm list: `src/research_reset/days_4_10.py`.

## Days 11–14 — research decision

Freeze the comparison **before** opening SPIRE-SIES and L2-ARCTIC. Recompute metrics independently. Uncertainty: resample **speakers**; keep training-seed variation in a separate table.

Decision memo: [`docs/DECISION_MEMO_TEMPLATE.md`](DECISION_MEMO_TEMPLATE.md). Continue toward a paper only if the finding survives independent speakers and is more than “we applied Pseudo2Real to another dataset.”

## Commands

```bash
# Rebuild listening set, gate split, and 12-run specs from the committed CSV
python -m research_reset

# Tests for sampling, pairing, scoring, stop rule
python -m pytest tests/test_research_reset.py -v

# Listening lab (two reviewers, transcripts hidden until both submit)
cd listen && npm run dev -- --port 43147 --hostname 127.0.0.1
```

Do not run the 12 GPU jobs until `data/research_reset/reconciliation.json` exists and the stop-rule thresholds in `design.json` are unchanged.
