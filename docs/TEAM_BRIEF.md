# 🎓 The Story So Far — a briefing for Team PW25_BJD_05

*by Kau, for Aditya, Adithya, and Aditi — read time ~10 minutes, no prerequisites* 💛

---

## 👋 Why you're reading this

Three things happened while everyone was buried in placement prep:

1. Our capstone territory got **encroached** (yes, we got byregowda'd 🏗️).
2. I did a proper literature search, found an escape route, and wrote a pivot memo.
3. I also built and ran a **secret side experiment on a rented GPU for ₹14** ($0.16) to test the *other* idea before parking it.

You haven't seen any of this yet. This document catches you up on all of it, from zero. By the end you'll know exactly what the pivot is, why it's actually *good* news, and what your weekend mission is. ☕

---

## 📖 Previously, on our capstone...

Quick refresher (you lived this, skim it):

- **Our thesis:** ASR systems look fine on *average* WER for Indian English, but nobody checks whether the errors fall evenly across accents — and nobody tests whether the gaps are *real* or just statistical noise.
- **What we built in Phase 2:** an evaluation pipeline on Svarah (19 accents → 3 language families) that reports per-family WER, ΔDP, ΔEO, noise-robustness — each with a **p-value** from a Poisson test. "A gap with p = 0.02 is structure; a gap with p = 0.08 is noise."
- **What worked:** the pipeline. It's honestly the best thing we made.
- **What didn't:** GRL debiasing made fairness *worse* (because SPIRE-SIES has zero Sino-Tibetan speakers — you can't reverse a gradient for a group the model never sees). We pivoted to GRPO, which cut WER 18.4% → 16.4% and moved the gap's p-value from "structural" to "not distinguishable from noise."

Hold that last claim loosely. We'll come back to it. 🫣

---

## 💔 Then August happened

I ran a real literature search (63 records → 12 kept). Three findings, two painful, one hopeful:

### 1. Sarvam took our thesis 🥲

Saaras V3 (their flagship ASR) was evaluated **on Svarah** — our exact benchmark — and their blog says why: Indian English is underrepresented in Western datasets. That's our Chapter 1, shipped by a funded lab. They also trained with **RL post-training on a million+ hours**. Our GRPO ran 1,500 steps on one GPU. We do not win that fight.

**BUT:** they report *one aggregate number*. No per-accent breakdown. No significance test. Nobody — literally nobody — has checked whether India's flagship ASR model serves all Indian accents equally. 👀

### 2. Our Poisson test already exists in print 🙈

ASR-FAIRBENCH (Interspeech 2025) uses a **mixed-effects Poisson regression** for exactly this. It's citation [2] *in our own report*. Theirs is better than ours (it has speaker random effects — which we listed as our own limitation without realizing the fix was published). So we can't claim the statistical machinery. We *can* adopt it, cite it, and stop being a soft target.

### 3. Audits get published ✅

A Nov 2025 paper audited 8 ASR systems (including Sarvam!) on Indian clinical speech and got through review. Third-party audits of Indian ASR are a *normal, publishable thing*. Their axes were role and gender — **the accent axis is still open.**

---

## 💡 The pivot: from building a fair model → auditing everyone else's

**Proposed title:** *"Whose Indian accent? A significance-tested subgroup audit of Indian and frontier ASR systems."*

The idea in one sentence: **point the pipeline we already built at every ASR system we can reach** — Whisper family, Saaras V3, Qwen3-ASR, whatever else — and report per-family WER with a p-value and a noise condition for each, all through one frozen decode path so the rows are actually comparable.

Why this is the right move for us, specifically, right now:

| 😩 The old plan (GRPO paper) | 😌 The audit |
|---|---|
| Competes with Sarvam's million GPU-hours | Uses zero training compute |
| Needs weeks we don't have | A few GPU-hours total, ~₹100–200 |
| Our headline claim is fragile (see below) | Our failed experiments become *evidence* |
| Placement season says no | Runs in evenings, mostly delegated to agents |

And the beautiful part: **our failures become the motivation section.** GRL failed because the adversary couldn't see Sino-Tibetan speakers → "you cannot fix a gap you cannot see." GRPO improved the average while Indo-Aryan gained 2.5 points and Sino-Tibetan gained 0.3 → "aggregate movement hides subgroup movement." Both are now exhibits, not claims we have to defend in a viva. 🛡️

### ⚠️ Two integrity checks first (the "Blockers")

Before we show anyone anything, I'm running two cheap CPU checks on our *own* Phase 2 numbers:

- **Blocker A:** our "the gap became non-structural" result might actually be the bias *concentrating* on Bodo (our smallest group) while the p-value lost power. If so, §6.3 gets rewritten — honestly. Either outcome supports the pivot.
- **Blocker B:** Svarah has parallel utterances and our split was by *speaker*, not by *sentence text* — so some of our 2-point WER win might be the decoder memorizing transcripts. Need to know Saturday, not the night before the panel.

Finding your own bugs before the reviewers do is what an audit paper's authors *should* be doing. It's on brand. ✨

---

## 🤫 The secret side quest: whispr-nano

While researching, I found the idea I *really* loved: **on-policy distillation (OPD)**. Quick cute explainer:

> 🧑‍🏫 **Normal distillation:** the teacher writes perfect example transcripts, the student copies them. But the student never practices recovering from *its own* mistakes — like learning cricket by only watching highlights.
>
> 🏏 **On-policy distillation:** the *student* attempts the transcription, and the teacher grades *every word of the student's own attempt*. The student learns exactly where its own instincts are wrong. (This is the recipe behind recent LLM post-training work — GKD, Thinking Machines, Qwen3.)

Bad news: two 2026 papers already did OPD for ASR (Ark-ASR, LS-MOPD). So it's not our paper *this* cycle — it's parked, deliberately, as the **Interspeech 2027 story** (imagine: "whose accent survives distillation?" — our audit machinery × OPD training 🤝).

But before parking it, I wanted proof it actually works. So I built it: **[whispr-nano](../README.md)** — whisper-tiny learning from whisper-large-v3, on Modal's serverless GPUs.

Results (real numbers, one T4 GPU, total cost **$0.16**):

| | WER before → after | teacher-student KL before → after |
|---|---|---|
| on-policy | 10.09% → **7.89%** | 0.633 → **0.496** |
| off-policy (pseudo-labels) | 10.09% → 7.57% | 0.633 → 0.524 |

Both work. On-policy wins on the calibration metric (the thing it optimizes), exactly as the literature predicts. The whole thing — code, tests, plots, a research writeup with citations, and a **Modal credits application** (`PROPOSAL.md`) — is in this repo. The entire experiment including my failed debugging runs cost about **$0.22**. That's one samosa. 🥟

Why you care *this* weekend: the Modal setup (cached images, cost guardrails, storage volumes) is **exactly the skeleton we'll reuse to run the audit**, and the credits pitch funds the 2027 story.

---

## 🗓️ Your weekend missions

Full step-by-step ops guide with agent prompts: **[WEEKEND_PLAN.md](WEEKEND_PLAN.md)**. The short version:

### 🌙 Friday night, all of us, 45 min call
Freeze the CSV schema + decode rules (greedy, temp 0, Whisper text normalizer for everyone, ONE shared noisy eval set). Check API keys and Modal access. This call is the difference between a smooth weekend and Sunday-night column-reconciliation hell.

### 🧑‍🔬 Kau (me)
Blockers A & B tonight → mixed-effects stats upgrade → §6.3 rewrite → the panel deck. My agents write the stats code; I judge the outputs.

### 🐳 Aditya — the Whisper fleet
Port our eval to Modal (agent does the port, using whispr-nano as the template — brief is in the plan). Run whisper-tiny/small/large-v3/distil-whisper, clean + noisy. Eyeball 10 transcripts per system before launching the full run. Budget: ~$1–2 total.

### 🛰️ Adithya — the outside world
**First 30 min:** the Saaras gate — send 5 clips through their API by hand and check the output is scoreable (this is our biggest unknown). Then agent-built API client with caching + backoff. Then Qwen3-ASR (20-min steerability check first — it might just refuse). If a toolchain breaks, you're the designated fallback: grab extra Whisper-family models instead. Post projected API costs to the group *before* the full run. 💸

### 🧾 Aditi — the glue and the receipts
Agent-built results assembler that **hard-fails on schema drift** (you are the schema police 👮). Fix the three confirmed report bugs (missing HuBERT row, §7.2's half-sentence, the [17] Radhakrishnan/Unni mix-up). Build the receipts pack — screenshots of Sarvam's aggregate-only blog, ASR-FAIRBENCH's protocol, the clinical audit — so the panel doesn't have to take our word for anything.

### 🤖 The agent rule
Anything that is *code with a clear spec* — clients, scorers, stats fits, schema validators, deck skeletons — goes to an agent with the briefs in the plan. Anything that is *judgment* — eyeballing transcripts, interpreting Blocker A, deciding fallbacks, the deck's story — stays human. We are four people with placements; the agents are the fifth teammate who doesn't sleep. 🌙

---

## ✅ Sunday 20:00 — we decide three things

1. **Drop the GRPO paper and run the audit?** (my vote: yes — and by Sunday we'll have actual audit rows on screen, which is the strongest argument)
2. **Who takes which systems** — confirm against what actually ran
3. **Adopt mixed-effects stats?** (yes — adopt, cite, stop being the soft target)

Then the deck goes to BJD *before* the panel — he already told Adithya to pivot, so he's primed; his edits predict the panel's objections.

---

## 🍀 Tiny glossary

- **Svarah** — 9.6h benchmark of Indian-accented English, 19 accents. Our home turf.
- **ΔDP** — the WER spread between best and worst language family. Big = unfair.
- **Poisson / mixed-effects test** — the thing that turns "the gap looks big" into "the gap is real, p = 0.02."
- **On-policy distillation (OPD)** — student practices, teacher grades the student's own attempt. Parked till 2027.
- **Reverse KL** — "how surprised is the teacher by what the student was about to say." Lower = student better calibrated.
- **Modal** — serverless GPUs, billed per second. Where whispr-nano ran for $0.16, and where the audit will run.
- **byregowda'd** — *(v., colloquial)* to have one's research land encroached upon by a well-funded neighbour. 🏗️

---

*Questions → the group chat. Panic → also the group chat, but with the 🚨 emoji so we know.*

**Let's go get our row of the table.** 🚀
