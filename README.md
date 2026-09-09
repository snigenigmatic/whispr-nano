# PW25_BJD_05 — Whose Indian Accent?

**Current work: a two-week research reset.** The goal is a specific,
reproducible problem that existing methods do not adequately solve — not a
finished manuscript.

Candidate question (hypothesis, not a novelty claim):

> When do corrections learned from a teacher’s mistakes transfer to
> Indian-English speakers, and when do they fail?

| | |
|---|---|
| **Project ID** | PW25_BJD_05 |
| **Guide** | Dr. Bhaskarjyoti Das |
| **Team** | Aditya Sharma, Adithya V Holla, C Kaustubh, Aditi Mangala Udaya |
| **This week** | Days 1–3 of [`docs/RESEARCH_RESET.md`](docs/RESEARCH_RESET.md) |
| **Advisor page** | [`docs/ADVISOR_ONE_PAGER.md`](docs/ADVISOR_ONE_PAGER.md) |

## Days 1–3, in this repo

1. **One page for the advisor** — negative result, the question, and
   [Pseudo2Real](https://aclanthology.org/2026.findings-acl.59/) as the closest
   competing method. Print `docs/ADVISOR_ONE_PAGER.md`.
2. **120-clip dual transcription** — six L1 groups (Hindi, Bengali, Marathi,
   Tamil, Telugu, Bodo) × accepted/rejected × 10. Two reviewers, blind to both
   transcripts. Frozen IDs in `data/research_reset/`. Protocol:
   [`docs/LISTENING_PROTOCOL.md`](docs/LISTENING_PROTOCOL.md). Listening lab:
   `listen/` (see below).
3. **Label-source experiment, specified, not launched.** 2 recording pools × 2
   label sources × 3 paired seeds = **12 runs**. Primary checkpoint **2,400
   updates**, designated before launch in `data/research_reset/design.json`.
   Stop rule: teacher targets must worsen matched-random macro WER by ≥ 0.5 pp
   on average, in at least two seeds, with no reversal under the Whisper
   normalizer — or we close this mechanism.

SPIRE-SIES and L2-ARCTIC stay frozen. We do not invent a new loss. Existing
audit and on-policy distillation numbers stay **exploratory**.

```bash
PYTHONPATH=src python -m research_reset          # rebuild listening set + 12-run specs
python -m pytest tests/test_research_reset.py -v
cd listen && npm install && npm run dev -- --port 43147 --hostname 127.0.0.1
```

## Exploratory background (not the claim)

A significance-tested subgroup audit of six ASR systems on a frozen 2,500-clip
Svarah sample: every system has a structural per-family gap, and Sino-Tibetan
is worst in every row — including Saaras V3 (best overall WER, not the fairest).
Full table and reproduction:
[`audit/svarah_fairness_audit/README.md`](audit/svarah_fairness_audit/README.md).

On-policy distillation of whisper-tiny from large-v3 is parked as a side-quest
([`docs/ONPOLICY_DISTILLATION.md`](docs/ONPOLICY_DISTILLATION.md)). It did not
beat off-policy pseudo-labels on LibriSpeech, and it is not the Days 1–3 gate.

## Repository layout

```
docs/ADVISOR_ONE_PAGER.md     # one page for Dr. Das
docs/RESEARCH_RESET.md        # two-week plan
docs/LISTENING_PROTOCOL.md    # two-reviewer audio check
data/research_reset/          # frozen 120 IDs, gate split, 12-run specs
src/research_reset/           # sampling, scoring, pairing, stop rule
listen/                       # two-reviewer lab (Next.js)
audit/svarah_fairness_audit/  # six-system audit (exploratory)
src/onpolicy_distill/         # OPD side-quest
experiments/                  # orx runner (OPD tree)
```

## Relationship to `ast-asr`

Phase 1/2 code lives in [`snigenigmatic/ast-asr`](https://github.com/snigenigmatic/ast-asr)
(`fair-cispo-work`). This repo is the Phase 3 / research-reset home. Handoff:
[`docs/AST_ASR_PUSH_HANDOFF.md`](docs/AST_ASR_PUSH_HANDOFF.md).
