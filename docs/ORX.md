# OpenResearch (`orx`) on this repo

The [OpenResearch CLI](https://github.com/alphaXiv/OpenResearch) drives a
**tree of experiment nodes**. Each node is a git branch. Every node runs the
**same** command; the only thing that may differ is the committed
`experiments/run.json` (and any code on that branch).

This is the official skill the user pointed at
(`/Users/prateekraj/.cursor/skills/orx/SKILL.md` on a local Mac). The same
document is vendored here as [`.cursor/skills/orx/SKILL.md`](../.cursor/skills/orx/SKILL.md)
and installed into the agent via `orx install-skills --agent cursor --full`.

## Install

```bash
curl -LsSf https://openresearch.sh/install.sh | sh
orx install-skills --agent cursor --full
# optional, only for hosted OpenResearch GPUs:
orx login
```

Local project + `orx exp run --backend local` do **not** need a login.

## Fixed run contract

```text
/workspace/.venv/bin/python -m experiments.run
```

That process reads `experiments/run.json`, trains one Whisper arm, and prints
a `===== RESULT SUMMARY =====` block. Judge nodes from `orx logs`, not from
memory.

Hyperparameters live in the JSON. Do not pass `LR=…` or change the command
per node.

## What the first round is asking

The published Whisper-AED cell we occupy is: **teacher-logit on-policy
distillation** (student prefixes, reverse KL vs a teacher) versus the older
ASR trick of **scheduled sampling** (on-policy or mixed prefixes, gold CE).
Moriya et al. ICASSP 2023 did the latter on a Conformer RNN-T; they did not
do teacher-logit OPD. LS-MOPD / OPSD / Ark-ASR (2026) do OPD on LLM decoders.

Round 1 (CPU, `whisper-tiny` as both student and teacher, LibriSpeech dummy)
isolates the **training signal** before we spend GPU budget on
`whisper-large-v3`:

| Node | `policy` | What it tests |
|---|---|---|
| Baseline | `on_policy` | Student prefixes + reverse KL vs the teacher (here: a frozen copy of tiny) |
| Child | `scheduled_sampling` | Mixed gold/student prefixes + gold CE (Moriya-style) |
| Child | `off_policy` | Teacher prefixes + reverse KL (Distil-Whisper-style ablation) |

Tiny-as-teacher is a cheap probe, not the capstone claim. The next descent
after a winner is swapping the teacher to `openai/whisper-large-v3` on Modal
or HF Jobs — same command, different committed JSON.

## Drive the tree

```bash
orx projects
orx project view <projectId>
orx exp status <expId>
orx runs <projectId>
orx logs <runId>
orx exp wait --project <projectId>
```

Create a child, check out its `orx/<slug>` branch, edit `experiments/run.json`,
commit, then `orx exp run <id> --backend local`. Never edit a node a run has
already answered.

## Cost

This VM has no GPU and no Modal/HF token in the orx compute settings. Local
runs stay on CPU with tiny/tiny. Do not launch `--backend modal` or `--backend hf`
until those tokens are configured. The $50 Hugging Face credit cannot pay
Sarvam or Modal.
