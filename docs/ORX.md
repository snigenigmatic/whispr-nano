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

## First two rounds (ran locally, 2026-09-09)

CPU tiny/tiny, LibriSpeech dummy, 8/4 clips, 6 steps. Project
`3259e90d-d5d9-4e9a-8b9c-505570dcbc80`.

| Node | Policy | ΔWER pp | Verdict |
|---|---|---|---|
| Baseline `orx/baseline-on-policy-reverse-kl` | on_policy KL | +0.00 | flat (expected: identical teacher) |
| Child `orx/off-policy-reverse-kl` | off_policy KL | +0.00 | flat |
| Child `orx/scheduled-sampling-gold-ce` | scheduled_sampling p=0.5 | **−2.70** | improved — promoted |
| Grandchild mix 0.0 | scheduled_sampling p=0.0 | −2.70 | tie |
| Grandchild mix 1.0 | scheduled_sampling p=1.0 | −2.70 | tie |

Gold CE moved WER; reverse KL against a copy of tiny did not. Mix rate did
not change the 4-clip WER. Stop sweeping mix on this set.

## Round 3 — real teacher, Modal T4 (2026-09-09)

Siblings under the scheduled-sampling winner. `whisper-tiny` student,
`whisper-large-v3` teacher, 16 train / 8 eval, 16 steps, CUDA T4, encoders
frozen. Command: `python -m experiments.run`. Earlier sandboxes died on
image deps (3.12-only pins, torchcodec, relative flac paths) and were
repaired in place — those runs answered nothing.

| Node | Policy | ΔWER pp | KL before→after | Wall | Verdict |
|---|---|---|---|---|---|
| `orx/on-policy-kl-large-v3-teacher` | on_policy | +0.00 | 0.827→0.760 | 70s | flat WER, KL improved |
| `orx/scheduled-sampling-large-v3-teacher` | scheduled_sampling | +0.00 | 0.827→1.041 | 44s | flat WER, KL worse |

Teacher WER on the same 8 clips: 14.29%. Student stayed at 29.32%. This
smoke is too short to move WER; on-policy is the only arm that reduced
reverse KL. Next descent: the original 60-step PoC budget on the on-policy
node, not another mix sweep.

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
