---
name: orx
description: Drive automated ML research on OpenResearch with the `orx` CLI — create experiments, launch and monitor runs on compute, analyze local results and logs, and search literature. Use whenever the user wants to understand, explain, explore, or work on an OpenResearch project, run experiments, do auto-research, or mentions orx or OpenResearch.
---

# OpenResearch (`orx`)

You drive OpenResearch through the `orx` command-line tool. The authoritative
operating manual is bundled inside the CLI, so **load it at the start of every
session** instead of relying on this file or prior memory.

This repo vendors the same skill the official installer writes to
`~/.cursor/skills/orx/SKILL.md` (`orx install-skills --agent cursor --full`).
Install the CLI with `curl -LsSf https://openresearch.sh/install.sh | sh`.

## 1. Load the bundled guide

```bash
orx skill
```

This prints the current manual — the cardinal rules and a command
quick-reference — followed by a **bundled index of modules**. Read it before taking
any action. For the detail on a specific area, run `orx skill <name>` to print
that module (e.g. `orx skill experiment-tree`, `orx skill compute`); the same
command reads that module directly from the installed CLI.

## 2. Cardinal rules (non-negotiable)

1. **Never edit a node once a run has answered it.** Branch a child.
2. **The run command and environment are a fixed contract** on every node.
   This project's command is ` /workspace/.venv/bin/python -m experiments.run `.
   Variants live in committed `experiments/run.json`, never in env vars or flags.
3. **Vary committed code/config, not knobs-in-the-command.**
4. **Grow the tree downward** (stacked bushes). Fan siblings for one decision,
   then descend onto that round's winner. Do not hang a long row off the root.

## 3. This project's tree

Local research commands do not require `orx login`. GPU backends (Modal, HF
Jobs, OpenResearch boxes) do.

```bash
orx projects
orx project view <projectId>
orx create-experiment <projectId> --title "Baseline"
orx create-experiment <projectId> --parent <id> --title "Scheduled sampling" \
  --description "Set experiments/run.json policy to scheduled_sampling; change nothing else."
# checkout the printed orx/<slug> branch, edit run.json, commit, then:
orx exp run <expId> --backend local
orx exp wait --project <projectId>
orx logs <runId>
```

Full project notes: `docs/ORX.md`.
