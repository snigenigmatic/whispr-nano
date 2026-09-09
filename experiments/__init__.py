"""Fixed-command experiment runner for the OpenResearch (`orx`) tree.

Every experiment node runs the same command (`uv`-backed `python -m experiments.run`).
Variants differ only in the committed `experiments/run.json` on that node's branch.
"""
