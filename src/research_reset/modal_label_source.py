"""Modal entry for the 12 label-source runs.

This file does not launch jobs on import. The local entrypoint refuses to
start until reconciliation.json exists and RESEARCH_RESET_LAUNCH=yes.

Usage (after the listening gate):

    RESEARCH_RESET_LAUNCH=yes modal run src/research_reset/modal_label_source.py
"""

from __future__ import annotations

from pathlib import Path

import modal

from research_reset.constants import DATA_DIR, PRIMARY_CHECKPOINT, REPO_ROOT

app = modal.App("research-reset-label-source")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("ffmpeg")
    .pip_install(
        "torch==2.13.0",
        "transformers==5.16.1",
        "accelerate==1.11.0",
        "datasets==4.4.1",
        "soundfile==0.14.0",
        "numpy==2.5.2",
        "jiwer>=3.0",
    )
    .env({"HF_HOME": "/cache/huggingface", "HF_HUB_ENABLE_HF_TRANSFER": "0"})
)

hf_cache = modal.Volume.from_name("ast-asr-cache", create_if_missing=True)


def _preflight() -> Path:
    import os

    recon = DATA_DIR / "reconciliation.json"
    if not recon.exists():
        raise SystemExit(
            "Refusing to launch: data/research_reset/reconciliation.json is missing. "
            "Finish the 120-clip dual transcription first."
        )
    if os.environ.get("RESEARCH_RESET_LAUNCH") != "yes":
        raise SystemExit(
            "Refusing to launch: set RESEARCH_RESET_LAUNCH=yes after the advisor "
            "has seen docs/ADVISOR_ONE_PAGER.md and the stop-rule in "
            "data/research_reset/design.json is still untouched."
        )
    return recon


@app.function(
    image=image,
    gpu="T4",
    timeout=PRIMARY_CHECKPOINT * 8 + 1800,
    volumes={"/cache/huggingface": hf_cache},
    secrets=[modal.Secret.from_name("huggingface")],
)
def train_run(spec: dict) -> dict:
    """Placeholder GPU body. Implemented when the 12 runs are actually scheduled.

    Pairing contract: the caller passes the frozen spec from run_specs.json.
    Do not sample a new recording order here.
    """
    raise NotImplementedError(
        f"{spec['run_id']}: GPU trainer ships after reconciliation. "
        f"primary_checkpoint={spec['training']['primary_checkpoint']}"
    )


@app.local_entrypoint()
def main():
    import json

    _preflight()
    specs = json.loads((DATA_DIR / "run_specs.json").read_text())
    print(f"Would launch {len(specs)} paired runs from {REPO_ROOT}. Not starting them from this stub.")
    for spec in specs:
        print(spec["run_id"], "n=", spec["n_recordings"], "pair=", spec["pair_id"])
