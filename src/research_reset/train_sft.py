"""Whisper CE fine-tune used by the 12 label-source runs.

The 12 GPU jobs are not launched from this module. Call `train_one_run`
from Modal only after the listening reconciliation is done and the
advisor has seen the one-pager.

Encoder is frozen. Decoder is trained with teacher-forced CE on either
the dataset reference or the teacher transcript of the same recording.
Initialization, clip order, greedy eval decode, and optimizer are taken
from the run spec so paired arms stay paired.
"""

from __future__ import annotations

from typing import Any, Sequence


def target_text(row: dict[str, Any], label_source: str) -> str:
    if label_source == "reference":
        return row["reference_text"]
    if label_source == "teacher":
        return row.get("teacher_text") or row["hypothesis_text"]
    raise ValueError(f"unknown label_source {label_source!r}")


def checkpoint_schedule(num_updates: int, checkpoints: Sequence[int]) -> list[int]:
    return [c for c in checkpoints if 0 < c <= num_updates]


def pair_fingerprint(spec: dict[str, Any]) -> tuple:
    """Values that must be identical across the two label sources of a pair."""
    return (
        spec["recording_pool"],
        spec["seed"],
        tuple(spec["recording_ids"]),
        spec["training"]["student_model"],
        spec["training"]["optimizer"]["learning_rate"],
        spec["training"]["optimizer"]["weight_decay"],
        spec["training"]["batch_size"],
        spec["training"]["freeze_encoder"],
        tuple(spec["training"]["checkpoints"]),
        spec["training"]["teacher_decode"]["do_sample"],
        spec["training"]["teacher_decode"]["dtype"],
    )
