"""12-run specs: 2 recording pools × 2 label sources × 3 paired seeds.

Initialization, recording order, decoding, and optimizer are fixed
within each (pool, seed) pair. The only intended difference between
reference and teacher arms is the transcript used as the training target.
"""

from __future__ import annotations

from typing import Any, Sequence

from .constants import (
    ALTERNATE_NORMALIZER,
    LABEL_SOURCES,
    PAIRED_SEEDS,
    PRIMARY_CHECKPOINT,
    PRIMARY_NORMALIZER,
    RECORDING_POOLS,
    SECONDARY_CHECKPOINT,
    STOP_RULE_MEAN_PP,
    STOP_RULE_MIN_SEEDS_WORSE,
    STUDENT_MODEL,
    TEACHER_MODEL,
    TEACHER_SYSTEM,
)
from .pools import construct_filtered_pool, construct_matched_random_pool, ordered_recordings


SHARED_TRAINING = {
    "student_model": STUDENT_MODEL,
    "teacher_model": TEACHER_MODEL,
    "teacher_decode": {
        "do_sample": False,
        "num_beams": 1,
        "language": "en",
        "task": "transcribe",
        "max_new_tokens": 200,
        "dtype": "float32",
    },
    "optimizer": {
        "name": "adamw",
        "learning_rate": 3e-5,
        "weight_decay": 0.1,
        "betas": [0.9, 0.999],
        "warmup_steps": 60,
    },
    "batch_size": 8,
    "freeze_encoder": True,
    "checkpoints": [SECONDARY_CHECKPOINT, PRIMARY_CHECKPOINT],
    "primary_checkpoint": PRIMARY_CHECKPOINT,
    "primary_normalizer": PRIMARY_NORMALIZER,
    "alternate_normalizer": ALTERNATE_NORMALIZER,
}


def build_run_specs(
    train_rows: Sequence[dict[str, Any]],
    *,
    seeds: Sequence[int] = PAIRED_SEEDS,
) -> list[dict[str, Any]]:
    filtered = construct_filtered_pool(train_rows)
    specs: list[dict[str, Any]] = []
    for seed in seeds:
        pools = {
            "filtered": ordered_recordings(filtered, seed=seed),
            "matched_random": ordered_recordings(
                construct_matched_random_pool(train_rows, filtered, seed=seed),
                seed=seed,
            ),
        }
        for pool_name in RECORDING_POOLS:
            recordings = pools[pool_name]
            recording_ids = [r["utt_id"] for r in recordings]
            for label in LABEL_SOURCES:
                specs.append(
                    {
                        "run_id": f"{pool_name}__{label}__seed{seed}",
                        "recording_pool": pool_name,
                        "label_source": label,
                        "seed": seed,
                        "pair_id": f"{pool_name}__seed{seed}",
                        "n_recordings": len(recordings),
                        "recording_ids": recording_ids,
                        "training": SHARED_TRAINING,
                    }
                )
    if len(specs) != len(seeds) * len(RECORDING_POOLS) * len(LABEL_SOURCES):
        raise RuntimeError("expected 12 runs for 3 paired seeds")
    return specs


def design_document(specs: Sequence[dict[str, Any]]) -> dict[str, Any]:
    return {
        "experiment": "label_source_2x2x3",
        "designated_before_launch": True,
        "n_runs": len(specs),
        "seeds": list(PAIRED_SEEDS),
        "recording_pools": list(RECORDING_POOLS),
        "label_sources": list(LABEL_SOURCES),
        "teacher_hyps_from": TEACHER_SYSTEM,
        "primary_checkpoint_updates": PRIMARY_CHECKPOINT,
        "secondary_checkpoint_updates": SECONDARY_CHECKPOINT,
        "primary_normalizer": PRIMARY_NORMALIZER,
        "alternate_normalizer": ALTERNATE_NORMALIZER,
        "pairing_contract": (
            "Within each (recording_pool, seed) pair the two label-source "
            "arms share initialization, recording order, greedy decoding, "
            "and optimizer settings. Do not retune one arm."
        ),
        "stop_rule": {
            "metric": "matched_random macro WER at 2400 updates",
            "teacher_minus_reference_mean_pp": STOP_RULE_MEAN_PP,
            "min_seeds_with_deterioration": STOP_RULE_MIN_SEEDS_WORSE,
            "alternate_normalizer_must_not_reverse": True,
            "interpretation": "practical screening threshold, not significance",
        },
        "held_out": [
            "SPIRE-SIES evaluation cohort",
            "L2-ARCTIC test set",
        ],
        "do_not_invent_a_new_loss": True,
        "training": SHARED_TRAINING,
        "run_ids": [s["run_id"] for s in specs],
    }
