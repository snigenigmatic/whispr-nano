"""Screening stop rule for the 12-run label-source experiment.

Continue to Days 4–10 only if, on the matched-random pool, teacher
targets worsen macro WER relative to reference targets by at least 0.5
percentage points on average at 2,400 updates, the deterioration appears
in at least two of the three paired seeds, and the alternate (Whisper)
normalizer does not reverse the sign of the mean difference.

This is a practical screen, not a significance test.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from .constants import (
    ALTERNATE_NORMALIZER,
    PAIRED_SEEDS,
    PRIMARY_CHECKPOINT,
    PRIMARY_NORMALIZER,
    STOP_RULE_MEAN_PP,
    STOP_RULE_MIN_SEEDS_WORSE,
)


def _macro(results: Mapping[str, Any], run_id: str, *, checkpoint: int, normalizer: str) -> float:
    run = results[run_id]
    key = f"macro_wer@{checkpoint}:{normalizer}"
    if key not in run:
        raise KeyError(f"{run_id} missing {key}")
    return float(run[key])


def paired_deltas(
    results: Mapping[str, Any],
    *,
    seeds: Sequence[int] = PAIRED_SEEDS,
    checkpoint: int = PRIMARY_CHECKPOINT,
    normalizer: str = PRIMARY_NORMALIZER,
    pool: str = "matched_random",
) -> dict[int, float]:
    """Teacher − reference macro WER, in percentage points, per seed."""
    out: dict[int, float] = {}
    for seed in seeds:
        teacher = _macro(
            results, f"{pool}__teacher__seed{seed}", checkpoint=checkpoint, normalizer=normalizer
        )
        reference = _macro(
            results, f"{pool}__reference__seed{seed}", checkpoint=checkpoint, normalizer=normalizer
        )
        out[int(seed)] = (teacher - reference) * 100.0
    return out


def apply_stop_rule(
    results: Mapping[str, Any],
    *,
    seeds: Sequence[int] = PAIRED_SEEDS,
    checkpoint: int = PRIMARY_CHECKPOINT,
) -> dict[str, Any]:
    primary = paired_deltas(
        results, seeds=seeds, checkpoint=checkpoint, normalizer=PRIMARY_NORMALIZER
    )
    alternate = paired_deltas(
        results, seeds=seeds, checkpoint=checkpoint, normalizer=ALTERNATE_NORMALIZER
    )
    mean_primary = sum(primary.values()) / len(primary)
    mean_alternate = sum(alternate.values()) / len(alternate)
    n_worse = sum(1 for d in primary.values() if d > 0)
    mean_ok = mean_primary >= STOP_RULE_MEAN_PP
    seeds_ok = n_worse >= STOP_RULE_MIN_SEEDS_WORSE
    no_reversal = mean_alternate > 0
    continue_days_4_10 = bool(mean_ok and seeds_ok and no_reversal)
    return {
        "continue_days_4_10": continue_days_4_10,
        "checkpoint": checkpoint,
        "primary_normalizer": PRIMARY_NORMALIZER,
        "alternate_normalizer": ALTERNATE_NORMALIZER,
        "mean_delta_pp_primary": mean_primary,
        "mean_delta_pp_alternate": mean_alternate,
        "per_seed_delta_pp_primary": {str(k): v for k, v in primary.items()},
        "per_seed_delta_pp_alternate": {str(k): v for k, v in alternate.items()},
        "n_seeds_with_deterioration": n_worse,
        "thresholds": {
            "mean_pp": STOP_RULE_MEAN_PP,
            "min_seeds_worse": STOP_RULE_MIN_SEEDS_WORSE,
            "alternate_mean_must_be_positive": True,
        },
        "decision": (
            "proceed to Days 4–10"
            if continue_days_4_10
            else "close this mechanism under the tested conditions; keep the technical report"
        ),
    }
