"""Days 1–3 research reset: listening protocol, partitions, label-source gate.

The candidate question is a hypothesis, not a novelty claim:

    When do corrections learned from a teacher's mistakes transfer to
    Indian-English speakers, and when do they fail?

Nothing in this package invents a new loss. The 12-run label-source
experiment is a screening gate. Existing audit and OPD numbers are
exploratory background.
"""

from .constants import (
    ALTERNATE_NORMALIZER,
    L1_GROUPS,
    PAIRED_SEEDS,
    PRIMARY_CHECKPOINT,
    PRIMARY_NORMALIZER,
    SAMPLE_SEED,
    SECONDARY_CHECKPOINT,
)

__all__ = [
    "ALTERNATE_NORMALIZER",
    "L1_GROUPS",
    "PAIRED_SEEDS",
    "PRIMARY_CHECKPOINT",
    "PRIMARY_NORMALIZER",
    "SAMPLE_SEED",
    "SECONDARY_CHECKPOINT",
]
