"""Frozen design choices. Change these only by committing a new revision
before any of the 12 label-source runs launch — never after."""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
AUDIT_CSV = (
    REPO_ROOT
    / "audit"
    / "svarah_fairness_audit"
    / "results_whisper-large-v3_svarah_clean.csv"
)
DATA_DIR = REPO_ROOT / "data" / "research_reset"

# Sample construction / listening-set RNG. Distinct from the three paired
# training seeds so a later re-draw of the 120 cannot silently reshuffle
# optimizer state.
SAMPLE_SEED = 20260909

# Six L1 groups: three Indo-Aryan (Hindi belt, eastern, western), two
# Dravidian literary languages, and Svarah's only Sino-Tibetan language.
# Bodo remains a small group and is reported as descriptive in Days 4–10.
L1_GROUPS: tuple[str, ...] = (
    "Hindi",
    "Bengali",
    "Marathi",
    "Tamil",
    "Telugu",
    "Bodo",
)
FAMILY_OF: dict[str, str] = {
    "Hindi": "Indo-Aryan",
    "Bengali": "Indo-Aryan",
    "Marathi": "Indo-Aryan",
    "Tamil": "Dravidian",
    "Telugu": "Dravidian",
    "Bodo": "Sino-Tibetan",
}
PER_CELL = 10  # accepted + rejected per L1 → 120 clips
N_LISTENING = len(L1_GROUPS) * 2 * PER_CELL

# Teacher whose transcripts define accepted/rejected for the listening set
# and the label-source experiment. Matches the already-scored audit CSV.
TEACHER_SYSTEM = "whisper-large-v3"
STUDENT_MODEL = "openai/whisper-tiny"
TEACHER_MODEL = "openai/whisper-large-v3"

# Three paired seeds → 12 runs (2 pools × 2 label sources × 3 seeds).
# Within a (pool, seed) pair, reference vs teacher share initialization,
# recording order, decoding, and optimizer settings.
PAIRED_SEEDS: tuple[int, ...] = (12, 34, 56)

# Designated before launch. 2,400 is primary; 600 is a diagnostic snapshot.
SECONDARY_CHECKPOINT = 600
PRIMARY_CHECKPOINT = 2400

PRIMARY_NORMALIZER = "capstone"
ALTERNATE_NORMALIZER = "whisper"

# Practical screening threshold, not statistical significance.
STOP_RULE_MEAN_PP = 0.5
STOP_RULE_MIN_SEEDS_WORSE = 2

# Gate-only split of the already-scored 2,500 Svarah IDs (after removing
# the 120 listening clips). Days 4–10 rebuild train/dev with verified
# speaker identities; SPIRE-SIES and L2-ARCTIC stay untouched.
GATE_EVAL_FRACTION = 0.20

RECORDING_POOLS: tuple[str, ...] = ("filtered", "matched_random")
LABEL_SOURCES: tuple[str, ...] = ("reference", "teacher")
