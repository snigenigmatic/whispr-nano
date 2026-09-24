"""Days 4–10 run only if the Days 1–3 stop rule fires.

Do not invent a new loss. If ordinary pseudo-label adaptation, uDistil-Whisper
filtering, or Pseudo2Real closes the gap, record that and stop.
"""

from __future__ import annotations

COMPARISON_ARMS = (
    {
        "name": "pseudo_label_sft",
        "description": "Ordinary teacher-transcript adaptation, no filtering.",
        "privileged": False,
    },
    {
        "name": "udistil_confidence_filter",
        "description": (
            "Label-free / confidence-informed filtering of teacher transcripts "
            "(Waheed et al., NAACL 2025, uDistil-Whisper)."
        ),
        "privileged": False,
    },
    {
        "name": "pseudo2real",
        "description": (
            "Task-arithmetic correction vector from source real vs pseudo "
            "(Lin et al., ACL Findings 2026)."
        ),
        "privileged": False,
    },
    {
        "name": "reference_sft",
        "description": "Reference-supervised adaptation. Privileged diagnostic, not a fair competitor.",
        "privileged": True,
    },
)

FROZEN_TEST_SETS = (
    "SPIRE-SIES prepared evaluation cohort",
    "L2-ARCTIC test set",
)

RULES = (
    "Match initialization, training exposure, and tuning allowances across fair arms.",
    "Account source annotations and target annotations separately.",
    "Report every supported L1; mark small groups (including Bodo) as descriptive.",
    "Inspect whether remaining errors recur across speakers and concern spoken content.",
    "Do not invent a new loss. Any extension must address a demonstrated remaining failure.",
    "Test results do not trigger further tuning.",
)
