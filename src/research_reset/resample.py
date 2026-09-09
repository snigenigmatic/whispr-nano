"""Speaker-level bootstrap for Days 11–14 uncertainty.

Training-seed variation is reported separately. This helper resamples
eval *speakers* (or, when speaker ids are missing, utterance ids with a
loud warning) and recomputes macro WER.
"""

from __future__ import annotations

from typing import Any, Callable, Sequence


def resample_groups(
    rows: Sequence[dict[str, Any]],
    *,
    speaker_key: str = "speaker_id",
    n_boot: int = 1000,
    seed: int = 20260909,
    metric: Callable[[list[dict[str, Any]]], float],
) -> dict[str, Any]:
    import numpy as np

    if not rows:
        raise ValueError("no rows to resample")
    have_speakers = all(speaker_key in r and r[speaker_key] not in (None, "") for r in rows)
    key = speaker_key if have_speakers else "utt_id"
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(str(row[key]), []).append(row)
    ids = sorted(groups)
    rng = np.random.default_rng(seed)
    scores = []
    for _ in range(n_boot):
        draw = rng.choice(ids, size=len(ids), replace=True)
        boot_rows = [item for i in draw for item in groups[str(i)]]
        scores.append(float(metric(boot_rows)))
    arr = np.asarray(scores, dtype=float)
    return {
        "unit": "speaker" if have_speakers else "utterance",
        "warning": None
        if have_speakers
        else "speaker_id missing; resampled utterances. Replace before the Days 11–14 memo.",
        "n_units": len(ids),
        "n_boot": n_boot,
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=1)) if len(arr) > 1 else 0.0,
        "ci95": [float(np.quantile(arr, 0.025)), float(np.quantile(arr, 0.975))],
    }
