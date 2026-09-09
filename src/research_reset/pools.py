"""Filtered vs matched-random recording pools.

Filtered: teacher transcript matches the reference under the primary
(capstone) normalizer — Distil-Whisper-style keep-if-clean.

Matched-random: the same number of recordings, matched on primary
language and duration quartile, drawn from the full train pool without
regard to teacher-reference agreement.

Label source is applied later: the same recordings are paired with
either the dataset reference or the teacher hypothesis.
"""

from __future__ import annotations

from typing import Any, Sequence

from .constants import PRIMARY_NORMALIZER
from .scoring import get_normalizer


def is_filtered(row: dict[str, Any], *, normalizer: str = PRIMARY_NORMALIZER) -> bool:
    if normalizer == "capstone" and "accepted" in row:
        return bool(row["accepted"])
    norm = get_normalizer(normalizer)
    return norm(row["reference_text"]) == norm(row["teacher_text"] if "teacher_text" in row else row["hypothesis_text"])


def _duration_quartiles(rows: Sequence[dict[str, Any]]) -> dict[str, int]:
    durations = sorted(float(r["duration_s"]) for r in rows)
    if not durations:
        return {}
    n = len(durations)

    def q(p: float) -> float:
        i = min(n - 1, max(0, int(round((n - 1) * p))))
        return durations[i]

    cuts = (q(0.25), q(0.50), q(0.75))

    def bin_of(dur: float) -> int:
        if dur <= cuts[0]:
            return 0
        if dur <= cuts[1]:
            return 1
        if dur <= cuts[2]:
            return 2
        return 3

    return {r["utt_id"]: bin_of(float(r["duration_s"])) for r in rows}


def construct_filtered_pool(train_rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    pool = [r for r in train_rows if is_filtered(r)]
    return sorted(pool, key=lambda r: r["utt_id"])


def construct_matched_random_pool(
    train_rows: Sequence[dict[str, Any]],
    filtered: Sequence[dict[str, Any]],
    *,
    seed: int,
) -> list[dict[str, Any]]:
    """Same n as `filtered`, matched on (language, duration quartile).

    The RNG is the training seed so each paired seed gets its own
    matched-random draw, while reference/teacher arms of that seed share it.
    """
    import numpy as np

    bins = _duration_quartiles(train_rows)
    filtered_keys = [
        (r["primary_language"], bins[r["utt_id"]]) for r in filtered
    ]
    by_key: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for row in train_rows:
        key = (row["primary_language"], bins[row["utt_id"]])
        by_key.setdefault(key, []).append(row)

    rng = np.random.default_rng(seed)
    used: set[str] = set()
    chosen: list[dict[str, Any]] = []
    for lang, qbin in filtered_keys:
        pick = _draw_match(by_key, lang, qbin, used, rng)
        used.add(pick["utt_id"])
        chosen.append(pick)
    return chosen


def _draw_match(by_key, lang: str, qbin: int, used: set[str], rng) -> dict[str, Any]:
    for delta in (0, -1, 1, -2, 2, -3, 3):
        key = (lang, qbin + delta)
        if key not in by_key:
            continue
        available = [r for r in by_key[key] if r["utt_id"] not in used]
        if available:
            j = int(rng.integers(0, len(available)))
            return available[j]
    # Last resort: any unused clip of the same language.
    leftover = [
        r
        for key, rows in by_key.items()
        if key[0] == lang
        for r in rows
        if r["utt_id"] not in used
    ]
    if not leftover:
        raise ValueError(f"cannot match a clip for language={lang} quartile={qbin}")
    j = int(rng.integers(0, len(leftover)))
    return leftover[j]


def ordered_recordings(rows: Sequence[dict[str, Any]], *, seed: int) -> list[dict[str, Any]]:
    """Permutation used for both label sources of a paired seed."""
    import numpy as np

    rng = np.random.default_rng(seed + 1_000_003)
    idx = rng.permutation(len(rows))
    return [rows[int(i)] for i in idx]
