"""Dataset boundaries and deterministic balanced-target selection."""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from typing import Any, Iterable, Mapping, Sequence


REFERENCE_FIELDS = frozenset({"reference", "reference_text", "sentence", "text", "transcript"})


def stable_rank(seed: int, uid: str) -> str:
    return hashlib.sha256(f"{seed}\0{uid}".encode("utf-8")).hexdigest()


def balanced_target_rows(
    rows: Iterable[Mapping[str, Any]], *, total: int, seed: int = 12
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Choose the same count per accent using a deterministic content hash.

    Input rows must already contain only metadata plus an audio handle/path.
    Reference-like fields are rejected so target labels cannot leak into the
    adaptation manifest.
    """
    if total <= 0:
        raise ValueError("total must be positive")
    by_group: dict[str, list[dict[str, Any]]] = defaultdict(list)
    seen: set[str] = set()
    for source in rows:
        leaked = REFERENCE_FIELDS & set(source)
        if leaked:
            raise ValueError(f"target row contains reference-like fields: {sorted(leaked)}")
        row = dict(source)
        uid = str(row.get("utt_id", "")).strip()
        speaker = str(row.get("speaker_id", "")).strip()
        accent = str(row.get("accent", "")).strip()
        if not uid or not speaker or not accent:
            raise ValueError("target rows require utt_id, speaker_id, and accent")
        if uid in seen:
            raise ValueError(f"duplicate target utterance: {uid}")
        seen.add(uid)
        by_group[accent].append(row)
    if len(by_group) < 2:
        raise ValueError("balanced target selection requires multiple accent groups")
    per_group = total // len(by_group)
    if per_group <= 0:
        raise ValueError("target total is smaller than group count")
    for group, group_rows in by_group.items():
        if len(group_rows) < per_group:
            raise ValueError(f"insufficient target rows for {group}: {len(group_rows)} < {per_group}")
    selected: list[dict[str, Any]] = []
    for group in sorted(by_group):
        selected.extend(sorted(by_group[group], key=lambda r: stable_rank(seed, r["utt_id"]))[:per_group])
    selected.sort(key=lambda r: r["utt_id"])
    report = {
        "seed": seed,
        "requested_total": total,
        "selected_total": len(selected),
        "per_group": per_group,
        "available_by_accent": dict(sorted(Counter(str(r["accent"]) for rows_ in by_group.values() for r in rows_).items())),
        "selected_by_accent": dict(sorted(Counter(str(r["accent"]) for r in selected).items())),
        "selected_speakers": len({r["speaker_id"] for r in selected}),
    }
    return selected, report


def validate_speaker_boundary(
    target_train: Sequence[Mapping[str, Any]], sealed_test: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    train_speakers = {str(row["speaker_id"]) for row in target_train}
    test_speakers = {str(row["speaker_id"]) for row in sealed_test}
    overlap = sorted(train_speakers & test_speakers)
    if overlap:
        raise ValueError(f"SPIRE train/test speaker overlap: {overlap[:10]}")
    train_ids = {str(row["utt_id"]) for row in target_train}
    test_ids = {str(row["utt_id"]) for row in sealed_test}
    if train_ids & test_ids:
        raise ValueError("SPIRE train/test utterance overlap")
    return {
        "target_train_utterances": len(train_ids),
        "target_train_speakers": len(train_speakers),
        "sealed_test_utterances": len(test_ids),
        "sealed_test_speakers": len(test_speakers),
        "speaker_overlap": 0,
        "utterance_overlap": 0,
    }


def manifest_sha256(rows: Sequence[Mapping[str, Any]]) -> str:
    payload = json.dumps(
        list(rows), sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()
