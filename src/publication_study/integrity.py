"""Dataset boundaries and deterministic balanced-target selection."""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter, defaultdict
from typing import Any, Iterable, Mapping, Sequence


REFERENCE_FIELDS = frozenset({"reference", "reference_text", "sentence", "text", "transcript"})
TARGET_ACCENTS = ("Bengali", "Hindi", "Kannada", "Marathi", "Tamil", "Telugu")


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


def validate_target_teacher_artifacts(
    target_rows: Sequence[Mapping[str, Any]],
    pseudo_rows: Sequence[Mapping[str, Any]],
    audit_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Validate exact prediction coverage and the deterministic blind audit."""
    if len(target_rows) != 4800 or len(pseudo_rows) != 4800 or len(audit_rows) != 120:
        raise ValueError("target teacher artifacts have unexpected row counts")

    def unique_index(rows: Sequence[Mapping[str, Any]], name: str) -> dict[str, Mapping[str, Any]]:
        indexed: dict[str, Mapping[str, Any]] = {}
        for row in rows:
            uid = str(row.get("utt_id", ""))
            if not uid or uid in indexed:
                raise ValueError(f"{name} contains an empty or duplicate utterance ID")
            indexed[uid] = row
        return indexed

    targets = unique_index(target_rows, "target manifest")
    predictions = unique_index(pseudo_rows, "pseudo-label manifest")
    if set(targets) != set(predictions):
        raise ValueError("pseudo-label IDs do not exactly match the target manifest")
    identity_fields = (
        "speaker_id", "accent", "audio_path", "audio_sha256", "duration_s",
        "source_sample_rate", "source_dataset_split",
    )
    for uid, row in predictions.items():
        leaked = REFERENCE_FIELDS & set(row)
        if leaked:
            raise ValueError(f"pseudo-label row contains corpus reference fields: {sorted(leaked)}")
        if any(row.get(key) != targets[uid].get(key) for key in identity_fields):
            raise ValueError(f"pseudo-label metadata differs from target manifest: {uid}")
        if not isinstance(row.get("pseudo_text"), str) or not row["pseudo_text"].strip():
            raise ValueError(f"empty target pseudo-label: {uid}")
        mean = row.get("teacher_mean_logprob")
        minimum = row.get("teacher_min_logprob")
        if not all(isinstance(value, (int, float)) and math.isfinite(value) for value in (mean, minimum)):
            raise ValueError(f"invalid target confidence: {uid}")
        if minimum > mean or mean > 1e-6:
            raise ValueError(f"inconsistent target confidence: {uid}")
        token_ids = row.get("teacher_generated_token_ids")
        if (
            not isinstance(row.get("teacher_token_count"), int)
            or row["teacher_token_count"] <= 0
            or not isinstance(token_ids, list)
            or not token_ids
            or any(not isinstance(token, int) for token in token_ids)
        ):
            raise ValueError(f"invalid target teacher token receipt: {uid}")

    audit = unique_index(audit_rows, "blind audit")
    if set(audit) - set(predictions):
        raise ValueError("blind audit contains IDs outside the target predictions")
    forbidden_audit = REFERENCE_FIELDS | frozenset(
        {"pseudo_text", "teacher_generated_token_ids", "label_source"}
    )
    for uid, row in audit.items():
        leaked = forbidden_audit & set(row)
        if leaked:
            raise ValueError(f"blind audit leaks model or reference text: {sorted(leaked)}")
        prediction = predictions[uid]
        if row.get("accent") != prediction.get("accent"):
            raise ValueError(f"blind audit accent differs from prediction: {uid}")
        if row.get("teacher_mean_logprob") != prediction.get("teacher_mean_logprob"):
            raise ValueError(f"blind audit confidence differs from prediction: {uid}")

    expected_ids: set[str] = set()
    strata = Counter()
    for accent in TARGET_ACCENTS:
        group = sorted(
            (row for row in pseudo_rows if row["accent"] == accent),
            key=lambda row: (row["teacher_mean_logprob"], row["utt_id"]),
        )
        if len(group) != 800:
            raise ValueError(f"target pseudo-label balance changed for {accent}")
        low = {row["utt_id"] for row in group[:10]}
        high = {row["utt_id"] for row in group[-10:]}
        expected_ids.update(low | high)
        for uid in low:
            strata[(accent, "low")] += int(audit.get(uid, {}).get("confidence_stratum") == "low")
        for uid in high:
            strata[(accent, "high")] += int(audit.get(uid, {}).get("confidence_stratum") == "high")
    if set(audit) != expected_ids or any(
        strata[(accent, stratum)] != 10
        for accent in TARGET_ACCENTS for stratum in ("low", "high")
    ):
        raise ValueError("blind audit is not the exact high/low confidence selection")
    return {
        "target_rows": len(targets), "pseudo_rows": len(predictions),
        "audit_rows": len(audit), "pseudo_id_coverage": 1.0,
        "audit_per_accent_stratum": 10, "reference_fields": 0,
    }


def manifest_sha256(rows: Sequence[Mapping[str, Any]]) -> str:
    payload = json.dumps(
        list(rows), sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()
