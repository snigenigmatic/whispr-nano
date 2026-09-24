"""Validate and summarize exports from the blinded listening-review v2 UI.

This module deliberately reads only the public packet.  In particular, it does
not open ``sealed_mapping.json``: reviewer imports and their derived artifacts
must remain blind to model identities and prediction metrics.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_PACKET = ROOT / "research/bridge_open/listening_review_v2/review_items.json"
DEFAULT_METADATA = ROOT / "research/bridge_open/listening_review_v2/packet_metadata.json"
LANGUAGES = frozenset({"hi", "bn", "as"})
EXPECTED_LANGUAGE_COUNTS = {"hi": 18, "bn": 16, "as": 16}
EXPECTED_ITEM_COUNT = 50
EXPECTED_CARD_COUNT = 168
ALLOWED_REFERENCE = frozenset({"yes", "no", "uncertain"})
ALLOWED_AUDIO = frozenset({"clear", "usable", "unclear"})
ALLOWED_BINARY = frozenset({"yes", "no", "uncertain"})
ALLOWED_MEANING = frozenset({"yes", "partly", "no", "uncertain"})
STATUS_UNREVIEWED = "UNREVIEWED"
STATUS_IN_PROGRESS = "IN_PROGRESS"
STATUS_REVIEWED = "REVIEWED"


class ReviewImportError(ValueError):
    """An export cannot be accepted as a blinded-review response."""


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _canonical_sha256(value: Any) -> str:
    return _sha256_bytes(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    )


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ReviewImportError(message)


def _load_json(path: Path) -> tuple[Any, bytes]:
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise ReviewImportError(f"cannot read {path}: {exc}") from exc
    try:
        return json.loads(raw), raw
    except json.JSONDecodeError as exc:
        raise ReviewImportError(f"invalid JSON in {path}: {exc}") from exc


def _validate_value(value: Any, allowed: frozenset[str], field: str, review_id: str) -> str:
    _require(isinstance(value, str), f"{review_id}: {field} must be a string")
    _require(value in allowed or value == "", f"{review_id}: invalid {field!r} value {value!r}")
    return value


def _response_has_content(response: dict[str, Any]) -> bool:
    if response["reference_correct"] or response["audio_quality"] or response["notes"]:
        return True
    return any(
        card[field]
        for card in response["systems"]
        for field in ("language_correct", "script_correct", "meaning_preserved")
    )


def recompute_status(response: dict[str, Any], labels: Iterable[str]) -> str:
    """Apply the UI's completion rule without trusting its exported status."""
    by_label = {card["system_label"]: card for card in response["systems"]}
    complete = bool(response["reference_correct"] and response["audio_quality"]) and all(
        by_label[label]["language_correct"]
        and by_label[label]["script_correct"]
        and by_label[label]["meaning_preserved"]
        for label in labels
    )
    if complete:
        return STATUS_REVIEWED
    return STATUS_IN_PROGRESS if _response_has_content(response) else STATUS_UNREVIEWED


def load_and_validate_packet(packet_path: Path = DEFAULT_PACKET, metadata_path: Path = DEFAULT_METADATA) -> list[dict[str, Any]]:
    """Load the frozen public packet and check its fixed cohort shape."""
    packet, _ = _load_json(packet_path)
    metadata, _ = _load_json(metadata_path)
    _require(isinstance(packet, list), "packet must be a JSON array")
    _require(len(packet) == EXPECTED_ITEM_COUNT, f"packet must contain {EXPECTED_ITEM_COUNT} items")
    _require(metadata.get("packet_status") == STATUS_UNREVIEWED, "packet is not the frozen UNREVIEWED packet")
    _require(metadata.get("fixed_cohort_count") == EXPECTED_ITEM_COUNT, "packet metadata has wrong fixed cohort count")
    _require(metadata.get("counts_by_language") == EXPECTED_LANGUAGE_COUNTS, "packet metadata has wrong language counts")

    ids: set[str] = set()
    card_count = 0
    observed_languages: Counter[str] = Counter()
    for item in packet:
        _require(isinstance(item, dict), "packet item must be an object")
        review_id = item.get("review_id")
        language = item.get("language")
        systems = item.get("systems")
        _require(isinstance(review_id, str) and review_id, "packet has a missing review_id")
        _require(review_id not in ids, f"packet has duplicate review_id {review_id!r}")
        _require(language in LANGUAGES, f"{review_id}: unsupported packet language {language!r}")
        _require(isinstance(systems, list) and systems, f"{review_id}: packet has no anonymous cards")
        labels = [card.get("system_label") if isinstance(card, dict) else None for card in systems]
        _require(all(isinstance(label, str) and label for label in labels), f"{review_id}: invalid anonymous card label")
        _require(len(labels) == len(set(labels)), f"{review_id}: duplicate anonymous card label in packet")
        _require(item.get("review_status") == STATUS_UNREVIEWED, f"{review_id}: packet must remain UNREVIEWED")
        ids.add(review_id)
        observed_languages[language] += 1
        card_count += len(systems)
    _require(dict(observed_languages) == EXPECTED_LANGUAGE_COUNTS, "packet does not have the fixed 18/16/16 language split")
    _require(card_count == EXPECTED_CARD_COUNT, f"packet must contain {EXPECTED_CARD_COUNT} anonymous cards")
    return packet


def _normalize_record(record: Any, packet_item: dict[str, Any]) -> dict[str, Any]:
    review_id = packet_item["review_id"]
    _require(isinstance(record, dict), f"{review_id}: export record must be an object")
    _require(record.get("review_id") == review_id, f"{review_id}: wrong review_id")
    _require(record.get("language") == packet_item["language"], f"{review_id}: wrong language")
    _require(isinstance(record.get("notes", ""), str), f"{review_id}: notes must be a string")
    systems = record.get("systems")
    _require(isinstance(systems, list), f"{review_id}: systems must be an array")
    expected_labels = [card["system_label"] for card in packet_item["systems"]]
    supplied_labels = [card.get("system_label") if isinstance(card, dict) else None for card in systems]
    _require(len(supplied_labels) == len(set(supplied_labels)), f"{review_id}: duplicate system_label")
    _require(set(supplied_labels) == set(expected_labels), f"{review_id}: anonymous card labels do not exactly match packet")

    normalized_cards: list[dict[str, str]] = []
    cards_by_label = {card["system_label"]: card for card in systems}
    for label in expected_labels:
        card = cards_by_label[label]
        normalized_cards.append(
            {
                "system_label": label,
                "language_correct": _validate_value(card.get("language_correct", ""), ALLOWED_BINARY, "language_correct", review_id),
                "script_correct": _validate_value(card.get("script_correct", ""), ALLOWED_BINARY, "script_correct", review_id),
                "meaning_preserved": _validate_value(card.get("meaning_preserved", ""), ALLOWED_MEANING, "meaning_preserved", review_id),
            }
        )
    normalized = {
        "review_id": review_id,
        "language": packet_item["language"],
        "reference_correct": _validate_value(record.get("reference_correct", ""), ALLOWED_REFERENCE, "reference_correct", review_id),
        "audio_quality": _validate_value(record.get("audio_quality", ""), ALLOWED_AUDIO, "audio_quality", review_id),
        "notes": record.get("notes", ""),
        "systems": normalized_cards,
    }
    expected_status = recompute_status(normalized, expected_labels)
    _require(record.get("review_status") == expected_status, f"{review_id}: stale or false review_status (expected {expected_status})")
    normalized["review_status"] = expected_status
    return normalized


def validate_export(export: Any, packet: list[dict[str, Any]], reviewer_languages: set[str]) -> list[dict[str, Any]]:
    """Validate a full UI JSON download, retaining blanks as pending records."""
    _require(isinstance(export, list), "review export must be a JSON array")
    _require(len(export) == len(packet) == EXPECTED_ITEM_COUNT, "review export must include all 50 records, including blanks")
    _require(reviewer_languages and reviewer_languages <= LANGUAGES, "--languages must be one or more of: hi,bn,as")
    by_id = {item["review_id"]: item for item in packet}
    supplied_ids = [row.get("review_id") if isinstance(row, dict) else None for row in export]
    _require(len(supplied_ids) == len(set(supplied_ids)), "review export has duplicate review_id")
    _require(set(supplied_ids) == set(by_id), "review export review_ids do not exactly match the frozen packet")
    normalized = [_normalize_record(row, by_id[row["review_id"]]) for row in export]
    for record in normalized:
        _require(
            record["review_status"] != STATUS_REVIEWED or record["language"] in reviewer_languages,
            f"{record['review_id']}: completed review is outside declared reviewer competency ({record['language']})",
        )
    return normalized


def _answer_breakdown(values: Iterable[str], positive: str = "yes") -> dict[str, Any]:
    counts = Counter(values)
    evaluable = counts["yes"] + counts["no"] + counts.get("partly", 0)
    return {
        "yes": counts["yes"],
        "no": counts["no"],
        "partly": counts.get("partly", 0),
        "uncertain": counts["uncertain"],
        "evaluable": evaluable,
        "yes_rate_among_evaluable": (counts[positive] / evaluable) if evaluable else None,
    }


def build_summary(records: list[dict[str, Any]], packet: list[dict[str, Any]], reviewer: str, reviewer_languages: set[str]) -> dict[str, Any]:
    """Produce descriptive, reviewed-only results without unblinding any card."""
    packet_by_id = {item["review_id"]: item for item in packet}
    reviewed = [record for record in records if record["review_status"] == STATUS_REVIEWED]
    pending = [record for record in records if record["review_status"] != STATUS_REVIEWED]
    by_label: dict[str, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))
    for record in reviewed:
        for card in record["systems"]:
            for field in ("language_correct", "script_correct", "meaning_preserved"):
                by_label[card["system_label"]][field].append(card[field])
    card_analysis = {
        label: {field: _answer_breakdown(values) for field, values in sorted(fields.items())}
        for label, fields in sorted(by_label.items())
    }
    reviewed_selection = Counter(
        kind for record in reviewed for kind in packet_by_id[record["review_id"]]["selection_kinds"]
    )
    return {
        "schema_version": "blinded-review-import-v1",
        "reviewer_pseudonym": reviewer,
        "reviewer_competency_languages": sorted(reviewer_languages),
        "counts": {
            "items_total": len(records),
            "items_reviewed": len(reviewed),
            "items_pending": len(pending),
            "items_unreviewed": sum(r["review_status"] == STATUS_UNREVIEWED for r in records),
            "items_in_progress": sum(r["review_status"] == STATUS_IN_PROGRESS for r in records),
            "reviewed_by_language": dict(sorted(Counter(r["language"] for r in reviewed).items())),
            "reviewed_selection_kinds": dict(sorted(reviewed_selection.items())),
        },
        "analysis_scope": {
            "included_status": STATUS_REVIEWED,
            "excluded_statuses": [STATUS_UNREVIEWED, STATUS_IN_PROGRESS],
            "sampling_note": (
                "Reviewed records come from a fixed mixture of random-stratified and targeted-review-queue "
                "selections. These descriptive reviewed-record results are not aggregate population estimates."
            ),
            "uncertain_and_no_handling": (
                "Uncertain answers are excluded from evaluable denominators and reported separately; no answers "
                "remain evaluable negative judgments and are also reported separately."
            ),
        },
        "reviewed_item_outcomes": {
            "reference_correct": _answer_breakdown(r["reference_correct"] for r in reviewed),
            "audio_quality": dict(sorted(Counter(r["audio_quality"] for r in reviewed).items())),
        },
        "anonymous_card_outcomes": card_analysis,
    }


def _safe_reviewer_stem(reviewer: str) -> str:
    stem = "".join(character if character.isalnum() or character in "-_" else "_" for character in reviewer)
    _require(stem and stem.strip("_"), "--reviewer must contain a usable pseudonym")
    return stem


def create_artifacts(
    export_path: Path,
    reviewer: str,
    reviewer_languages: set[str],
    output_dir: Path,
    packet_path: Path = DEFAULT_PACKET,
    metadata_path: Path = DEFAULT_METADATA,
) -> tuple[Path, Path]:
    """Validate an export and create one immutable receipt and summary pair."""
    _require(isinstance(reviewer, str) and reviewer.strip(), "--reviewer pseudonym is required")
    packet = load_and_validate_packet(packet_path, metadata_path)
    export, export_bytes = _load_json(export_path)
    records = validate_export(export, packet, reviewer_languages)
    input_sha256 = _sha256_bytes(export_bytes)
    packet_data_sha256 = _canonical_sha256(packet)
    summary = build_summary(records, packet, reviewer.strip(), reviewer_languages)
    receipt = {
        "schema_version": "blinded-review-import-v1",
        "reviewer_pseudonym": reviewer.strip(),
        "reviewer_competency_languages": sorted(reviewer_languages),
        "input_file": export_path.name,
        "input_sha256": input_sha256,
        "packet_data_sha256": packet_data_sha256,
        "packet_item_count": len(packet),
        "packet_anonymous_card_count": sum(len(item["systems"]) for item in packet),
        "reviewed_response_count": summary["counts"]["items_reviewed"],
        "pending_response_count": summary["counts"]["items_pending"],
        "validated_records": records,
        "sealed_mapping_included": False,
    }
    summary["input_sha256"] = input_sha256
    summary["packet_data_sha256"] = packet_data_sha256
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{_safe_reviewer_stem(reviewer.strip())}_{input_sha256[:12]}"
    receipt_path = output_dir / f"{stem}_receipt.json"
    summary_path = output_dir / f"{stem}_summary.json"
    _require(not receipt_path.exists() and not summary_path.exists(), "create-only artifacts already exist for this reviewer and input")
    try:
        with receipt_path.open("x", encoding="utf-8") as handle:
            json.dump(receipt, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        with summary_path.open("x", encoding="utf-8") as handle:
            json.dump(summary, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
    except Exception:
        # Keep a failed two-file write from looking like a valid completed receipt.
        if receipt_path.exists() and not summary_path.exists():
            receipt_path.unlink()
        raise
    return receipt_path, summary_path


def _parse_languages(value: str) -> set[str]:
    return {language.strip() for language in value.split(",") if language.strip()}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate a blinded listening-review v2 JSON export.")
    parser.add_argument("--input", required=True, type=Path, help="JSON downloaded from the v2 review UI")
    parser.add_argument("--reviewer", required=True, help="reviewer pseudonym; no real name")
    parser.add_argument("--languages", required=True, help="comma-separated languages the reviewer is competent to judge: hi,bn,as")
    parser.add_argument("--output-dir", required=True, type=Path, help="new directory for create-only receipt and summary")
    parser.add_argument("--packet", type=Path, default=DEFAULT_PACKET)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    args = parser.parse_args(argv)
    try:
        receipt, summary = create_artifacts(
            args.input, args.reviewer, _parse_languages(args.languages), args.output_dir, args.packet, args.metadata
        )
    except ReviewImportError as exc:
        parser.error(str(exc))
    print(json.dumps({"receipt": str(receipt), "summary": str(summary)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
