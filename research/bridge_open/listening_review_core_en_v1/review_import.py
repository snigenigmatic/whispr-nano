"""Create-only validator for the separate 12-item English resource-panel export.

It deliberately consumes only ``review_items.json``.  The model identity
mapping remains sealed and is never loaded or emitted by this importer.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable


PACKET = Path(__file__).with_name("review_items.json")
ITEM_COUNT = 12
CARD_COUNT = 72
UNREVIEWED = "UNREVIEWED"
IN_PROGRESS = "IN_PROGRESS"
REVIEWED = "REVIEWED"
REFERENCE = frozenset({"yes", "no", "uncertain"})
AUDIO = frozenset({"clear", "usable", "unclear"})
BINARY = frozenset({"yes", "no", "uncertain"})
MEANING = frozenset({"yes", "partly", "no", "uncertain"})


class ImportError(ValueError):
    """An English resource-panel export is not acceptable for import."""


def _require(ok: bool, message: str) -> None:
    if not ok:
        raise ImportError(message)


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _canonical_sha256(value: Any) -> str:
    return _sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8"))


def _load(path: Path) -> tuple[Any, bytes]:
    raw = path.read_bytes()
    try:
        return json.loads(raw), raw
    except json.JSONDecodeError as exc:
        raise ImportError(f"invalid JSON in {path}: {exc}") from exc


def load_packet(path: Path = PACKET) -> list[dict[str, Any]]:
    packet, _ = _load(path)
    _require(isinstance(packet, list) and len(packet) == ITEM_COUNT, "English packet must contain exactly 12 items")
    ids: set[str] = set()
    cards = 0
    for item in packet:
        _require(isinstance(item, dict), "packet item must be an object")
        review_id = item.get("review_id")
        systems = item.get("systems")
        _require(isinstance(review_id, str) and review_id and review_id not in ids, "packet has duplicate or missing review_id")
        _require(item.get("language") == "en", f"{review_id}: packet language must be en")
        _require(item.get("review_status") == UNREVIEWED, f"{review_id}: packet must remain UNREVIEWED")
        _require(isinstance(systems, list) and len(systems) == 6, f"{review_id}: packet must contain six cards")
        labels = [card.get("system_label") if isinstance(card, dict) else None for card in systems]
        _require(all(isinstance(label, str) and label for label in labels) and len(labels) == len(set(labels)), f"{review_id}: invalid anonymous labels")
        ids.add(review_id)
        cards += len(systems)
    _require(cards == CARD_COUNT, "English packet must contain exactly 72 anonymous cards")
    return packet


def _rating(value: Any, allowed: frozenset[str], field: str, review_id: str) -> str:
    _require(isinstance(value, str) and (value == "" or value in allowed), f"{review_id}: unknown {field} rating")
    return value


def _has_content(record: dict[str, Any]) -> bool:
    if record["reference_correct"] or record["audio_quality"] or record["notes"]:
        return True
    return any(card[field] for card in record["systems"] for field in ("language_correct", "script_correct", "meaning_preserved"))


def recompute_status(record: dict[str, Any], labels: Iterable[str]) -> str:
    cards = {card["system_label"]: card for card in record["systems"]}
    complete = bool(record["reference_correct"] and record["audio_quality"]) and all(
        cards[label]["language_correct"] and cards[label]["script_correct"] and cards[label]["meaning_preserved"]
        for label in labels
    )
    return REVIEWED if complete else IN_PROGRESS if _has_content(record) else UNREVIEWED


def validate_export(export: Any, packet: list[dict[str, Any]]) -> list[dict[str, Any]]:
    _require(isinstance(export, list) and len(export) == ITEM_COUNT, "export must include exactly 12 records, including pending blanks")
    packet_by_id = {item["review_id"]: item for item in packet}
    supplied_ids = [record.get("review_id") if isinstance(record, dict) else None for record in export]
    _require(len(supplied_ids) == len(set(supplied_ids)), "export has duplicate review_id")
    _require(set(supplied_ids) == set(packet_by_id), "export review_ids do not exactly match the English packet")
    validated: list[dict[str, Any]] = []
    for raw in export:
        item = packet_by_id[raw["review_id"]]
        review_id = item["review_id"]
        _require(raw.get("language") == "en", f"{review_id}: export language must be en")
        _require(isinstance(raw.get("notes", ""), str), f"{review_id}: notes must be a string")
        raw_cards = raw.get("systems")
        _require(isinstance(raw_cards, list), f"{review_id}: systems must be an array")
        expected_labels = [card["system_label"] for card in item["systems"]]
        supplied_labels = [card.get("system_label") if isinstance(card, dict) else None for card in raw_cards]
        _require(len(supplied_labels) == len(set(supplied_labels)), f"{review_id}: duplicate system_label")
        _require(set(supplied_labels) == set(expected_labels), f"{review_id}: missing, extra, or changed anonymous card")
        cards_by_label = {card["system_label"]: card for card in raw_cards}
        record = {
            "review_id": review_id,
            "language": "en",
            "reference_correct": _rating(raw.get("reference_correct", ""), REFERENCE, "reference_correct", review_id),
            "audio_quality": _rating(raw.get("audio_quality", ""), AUDIO, "audio_quality", review_id),
            "notes": raw.get("notes", ""),
            "systems": [],
        }
        for label in expected_labels:
            card = cards_by_label[label]
            record["systems"].append(
                {
                    "system_label": label,
                    "language_correct": _rating(card.get("language_correct", ""), BINARY, "language_correct", review_id),
                    "script_correct": _rating(card.get("script_correct", ""), BINARY, "script_correct", review_id),
                    "meaning_preserved": _rating(card.get("meaning_preserved", ""), MEANING, "meaning_preserved", review_id),
                }
            )
        status = recompute_status(record, expected_labels)
        _require(raw.get("review_status") == status, f"{review_id}: stale or false review_status (expected {status})")
        record["review_status"] = status
        validated.append(record)
    return validated


def create_receipt(input_path: Path, reviewer: str, output_dir: Path, packet_path: Path = PACKET) -> Path:
    _require(isinstance(reviewer, str) and reviewer.strip(), "--reviewer pseudonym is required")
    packet = load_packet(packet_path)
    export, raw = _load(input_path)
    records = validate_export(export, packet)
    stem = "".join(character if character.isalnum() or character in "-_" else "_" for character in reviewer.strip())
    _require(stem.strip("_"), "--reviewer must contain a usable pseudonym")
    input_sha256 = _sha256(raw)
    receipt = {
        "schema_version": "blinded-core-en-review-import-v1",
        "reviewer_pseudonym": reviewer.strip(),
        "input_file": input_path.name,
        "input_sha256": input_sha256,
        "packet_data_sha256": _canonical_sha256(packet),
        "packet_item_count": ITEM_COUNT,
        "packet_anonymous_card_count": CARD_COUNT,
        "counts": {
            "reviewed": sum(record["review_status"] == REVIEWED for record in records),
            "pending": sum(record["review_status"] != REVIEWED for record in records),
            "unreviewed": sum(record["review_status"] == UNREVIEWED for record in records),
            "in_progress": sum(record["review_status"] == IN_PROGRESS for record in records),
        },
        "validated_records": records,
        "sealed_mapping_included": False,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    destination = output_dir / f"{stem}_{input_sha256[:12]}_receipt.json"
    _require(not destination.exists(), "create-only receipt already exists for this reviewer and input")
    with destination.open("x", encoding="utf-8") as stream:
        json.dump(receipt, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    return destination


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate an English resource-panel review export.")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--reviewer", required=True, help="reviewer pseudonym")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--packet", type=Path, default=PACKET)
    args = parser.parse_args(argv)
    try:
        receipt = create_receipt(args.input, args.reviewer, args.output_dir, args.packet)
    except (ImportError, OSError) as exc:
        parser.error(str(exc))
    print(json.dumps({"receipt": str(receipt)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
