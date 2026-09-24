"""Synthetic test-only response exports for the blinded review importer."""
from __future__ import annotations

import json

import pytest

from research.bridge_open.core_study import review_import


def _packet():
    return review_import.load_and_validate_packet()


def _blank_export(packet):
    return [
        {
            "review_id": item["review_id"],
            "language": item["language"],
            "reference_correct": "",
            "audio_quality": "",
            "notes": "",
            "review_status": "UNREVIEWED",
            "systems": [
                {"system_label": card["system_label"], "language_correct": "", "script_correct": "", "meaning_preserved": ""}
                for card in item["systems"]
            ],
        }
        for item in packet
    ]


def _completed(record):
    record["reference_correct"] = "yes"
    record["audio_quality"] = "clear"
    for card in record["systems"]:
        card.update(language_correct="yes", script_correct="yes", meaning_preserved="partly")
    record["review_status"] = "REVIEWED"


def _write_json(path, payload):
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def test_blank_full_export_stays_pending_and_creates_blind_artifacts(tmp_path):
    packet = _packet()
    export = _blank_export(packet)
    source = tmp_path / "synthetic_test_only.json"
    _write_json(source, export)
    receipt_path, summary_path = review_import.create_artifacts(source, "test_reviewer", {"hi"}, tmp_path)
    receipt = json.loads(receipt_path.read_text())
    summary = json.loads(summary_path.read_text())
    assert receipt["reviewed_response_count"] == 0
    assert receipt["pending_response_count"] == 50
    assert all(row["review_status"] == "UNREVIEWED" for row in receipt["validated_records"])
    assert summary["counts"]["items_pending"] == 50
    assert summary["anonymous_card_outcomes"] == {}
    assert receipt["sealed_mapping_included"] is False
    rendered = receipt_path.read_text() + summary_path.read_text()
    assert "Whisper Turbo" not in rendered
    assert "not aggregate population estimates" in summary["analysis_scope"]["sampling_note"]


@pytest.mark.parametrize("mutation, message", [
    (lambda rows: rows.__setitem__(0, {**rows[0], "review_id": "wrong-id"}), "review_ids do not exactly match"),
    (lambda rows: rows.__setitem__(1, {**rows[1], "review_id": rows[0]["review_id"]}), "duplicate review_id"),
    (lambda rows: rows[0].__setitem__("review_status", "REVIEWED"), "stale or false review_status"),
    (lambda rows: rows[0]["systems"][0].__setitem__("meaning_preserved", "maybe"), "invalid 'meaning_preserved'"),
    (lambda rows: rows[0].__setitem__("language", "xx"), "wrong language"),
    (lambda rows: rows[0]["systems"][0].__setitem__("system_label", "system_stale"), "anonymous card labels do not exactly match"),
    (lambda rows: rows[0]["systems"][1].__setitem__("system_label", rows[0]["systems"][0]["system_label"]), "duplicate system_label"),
])
def test_synthetic_test_only_invalid_exports_are_rejected(mutation, message):
    packet = _packet()
    rows = _blank_export(packet)
    mutation(rows)
    with pytest.raises(review_import.ReviewImportError, match=message):
        review_import.validate_export(rows, packet, {"hi", "bn", "as"})


def test_rejects_completed_item_outside_declared_reviewer_competency():
    packet = _packet()
    rows = _blank_export(packet)
    bengali_index = next(index for index, row in enumerate(rows) if row["language"] == "bn")
    _completed(rows[bengali_index])
    with pytest.raises(review_import.ReviewImportError, match="outside declared reviewer competency"):
        review_import.validate_export(rows, packet, {"hi"})


def test_reviewed_only_summary_separates_uncertain_from_no():
    packet = _packet()
    rows = _blank_export(packet)
    hindi_index = next(index for index, row in enumerate(rows) if row["language"] == "hi")
    _completed(rows[hindi_index])
    rows[hindi_index]["systems"][0]["meaning_preserved"] = "uncertain"
    rows[hindi_index]["systems"][1]["meaning_preserved"] = "no"
    records = review_import.validate_export(rows, packet, {"hi"})
    summary = review_import.build_summary(records, packet, "test_reviewer", {"hi"})
    by_label = summary["anonymous_card_outcomes"]
    assert summary["counts"]["items_reviewed"] == 1
    assert summary["counts"]["items_pending"] == 49
    assert by_label[rows[hindi_index]["systems"][0]["system_label"]]["meaning_preserved"]["uncertain"] == 1
    assert by_label[rows[hindi_index]["systems"][1]["system_label"]]["meaning_preserved"]["no"] == 1
    assert by_label[rows[hindi_index]["systems"][0]["system_label"]]["meaning_preserved"]["evaluable"] == 0


def test_create_only_receipt_has_deterministic_input_and_packet_hashes(tmp_path):
    packet = _packet()
    source = tmp_path / "synthetic_test_only.json"
    _write_json(source, _blank_export(packet))
    receipt_path, _ = review_import.create_artifacts(source, "test_reviewer", {"hi"}, tmp_path)
    receipt = json.loads(receipt_path.read_text())
    assert receipt["input_sha256"] == review_import._sha256_bytes(source.read_bytes())
    assert receipt["packet_data_sha256"] == review_import._canonical_sha256(packet)
    with pytest.raises(review_import.ReviewImportError, match="create-only artifacts already exist"):
        review_import.create_artifacts(source, "test_reviewer", {"hi"}, tmp_path)
