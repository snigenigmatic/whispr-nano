"""Synthetic test-only exports; no test creates a completed human review."""
from __future__ import annotations

import json

import pytest

from research.bridge_open.listening_review_core_en_v1 import review_import


def _blank_export(packet):
    return [
        {
            "review_id": item["review_id"],
            "language": "en",
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


def _write(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")


def test_blank_export_creates_pending_blind_receipt(tmp_path):
    packet = review_import.load_packet()
    source = tmp_path / "synthetic_test_only_blank.json"
    _write(source, _blank_export(packet))
    receipt_path = review_import.create_receipt(source, "test_reviewer", tmp_path)
    receipt = json.loads(receipt_path.read_text())
    assert receipt["counts"] == {"reviewed": 0, "pending": 12, "unreviewed": 12, "in_progress": 0}
    assert receipt["sealed_mapping_included"] is False
    assert "OmniASR" not in receipt_path.read_text()
    with pytest.raises(review_import.ImportError, match="create-only receipt"):
        review_import.create_receipt(source, "test_reviewer", tmp_path)


@pytest.mark.parametrize("mutation, message", [
    (lambda rows: rows.pop(), "exactly 12 records"),
    (lambda rows: rows[0].__setitem__("review_id", "unknown-id"), "do not exactly match"),
    (lambda rows: rows[1].__setitem__("review_id", rows[0]["review_id"]), "duplicate review_id"),
    (lambda rows: rows[0]["systems"].pop(), "missing, extra, or changed anonymous card"),
    (lambda rows: rows[0]["systems"][1].__setitem__("system_label", rows[0]["systems"][0]["system_label"]), "duplicate system_label"),
    (lambda rows: rows[0]["systems"][0].__setitem__("meaning_preserved", "maybe"), "unknown meaning_preserved rating"),
])
def test_rejects_synthetic_test_only_invalid_exports(mutation, message):
    rows = _blank_export(review_import.load_packet())
    mutation(rows)
    with pytest.raises(review_import.ImportError, match=message):
        review_import.validate_export(rows, review_import.load_packet())


def test_preserves_uncertain_pending_response_without_completing_it(tmp_path):
    packet = review_import.load_packet()
    rows = _blank_export(packet)
    rows[0]["systems"][0]["meaning_preserved"] = "uncertain"
    rows[0]["review_status"] = "IN_PROGRESS"
    source = tmp_path / "synthetic_test_only_uncertain.json"
    _write(source, rows)
    receipt = json.loads(review_import.create_receipt(source, "test_reviewer", tmp_path).read_text())
    assert receipt["counts"] == {"reviewed": 0, "pending": 12, "unreviewed": 11, "in_progress": 1}
    assert receipt["validated_records"][0]["systems"][0]["meaning_preserved"] == "uncertain"
