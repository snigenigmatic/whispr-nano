import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "research"))

from bridge_open.audit_scripts import distribution, script_for


def test_tracked_scripts_and_digits():
    result = distribution("हिन्दी १२٣ العربية")
    assert result["counts"]["Devanagari"] > 0
    assert result["counts"]["Arabic"] > 0
    assert result["unassigned_letter_mark_count"] == 0
    assert result["assigned_fraction_of_letters_marks"] == 1.0


def test_tibetan_is_unassigned_and_empty_is_safe():
    assert script_for("བ") is None
    result = distribution("བ 123 !")
    assert result["counts"] == {}
    assert result["letter_mark_count"] == 1
    assert result["unassigned_letter_mark_count"] == 1
    assert result["assigned_fraction_of_letters_marks"] == 0.0
    assert distribution("")["assigned_fraction_of_letters_marks"] is None

