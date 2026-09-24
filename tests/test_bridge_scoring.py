import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "research"))

from bridge_open.scoring import normalize_v1, score


def test_normalize_v1_preserves_indic_marks_digits_and_symbols():
    assert normalize_v1("हिन्दी, १२३! Café\u0301 $+") == "हिन्दी १२३ café́ $+"


def test_normalize_v1_casefolds_and_collapses_unicode_whitespace():
    assert normalize_v1("  Straße\u00a0—\tHELLO… ") == "strasse hello"


def test_digits_are_not_converted_to_number_words():
    assert normalize_v1("I have 2 apples") != normalize_v1("I have two apples")
    result = score("I have 2 apples", "I have two apples")
    assert result["substitutions"] == 1
    assert result["ref_words"] == 4


def test_known_word_counts_and_rates():
    result = score("one two three", "one four")
    assert result["hits"] == 1
    assert result["substitutions"] == 1
    assert result["deletions"] == 1
    assert result["insertions"] == 0
    assert result["ref_words"] == 3
    assert result["wer"] == pytest.approx(2 / 3)
    assert result["wil"] == pytest.approx(5 / 6)


def test_cer_removes_whitespace_and_counts_unicode_codepoints():
    result = score("नमस्ते दुनिया", "नमस्ते दुनया")
    assert result["ref_chars"] == len("नमस्तेदुनिया")
    assert result["cer"] == pytest.approx(1 / len("नमस्तेदुनिया"))


def test_empty_reference_is_rejected_after_normalization():
    with pytest.raises(ValueError):
        score("!!!   ", "anything")


def test_empty_hypothesis_is_all_deletions():
    result = score("alpha beta", "   !!!")
    assert result["wer"] == 1.0
    assert result["cer"] == 1.0
    assert result["hits"] == 0
    assert result["substitutions"] == 0
    assert result["deletions"] == 2
    assert result["insertions"] == 0
