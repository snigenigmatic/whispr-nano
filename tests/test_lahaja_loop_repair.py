"""Check that the frozen repair fires only on sustained loops."""

from publication_study.loop_repair import suppress_adjacent_word_loops


def test_sustained_hindi_loop_preserves_prefix_and_suffix():
    repaired, removed = suppress_adjacent_word_loops(
        "जब लोग आते तो तो तो तो तो तो आगे जाते"
    )
    assert repaired == "जब लोग आते तो आगे जाते"
    assert removed == 5


def test_short_repetition_and_nonadjacent_words_survive():
    text = "हाँ हाँ हाँ हाँ फिर नहीं हाँ हाँ"
    assert suppress_adjacent_word_loops(text) == (text, 0)


def test_punctuation_does_not_hide_sustained_loop():
    assert suppress_adjacent_word_loops("चलो, चलो चलो चलो चलो!") == ("चलो,", 4)
