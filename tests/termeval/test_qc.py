from ast_asr.termeval.qc import has_repetition_loop


def test_repetition_loop_ignores_scattered_common_words():
    assert not has_repetition_loop("i like mountains and i like oceans so i can go anywhere".split())
    assert not has_repetition_loop("यह एक काम है और यह दूसरा काम है".split())


def test_repetition_loop_detects_contiguous_word_and_phrase_loops():
    assert has_repetition_loop("है है है है".split())
    assert has_repetition_loop("प्रश्न पर प्रश्न पर प्रश्न पर प्रश्न पर".split())
    assert has_repetition_loop("उनकी बारे में उनकी बारे में उनकी बारे में उनकी बारे में".split())
