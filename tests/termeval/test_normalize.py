from ast_asr.termeval.normalize import normalize
from ast_asr.termeval.tokens import token_class


def test_requested_examples():
    assert normalize("अब हम Python का code run करेंगे।") == "अब हम python का code run करेंगे"
    assert normalize("फ़ाइल को save कीजिए") == "फाइल को save कीजिए"
    assert normalize("LibreOffice Calc में cell A1") == "libreoffice calc में cell a1"
    assert normalize("१२३ और 123") == "123 और 123"
    assert normalize("e-mail\u200c भेजें") == "e mail भेजें"


def test_unicode_and_punctuation():
    assert normalize("don't / DON'T; run—now॥") == "don't don't run now"
    assert normalize("ABC\u200bDEF \u200dफ़ाइल") == "abcdef फाइल"
    assert normalize("३.१४") == "3 14"


def test_classes():
    assert [token_class(t) for t in ["python", "हिंदी", "123", "apiहिंदी", "don't"]] == [
        "LAT", "DEV", "NUM", "MIX", "LAT"
    ]
