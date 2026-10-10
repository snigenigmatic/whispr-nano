"""Reference-symmetric, Devanagari-preserving ASR text normalization."""

from __future__ import annotations

import unicodedata


_DEV_DIGITS = str.maketrans("०१२३४५६७८९", "0123456789")
_ZERO_WIDTH = {"\u200b", "\u200c", "\u200d", "\u2060", "\ufeff"}


def normalize(text: str) -> str:
    """Normalize a reference or hypothesis with exactly the same rules.

    Latin casing and punctuation are normalized; Indic letters and vowel marks
    are retained. Number *words* are deliberately not rewritten.
    """
    text = unicodedata.normalize("NFC", text)
    text = "".join(c for c in text if c not in _ZERO_WIDTH and unicodedata.category(c) != "Cf")
    text = text.translate(_DEV_DIGITS)
    # NFD exposes nukta even in precomposed characters (e.g. U+095E).
    text = unicodedata.normalize("NFC", unicodedata.normalize("NFD", text).replace("\u093c", ""))
    text = "".join(c.lower() if "LATIN" in unicodedata.name(c, "") else c for c in text)
    chars: list[str] = []
    for i, c in enumerate(text):
        if c in "-/":
            chars.append(" ")
        elif c in {"'", "’"} and i > 0 and i + 1 < len(text) and all(
            "LATIN" in unicodedata.name(n, "") for n in (text[i - 1], text[i + 1])
        ):
            chars.append("'")
        elif unicodedata.category(c)[0] in {"P", "S"}:
            chars.append(" ")
        else:
            chars.append(c)
    return " ".join("".join(chars).split())
