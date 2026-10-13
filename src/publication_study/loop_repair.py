"""Conservative, reference-free repair for adjacent ASR word loops.

The threshold was fixed using the 104-clip Lahaja development sample. It is
applied to unseen speakers without reading their references or edit counts.
"""

from __future__ import annotations

import unicodedata


MIN_REPETITIONS = 5


def _key(token: str) -> str:
    token = unicodedata.normalize("NFC", token).lower()
    return "".join(ch for ch in token if unicodedata.category(ch)[0] in "LMN")


def suppress_adjacent_word_loops(text: str) -> tuple[str, int]:
    """Keep one word from each run of at least five identical adjacent words.

    Returns the repaired text and the number of removed whitespace tokens.
    Punctuation on the first surviving word is retained; no reference text or
    audio-derived statistic enters the decision.
    """
    words = text.split()
    repaired: list[str] = []
    removed = 0
    i = 0
    while i < len(words):
        key = _key(words[i])
        j = i + 1
        if key:
            while j < len(words) and _key(words[j]) == key:
                j += 1
        run = j - i
        if key and run >= MIN_REPETITIONS:
            repaired.append(words[i])
            removed += run - 1
        else:
            repaired.extend(words[i:j])
        i = j
    return " ".join(repaired), removed
