"""Deterministic metrics for the independent BRIDGE-open pilot.

This scorer is deliberately an open pilot convention, rather than an attempt
to reproduce a proprietary BRIDGE scorer.  JiWER supplies the alignment and
standard WER/WIL definitions; normalization and the CER character unit are
kept here so they can be reviewed and frozen with the pilot.
"""

from __future__ import annotations

import unicodedata
from typing import Any

import jiwer


def normalize_v1(text: str) -> str:
    """Normalize pilot text while retaining Unicode scripts, marks and symbols.

    Punctuation (Unicode categories beginning with ``P``) is replaced with a
    space.  Symbols are intentionally retained, as are combining marks and
    digits; there is no number-to-word conversion.
    """

    if not isinstance(text, str):
        raise TypeError("text must be a string")
    normalized = unicodedata.normalize("NFC", text).casefold()
    normalized = "".join(
        " " if unicodedata.category(char).startswith("P") else char
        for char in normalized
    )
    return " ".join(normalized.split())


def score(reference: str, hypothesis: str) -> dict[str, Any]:
    """Return WER, CER, WIL and word-level error counts for one utterance.

    A reference that normalizes to no words is rejected because its rates have
    no meaningful denominator.  JiWER's empty-hypothesis behavior is retained:
    every reference word/character is a deletion and WER/CER are 1.0.
    """

    reference_normalized = normalize_v1(reference)
    hypothesis_normalized = normalize_v1(hypothesis)
    if not reference_normalized:
        raise ValueError("reference must contain at least one normalized token")

    words = jiwer.process_words(reference_normalized, hypothesis_normalized)

    # CER is over Unicode code points after removing all whitespace.  Python's
    # strings iterate by Unicode code point, which preserves combining marks as
    # individual units when they are present in the NFC result.
    reference_characters = "".join(reference_normalized.split())
    hypothesis_characters = "".join(hypothesis_normalized.split())
    characters = jiwer.process_characters(reference_characters, hypothesis_characters)

    return {
        "wer": words.wer,
        "cer": characters.cer,
        "wil": words.wil,
        "hits": words.hits,
        "substitutions": words.substitutions,
        "deletions": words.deletions,
        "insertions": words.insertions,
        "ref_words": len(reference_normalized.split()),
        "ref_chars": len(reference_characters),
        "hyp_words": len(hypothesis_normalized.split()),
        "character_errors": characters.substitutions + characters.deletions + characters.insertions,
    }
