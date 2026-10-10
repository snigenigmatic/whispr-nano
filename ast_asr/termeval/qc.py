"""Reference-free diagnostics for pathological ASR output."""

from __future__ import annotations

from collections.abc import Sequence


def has_repetition_loop(tokens: Sequence[str], *, repeats: int = 4, max_ngram: int = 4) -> bool:
    """Flag a phrase of 1–4 tokens repeated contiguously at least four times.

    Ordinary words recurring in separate parts of a long sentence are not a
    decoder loop. This check uses only the hypothesis, never the reference.
    """
    for ngram in range(1, min(max_ngram, len(tokens) // repeats) + 1):
        for start in range(len(tokens) - repeats * ngram + 1):
            phrase = tokens[start:start + ngram]
            if all(tokens[start + i * ngram:start + (i + 1) * ngram] == phrase
                   for i in range(1, repeats)):
                return True
    return False
