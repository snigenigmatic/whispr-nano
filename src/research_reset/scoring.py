"""Two WER normalizers and word-level edit counts.

`capstone` reproduces `ast_asr.metrics.normalize_for_wer` on the committed
whisper-large-v3 Svarah CSV (2,500/2,500 exact matches). It is the
**primary** normalizer for the label-source gate.

`whisper` is Radford et al. 2022 Appendix C.1 via Hugging Face
`EnglishTextNormalizer`. It is the **alternate** normalizer: the stop
rule fails if the teacher-vs-reference conclusion reverses under it.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from typing import Callable

NORMALIZER_NAMES = ("capstone", "whisper")


def normalize_capstone(text: str) -> str:
    """Lowercase, turn hyphens into spaces, strip punctuation except
    apostrophes and underscores. Matches the Phase 2 / audit convention."""
    text = text.lower()
    for src, dst in (("—", " "), ("–", " "), ("−", " "), ("-", " ")):
        text = text.replace(src, dst)
    text = re.sub(r"[^a-z0-9'_\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


@lru_cache(maxsize=1)
def _whisper_normalizer():
    from transformers.models.whisper.english_normalizer import EnglishTextNormalizer

    return EnglishTextNormalizer({})


def normalize_whisper(text: str) -> str:
    return _whisper_normalizer()(text) or ""


def get_normalizer(name: str) -> Callable[[str], str]:
    if name == "capstone":
        return normalize_capstone
    if name == "whisper":
        return normalize_whisper
    raise ValueError(f"unknown normalizer {name!r}; expected one of {NORMALIZER_NAMES}")


@dataclass(frozen=True)
class WordEditCounts:
    substitutions: int
    deletions: int
    insertions: int
    reference_words: int

    @property
    def errors(self) -> int:
        return self.substitutions + self.deletions + self.insertions

    @property
    def wer(self) -> float:
        if self.reference_words == 0:
            return 0.0 if self.errors == 0 else 1.0
        return self.errors / self.reference_words


def word_edit_counts(reference: str, hypothesis: str, *, normalizer: str = "capstone") -> WordEditCounts:
    norm = get_normalizer(normalizer)
    ref_tokens = (norm(reference) or "").split()
    hyp_tokens = (norm(hypothesis) or "").split()
    return _levenshtein(ref_tokens, hyp_tokens)


def _levenshtein(ref: list[str], hyp: list[str]) -> WordEditCounts:
    n, m = len(ref), len(hyp)
    # dp[i][j] = (sub, del, ins) for ref[:i], hyp[:j]
    dp_sub = [[0] * (m + 1) for _ in range(n + 1)]
    dp_del = [[0] * (m + 1) for _ in range(n + 1)]
    dp_ins = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        dp_del[i][0] = i
    for j in range(1, m + 1):
        dp_ins[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if ref[i - 1] == hyp[j - 1]:
                dp_sub[i][j] = dp_sub[i - 1][j - 1]
                dp_del[i][j] = dp_del[i - 1][j - 1]
                dp_ins[i][j] = dp_ins[i - 1][j - 1]
                continue
            # substitution
            s_sub, s_del, s_ins = dp_sub[i - 1][j - 1] + 1, dp_del[i - 1][j - 1], dp_ins[i - 1][j - 1]
            s_cost = s_sub + s_del + s_ins
            # deletion
            d_sub, d_del, d_ins = dp_sub[i - 1][j], dp_del[i - 1][j] + 1, dp_ins[i - 1][j]
            d_cost = d_sub + d_del + d_ins
            # insertion
            i_sub, i_del, i_ins = dp_sub[i][j - 1], dp_del[i][j - 1], dp_ins[i][j - 1] + 1
            i_cost = i_sub + i_del + i_ins
            best = min(s_cost, d_cost, i_cost)
            if best == s_cost:
                dp_sub[i][j], dp_del[i][j], dp_ins[i][j] = s_sub, s_del, s_ins
            elif best == d_cost:
                dp_sub[i][j], dp_del[i][j], dp_ins[i][j] = d_sub, d_del, d_ins
            else:
                dp_sub[i][j], dp_del[i][j], dp_ins[i][j] = i_sub, i_del, i_ins
    return WordEditCounts(dp_sub[n][m], dp_del[n][m], dp_ins[n][m], n)


def utterance_wer(reference: str, hypothesis: str, *, normalizer: str = "capstone") -> float:
    return word_edit_counts(reference, hypothesis, normalizer=normalizer).wer


def corpus_wer(pairs: list[tuple[str, str]], *, normalizer: str = "capstone") -> float:
    errors = 0
    ref_words = 0
    for ref, hyp in pairs:
        counts = word_edit_counts(ref, hyp, normalizer=normalizer)
        errors += counts.errors
        ref_words += counts.reference_words
    if ref_words == 0:
        return 0.0
    return errors / ref_words


def macro_wer(
    rows: list[dict],
    groups: tuple[str, ...],
    *,
    reference_key: str = "reference_text",
    hypothesis_key: str = "hypothesis_text",
    group_key: str = "primary_language",
    normalizer: str = "capstone",
) -> dict:
    """Unweighted mean of per-L1 WER. Missing groups are errors, not zeros —
    Bodo must not disappear into a silent empty bucket."""
    per_group: dict[str, float] = {}
    group_n: dict[str, int] = {}
    missing: list[str] = []
    for group in groups:
        subset = [r for r in rows if r[group_key] == group]
        group_n[group] = len(subset)
        if not subset:
            missing.append(group)
            continue
        pairs = [(r[reference_key], r[hypothesis_key]) for r in subset]
        per_group[group] = corpus_wer(pairs, normalizer=normalizer)
    if missing:
        raise ValueError(f"macro WER missing groups {missing}; every designated L1 must appear")
    return {
        "macro_wer": sum(per_group.values()) / len(groups),
        "per_group_wer": per_group,
        "group_n": group_n,
        "n": len(rows),
        "normalizer": normalizer,
    }
