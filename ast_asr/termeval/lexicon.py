"""Conservative provisional Latin↔Devanagari script-equivalence lexicon."""

from __future__ import annotations

from collections import Counter
import re

from indic_transliteration import sanscript
from indic_transliteration.sanscript import transliterate
from rapidfuzz.distance import Levenshtein

from .align import align_words
from .normalize import normalize
from .tokens import token_class


# Transliteration similarity alone mistakes short Hindi and English function
# words for each other (e.g. English "in" versus Hindi "इन" = "these").
_ENGLISH_FUNCTION_WORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "he",
    "her", "his", "i", "in", "is", "it", "me", "my", "no", "not", "of",
    "on", "or", "our", "she", "so", "that", "the", "their", "them", "there",
    "they", "this", "to", "us", "was", "we", "were", "with", "you", "your",
}
_HINDI_FUNCTION_WORDS = {
    "अब", "आप", "और", "उन", "इन", "इस", "उस", "एक", "का", "की", "के", "को",
    "क्या", "तो", "तक", "था", "थी", "थे", "नहीं", "ना", "पर", "फिर", "में",
    "यह", "वह", "से", "है", "हैं", "हम", "हो",
}


def romanize(devanagari: str) -> str:
    return transliterate(devanagari, sanscript.DEVANAGARI, sanscript.ITRANS).lower()


def phonetic_fold(text: str) -> str:
    text = text.lower().replace("ph", "f").replace("w", "v").replace("z", "j")
    text = text.replace("aa", "a").replace("ii", "i").replace("uu", "u")
    return re.sub(r"([a-z])\1+", r"\1", text)


def build_lexicon(predictions: list[dict], references: dict[str, str]) -> list[dict]:
    candidates: Counter[tuple[str, str]] = Counter()
    example: dict[tuple[str, str], str] = {}
    collision_dev: set[str] = set()
    for prediction in predictions:
        utt = prediction["utt_id"]
        ref = normalize(references[utt]).split()
        hyp = normalize(prediction["text"]).split()
        for edit in align_words(ref, hyp):
            if edit.ref_idx is None or edit.hyp_idx is None:
                continue
            a, b = ref[edit.ref_idx], hyp[edit.hyp_idx]
            ca, cb = token_class(a), token_class(b)
            if ca == cb == "DEV":
                collision_dev.add(a)
            elif edit.op == "SUB" and {ca, cb} == {"LAT", "DEV"}:
                latin, dev = (a, b) if ca == "LAT" else (b, a)
                candidates[(latin, dev)] += 1
                example.setdefault((latin, dev), utt)
    out = []
    for (latin, dev), n in candidates.most_common():
        romanized = romanize(dev)
        dist = Levenshtein.normalized_distance(phonetic_fold(romanized), phonetic_fold(latin))
        collision = dev in collision_dev or dev in _HINDI_FUNCTION_WORDS or latin in _ENGLISH_FUNCTION_WORDS
        status = "COLLISION" if collision else "AUTO_ACCEPT" if dist <= 0.34 and n >= 2 else "REVIEW"
        out.append({"latin": latin, "devanagari": dev, "romanized": romanized,
                    "dist": round(dist, 6), "status": status,
                    "n_occurrences": n, "example_utt": example[(latin, dev)]})
    return out


def approved_pairs(rows: list[dict]) -> set[tuple[str, str]]:
    return {(r["latin"], r["devanagari"]) for r in rows if r["status"] == "AUTO_ACCEPT"}
