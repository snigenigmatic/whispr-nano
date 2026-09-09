"""Stratified 120-clip dual-transcription sample.

Two reviewers transcribe independently, blind to the reference, the
teacher hypothesis, and each other. Accepted = teacher WER 0 under the
capstone normalizer; rejected = WER > 0.
"""

from __future__ import annotations

from collections import Counter
from typing import Any, Sequence

from .constants import (
    FAMILY_OF,
    L1_GROUPS,
    PER_CELL,
    SAMPLE_SEED,
    TEACHER_SYSTEM,
)
from .scoring import normalize_capstone, normalize_whisper, word_edit_counts


def _stable_rng(seed: int):
    import numpy as np

    return np.random.default_rng(seed)


def sample_listening_set(
    rows: Sequence[dict[str, Any]],
    *,
    seed: int = SAMPLE_SEED,
    l1_groups: Sequence[str] = L1_GROUPS,
    per_cell: int = PER_CELL,
) -> list[dict[str, Any]]:
    by_key: dict[tuple[str, bool], list[dict[str, Any]]] = {}
    for row in rows:
        lang = row["primary_language"]
        if lang not in l1_groups:
            continue
        by_key.setdefault((lang, bool(row["accepted"])), []).append(row)

    rng = _stable_rng(seed)
    chosen: list[dict[str, Any]] = []
    for lang in l1_groups:
        for accepted in (True, False):
            pool = sorted(by_key.get((lang, accepted), []), key=lambda r: r["utt_id"])
            if len(pool) < per_cell:
                raise ValueError(
                    f"need {per_cell} {'accepted' if accepted else 'rejected'} "
                    f"clips for {lang}, found {len(pool)}"
                )
            idx = rng.choice(len(pool), size=per_cell, replace=False)
            for i in sorted(int(x) for x in idx):
                src = pool[i]
                chosen.append(_listening_record(src, accepted=accepted))
    chosen.sort(key=lambda r: (r["primary_language"], not r["accepted"], r["utt_id"]))
    return chosen


def _listening_record(row: dict[str, Any], *, accepted: bool) -> dict[str, Any]:
    ref = row["reference_text"]
    hyp = row["hypothesis_text"]
    cap = word_edit_counts(ref, hyp, normalizer="capstone")
    try:
        wh = word_edit_counts(ref, hyp, normalizer="whisper")
        whisper_wer = wh.wer
        whisper_match = normalize_whisper(ref) == normalize_whisper(hyp)
    except Exception:
        whisper_wer = None
        whisper_match = None
    exact = ref.strip() == hyp.strip()
    cap_match = normalize_capstone(ref) == normalize_capstone(hyp)
    if exact:
        disagreement_kind = "exact_match"
    elif cap_match:
        disagreement_kind = "normalization_only"
    else:
        disagreement_kind = "content"
    return {
        "utt_id": row["utt_id"],
        "svarah_index": int(str(row["utt_id"]).split("_")[1]),
        "primary_language": row["primary_language"],
        "family": FAMILY_OF[row["primary_language"]],
        "gender": row.get("gender"),
        "duration_s": row["duration_s"],
        "accepted": accepted,
        "cell": f"{row['primary_language']}|{'accepted' if accepted else 'rejected'}",
        "teacher_system": TEACHER_SYSTEM,
        "reference_text": ref,
        "teacher_text": hyp,
        "capstone_wer": cap.wer,
        "whisper_wer": whisper_wer,
        "exact_match": exact,
        "capstone_match": cap_match,
        "whisper_match": whisper_match,
        "disagreement_kind": disagreement_kind,
        # Reviewers must not see these fields until both transcripts are in.
        "hidden_until_reconcile": ["reference_text", "teacher_text"],
    }


def listening_summary(items: Sequence[dict[str, Any]]) -> dict[str, Any]:
    cells = Counter(i["cell"] for i in items)
    kinds = Counter(i["disagreement_kind"] for i in items)
    return {
        "n": len(items),
        "seed": SAMPLE_SEED,
        "l1_groups": list(L1_GROUPS),
        "per_cell": PER_CELL,
        "cells": dict(sorted(cells.items())),
        "disagreement_kind": dict(sorted(kinds.items())),
        "utt_ids": [i["utt_id"] for i in items],
    }


REVIEWER_INSTRUCTIONS = """You are transcribing Indian-English speech.

Rules:
1. Listen with headphones. Play the clip as many times as you need.
2. Type what was said, including disfluencies you can clearly hear.
3. Do not look at any existing transcript — not the dataset reference, not
   the teacher hypothesis, not the other reviewer's notes.
4. Do not discuss a clip with the other reviewer until both of you have
   submitted that clip.
5. If a stretch is unintelligible, mark it [inaudible]. If you are unsure
   of a word, transcribe your best guess and mark it [uncertain: word].
6. Keep spoken numbers as you hear them (digits vs words). Do not
   "correct" Indian-English grammar or lexical choice.

After both reviewers finish, reconcile. For each disagreement, assign one
primary label:

- teacher_mistake — the recording matches the reference, not the teacher
- reference_mistake — the recording matches the teacher, not the reference
- both_wrong — neither transcript matches what was said
- normalization — same words; punctuation, casing, numerals, or spelling
- genuine_variation — both are plausible hearings of the same audio
- other — say what
"""
