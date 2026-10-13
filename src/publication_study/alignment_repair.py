"""Abstaining acoustic-alignment gate for the frozen adjacent-loop trigger.

Zero DTW duration is a diagnostic signal, not proof that a word was unspoken.
Only words in a run of >=5 identical tokens may be removed. Any mismatch between
transcript and alignment word boundaries causes abstention for the whole clip.
"""
from __future__ import annotations
from publication_study.loop_repair import _key, MIN_REPETITIONS


def repair_zero_duration(text: str, aligned: list[dict]) -> tuple[str, dict]:
    words=text.split()
    indexes=[i for i,w in enumerate(words) if _key(w)]
    usable=[w for w in aligned if _key(w['word'])]
    if [_key(words[i]) for i in indexes] != [_key(w['word']) for w in usable]:
        return text,dict(status='abstain_word_mapping',removed=0)
    duration={i:float(w['end'])-float(w['start']) for i,w in zip(indexes,usable)}
    if any(d<0 for d in duration.values()):
        return text,dict(status='abstain_invalid_duration',removed=0)
    drop=set();i=0
    while i<len(words):
        key=_key(words[i]);j=i+1
        if key:
            while j<len(words) and _key(words[j])==key:j+=1
        if key and j-i>=MIN_REPETITIONS:
            # Always keep the first occurrence; remove only later zero-span copies.
            drop.update(k for k in range(i+1,j) if duration[k]<=1e-8)
        i=j
    if not drop:return text,dict(status='unchanged',removed=0)
    return ' '.join(w for i,w in enumerate(words) if i not in drop),dict(status='repaired',removed=len(drop))


def duration_character_cap(text: str, duration_seconds: float) -> str:
    """Literature baseline: <= one Unicode character per 20 ms audio frame.

    Cut at a whitespace word boundary. Keep untouched output when within cap.
    Includes spaces in the character budget; no gold transcript is consulted.
    """
    cap=max(0,int(duration_seconds*50))
    if len(text)<=cap:return text
    end=text.rfind(' ',0,cap+1)
    return text[:end] if end>=0 else ''
