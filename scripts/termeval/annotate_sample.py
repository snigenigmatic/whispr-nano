"""Deterministic, explicitly unverified dictionary pre-labels for a 200-item audit.

These are *not* human or LLM-reviewed labels. They keep a term-error engineering
check possible when no annotation API or reviewer is available.
"""

from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path
import random

from ast_asr.termeval.normalize import normalize


ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "data/termeval/manifest.jsonl"
OUT = ROOT / "results/night1"
SEED = 20261011

# Deliberately high-precision candidate list; unknown names and terms are missed.
NE_LAT = {
    "libreoffice", "impress", "calc", "gedit", "jchempaint", "ubuntu", "linux",
    "python", "firefox", "thunderbird", "gmail", "google", "windows", "gnu",
    "shinchan", "dehradun", "india", "mumbai", "assam", "himalaya", "himalayas",
}
TECH_LAT = {
    "spreadsheet", "formula", "terminal", "compile", "variable", "snippets",
    "snippet", "plugin", "plugins", "hyperlink", "bandwidth", "code", "editor",
    "document", "documents", "slide", "slides", "font", "animation", "cursor",
    "dialog", "software", "browser", "operating", "system", "template", "templates",
    "formatting", "file", "folder", "pdf", "email", "e-mail", "interface",
}
EN_DEV = {
    "कंप्यूटर", "फाइल", "सेव", "स्लाइड", "फॉन्ट", "क्लिक", "ट्यूटोरियल",
    "डॉक्यूमेंट", "विंडो", "स्क्रीन", "फॉर्मूला", "टर्मिनल", "उबंटू",
    "लिबरऑफिस", "इम्प्रेस", "प्रोजेक्ट", "कोड", "सॉफ्टवेयर", "ब्राउजर",
    "माउस", "कीबोर्ड", "एडिटर", "ईमेल", "टाइम", "प्रॉब्लम", "फेस्टिवल",
    "कार्टून", "होमटाउन", "सिटी", "फूड", "फेवरेट", "फेवरेट", "फैमिली",
}
TERM_DEV = {
    "कंप्यूटर", "फाइल", "स्लाइड", "फॉन्ट", "ट्यूटोरियल", "डॉक्यूमेंट",
    "विंडो", "स्क्रीन", "फॉर्मूला", "टर्मिनल", "उबंटू", "लिबरऑफिस",
    "इम्प्रेस", "प्रोजेक्ट", "कोड", "सॉफ्टवेयर", "ब्राउजर", "एडिटर",
}

GUIDELINE = """TERM/NE: people, places, organizations, products, or software.
TERM/TECH: specialized domain vocabulary; omit everyday English borrowings.
EN_DEV: English-origin words written in Devanagari; may overlap TERM.
Ambiguous borrowings, acronyms, and numerals require human review; no silent resolution.
All labels in this file are dictionary candidates, not verified annotation."""


def round_robin_sample(rows: list[dict], count: int, rng: random.Random) -> list[dict]:
    by_speaker: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_speaker[row["speaker"]].append(row)
    for bucket in by_speaker.values():
        rng.shuffle(bucket)
    keys = sorted(by_speaker)
    rng.shuffle(keys)
    selected = []
    while len(selected) < count:
        progressed = False
        for key in keys:
            if by_speaker[key]:
                selected.append(by_speaker[key].pop())
                progressed = True
                if len(selected) == count:
                    break
        if not progressed:
            break
    return selected


def spans_for(text: str) -> tuple[list[str], list[dict]]:
    tokens = normalize(text).split()
    spans = []
    for i, token in enumerate(tokens):
        if token in NE_LAT:
            spans.append({"start": i, "end": i + 1, "label": "TERM", "subtype": "NE"})
        elif token in TECH_LAT or token in {normalize(x) for x in TERM_DEV}:
            spans.append({"start": i, "end": i + 1, "label": "TERM", "subtype": "TECH"})
        if token in {normalize(x) for x in EN_DEV}:
            spans.append({"start": i, "end": i + 1, "label": "EN_DEV", "subtype": None})
    return tokens, spans


def main() -> None:
    rows = [json.loads(line) for line in MANIFEST.open()]
    by_dataset = defaultdict(list)
    for row in rows:
        by_dataset[row["dataset"]].append(row)
    rng = random.Random(SEED)
    selected = round_robin_sample(by_dataset["mucs_hien"], 100, rng)
    for group in ("adult", "children"):
        selected.extend(round_robin_sample(
            [r for r in by_dataset["hiacc"] if r["group"] == group], 50, rng
        ))
    OUT.mkdir(exist_ok=True, parents=True)
    sample_path = OUT / "annotation_sample_ids.json"
    sample_path.write_text(json.dumps({"seed": SEED, "method": "speaker-balanced round-robin; HiACC 50 adult, 50 child",
                                       "items": [{"dataset": r["dataset"], "utt_id": r["utt_id"],
                                                  "speaker": r["speaker"], "group": r.get("group")}
                                                 for r in selected]}, indent=2))
    annotations = OUT / "annotations_heuristic_v0.jsonl"
    n_spans = 0
    with annotations.open("w") as handle:
        for row in selected:
            tokens, spans = spans_for(row["reference"])
            n_spans += len(spans)
            record = {"dataset": row["dataset"], "utt_id": row["utt_id"],
                      "speaker": row["speaker"], "tokens": tokens, "spans": spans,
                      "annotation_method": "dictionary_candidates_unverified_v0"}
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    (OUT / "annotation_guideline.md").write_text("# Unverified candidate annotation guideline\n\n" + GUIDELINE + "\n")
    print(json.dumps({"sample_utterances": len(selected), "candidate_spans": n_spans,
                      "annotation_method": "dictionary_candidates_unverified_v0"}))


if __name__ == "__main__":
    main()
