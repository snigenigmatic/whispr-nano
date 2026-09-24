"""CPU-only sensitivity audit for preserved FLEURS predictions.

This reads existing JSONL/JSON artifacts and never rewrites predictions or
their stored metrics.  The OpenAI BasicTextNormalizer implementation below is
copied from openai/whisper main at commit 86098128c0b4f24f0e2aa2994de830614b474227
(whisper/normalizers/basic.py, SHA-256 4742eaa040e0657fa1247a1361e0d856c62317a43326ea59a40c2e9edd8d2c38).
It is a secondary diagnostic; the study scorer remains the reported metric.
"""
from __future__ import annotations

import json
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path("/Users/prateekraj/whispr-nano")
ART = ROOT / "research/bridge_open/artifacts"
OUT = ART / "fleurs_scoring_audit"
MANIFEST = ROOT / "data/bridge_open/fleurs_test_v1/corpus-v2.jsonl"

import jiwer
import sys

sys.path.insert(0, str(ROOT))
from research.bridge_open.scoring import normalize_v1


def basic_normalize(text: str) -> str:
    """OpenAI BasicTextNormalizer with remove_diacritics=False, split_letters=False."""
    text = text.lower()
    text = re.sub(r"[<\[][^>\]]*[>\]]", "", text)
    text = re.sub(r"\(([^)]+?)\)", "", text)
    text = "".join(
        " " if unicodedata.category(c)[0] in "MSP" else c
        for c in unicodedata.normalize("NFKC", text)
    ).lower()
    return re.sub(r"\s+", " ", text)


def script(ch: str) -> str:
    name = unicodedata.name(ch, "")
    for label in ("DEVANAGARI", "BENGALI", "LATIN", "ARABIC", "GURMUKHI", "GUJARATI", "KANNADA", "MALAYALAM", "TAMIL", "TELUGU"):
        if label in name:
            return label
    return "OTHER"


def dominant(text: str) -> str:
    counts = Counter(script(c) for c in text if c.isalpha())
    return counts.most_common(1)[0][0] if counts else "NONE"


def score(ref: str, hyp: str, normalizer) -> dict:
    r, h = normalizer(ref), normalizer(hyp)
    words = jiwer.process_words(r, h)
    rc, hc = "".join(r.split()), "".join(h.split())
    chars = jiwer.process_characters(rc, hc)
    return dict(wer=words.wer, cer=chars.cer, ref_words=len(r.split()), hyp_words=len(h.split()),
                word_errors=words.substitutions + words.deletions + words.insertions,
                char_errors=chars.substitutions + chars.deletions + chars.insertions)


def main() -> None:
    refs = {json.loads(line)["conversation_id"]: json.loads(line) for line in MANIFEST.read_text().splitlines()}
    by_model: dict[str, dict[str, dict]] = defaultdict(dict)
    source_dirs = []
    for d in sorted(ART.glob("hf-fleurs-test-*-v2")):
        p = d / "predictions.jsonl"
        c = d / "config.json"
        if not p.is_file() or not c.is_file():
            continue
        cfg = json.loads(c.read_text())
        model = cfg.get("model_id")
        if model not in {"openai/whisper-large-v3", "openai/whisper-large-v3-turbo"}:
            continue
        source_dirs.append(str(d))
        for line in p.read_text().splitlines():
            row = json.loads(line)
            cid = row["conversation_id"]
            if cid in by_model[model]:
                raise ValueError(f"duplicate {model} prediction: {cid}")
            by_model[model][cid] = row

    report = {
        "source_dirs": source_dirs,
        "models": {},
        "normalizer_diagnostic": {
            "source": "https://github.com/openai/whisper/blob/86098128c0b4f24f0e2aa2994de830614b474227/whisper/normalizers/basic.py",
            "source_commit": "86098128c0b4f24f0e2aa2994de830614b474227",
            "source_sha256": "4742eaa040e0657fa1247a1361e0d856c62317a43326ea59a40c2e9edd8d2c38",
            "settings": "remove_diacritics=False, split_letters=False",
            "interpretation": "secondary diagnostic only; Indic scores remain the study scorer's NFC/casefold/punctuation convention",
        },
    }
    for model, rows in sorted(by_model.items()):
        langs = defaultdict(list)
        for cid, row in rows.items():
            if cid not in refs:
                raise ValueError(f"missing manifest reference: {cid}")
            refrow = refs[cid]
            reference = json.loads(Path(refrow["reference_path"]).read_text())["reference_text"]
            hyp = row.get("text_raw", "")
            study = score(reference, hyp, normalize_v1)
            # Use stored metric only as a consistency check; no artifact is changed.
            stored = row.get("metrics")
            if not isinstance(stored, dict) or "wer" not in stored:
                raise ValueError(f"missing stored WER: {model} {cid}")
            if abs(float(study["wer"]) - float(stored["wer"])) > 1e-9:
                raise ValueError(f"stored WER mismatch: {model} {cid}")
            basic = score(reference, hyp, basic_normalize)
            langs[row["language"]].append(dict(cid=cid, row=row, reference=reference, hypothesis=hyp, study=study, basic=basic))
        model_out = {"count": len(rows), "languages": {}}
        for lang, items in sorted(langs.items()):
            study_wers = [x["study"]["wer"] for x in items]
            study_cers = [x["study"]["cer"] for x in items]
            basic_wers = [x["basic"]["wer"] for x in items]
            script_mismatch = sum(dominant(x["reference"]) != dominant(x["hypothesis"]) for x in items)
            hyp_norm = [" ".join(unicodedata.normalize("NFC", x["hypothesis"]).casefold().split()) for x in items]
            dup = Counter(hyp_norm)
            repeated = sum(n - 1 for n in dup.values() if n > 1)
            trigram_excess = []
            trigram_max = []
            for text in hyp_norm:
                words = text.split()
                counts = Counter(zip(words, words[1:], words[2:]))
                trigram_excess.append(sum(n - 1 for n in counts.values() if n > 1))
                trigram_max.append(max(counts.values(), default=0))
            errors = sum(x["study"]["word_errors"] for x in items)
            ref_words = sum(x["study"]["ref_words"] for x in items)
            char_errors = sum(x["study"]["char_errors"] for x in items)
            ref_chars = sum(len("".join(normalize_v1(x["reference"]).split())) for x in items)
            model_out["languages"][lang] = {
                "n": len(items), "generation_limit": sum(x["row"].get("status") == "generation_limit" for x in items),
                "wer_over_1": sum(x["study"]["wer"] > 1 for x in items),
                "wer_equal_1": sum(x["study"]["wer"] == 1 for x in items),
                "script_mismatch": script_mismatch, "script_mismatch_rate": script_mismatch / len(items),
                "exact_repeated_hypothesis_excess": repeated,
                "repeated_trigram_excess_total": sum(trigram_excess),
                "utterances_with_repeated_trigram": sum(n > 0 for n in trigram_excess),
                "max_same_trigram_count": max(trigram_max, default=0),
                "mean_wer": sum(study_wers) / len(items), "micro_wer": errors / ref_words,
                "mean_cer": sum(study_cers) / len(items), "micro_cer": char_errors / ref_chars,
                "mean_basic_normalizer_wer": sum(basic_wers) / len(items),
                "basic_minus_study_mean_wer": sum(basic_wers) / len(items) - sum(study_wers) / len(items),
            }
        model_out["all"] = {
            "generation_limit": sum(x.get("status") == "generation_limit" for x in rows.values()),
            "wer_over_1": sum(score(json.loads(Path(refs[c]["reference_path"]).read_text())["reference_text"], r.get("text_raw", ""), normalize_v1)["wer"] > 1 for c, r in rows.items()),
            "script_mismatch": sum(dominant(json.loads(Path(refs[c]["reference_path"]).read_text())["reference_text"]) != dominant(r.get("text_raw", "")) for c, r in rows.items()),
        }
        report["models"][model] = model_out
        if set(rows) != set(refs):
            missing, extra = sorted(set(refs) - set(rows)), sorted(set(rows) - set(refs))
            raise ValueError(f"{model}: manifest ID set mismatch; missing={missing[:3]} extra={extra[:3]}")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "results.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
