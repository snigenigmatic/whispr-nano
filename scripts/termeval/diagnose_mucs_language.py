"""Compare matched auto-language and forced-Hindi MUCS engineering probes."""

from __future__ import annotations

from collections import Counter, defaultdict
import json
from pathlib import Path
import unicodedata

import jiwer

from ast_asr.termeval.normalize import normalize


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/night1"


def _has_arabic(text: str) -> bool:
    return any("ARABIC" in unicodedata.name(char, "") for char in text)


def main() -> None:
    manifest = {r["utt_id"]: r for r in map(json.loads, (ROOT / "data/termeval/manifest.jsonl").open())}
    ids = json.loads((OUT / "language_probe_ids.json").read_text())["utt_ids"]
    forced = [json.loads(line) for line in (OUT / "predictions_mucs_hien_forced_hi_probe.jsonl").open()]
    auto_file = (OUT / "predictions_mucs_hien.jsonl" if (OUT / "predictions_mucs_hien.jsonl").exists()
                 else OUT / "predictions_mucs_hien_auto_partial.jsonl")
    auto = [json.loads(line) for line in auto_file.open()]
    by_condition = {"auto_language": {p["utt_id"]: p for p in auto},
                    "forced_hi": {p["utt_id"]: p for p in forced}}
    if len(ids) != len(set(ids)) or len(ids) != 160:
        raise ValueError("probe IDs are not frozen 160-item set")
    if len(forced) != len(ids) or set(by_condition["forced_hi"]) != set(ids):
        raise ValueError("forced-Hindi probe predictions incomplete or duplicated")
    if any(set(ids) - set(rows) for rows in by_condition.values()):
        raise ValueError("missing matched auto-language predictions")
    if any(len({p["config_hash"] for p in rows.values()}) != 1 for rows in by_condition.values()):
        raise ValueError("mixed prediction configurations")
    report = {"purpose": "engineering QC only; first-640 ordered MUCS sample, 20 clips per speaker",
              "n_utterances": len(ids), "n_speakers": len({manifest[u]["speaker"] for u in ids}),
              "conditions": {}}
    for condition, predictions in by_condition.items():
        counts: Counter[str] = Counter()
        by_speaker: dict[str, Counter[str]] = defaultdict(Counter)
        for utt in ids:
            reference = normalize(manifest[utt]["reference"])
            hypothesis = normalize(predictions[utt]["text"])
            score = jiwer.process_words(reference, hypothesis)
            errors = score.substitutions + score.deletions + score.insertions
            n_ref = score.hits + score.substitutions + score.deletions
            speaker = manifest[utt]["speaker"]
            for target in (counts, by_speaker[speaker]):
                target["errors"] += errors
                target["reference_words"] += n_ref
                target["utterances"] += 1
                target["arabic_script_outputs"] += _has_arabic(hypothesis)
                target["repeated_token_ge4"] += any(n >= 4 for n in Counter(hypothesis.split()).values())
                target["utterance_wer_ge200pct"] += score.wer >= 2
        report["conditions"][condition] = {
            **counts, "WER": counts["errors"] / counts["reference_words"],
            "speakers": {speaker: {**value, "WER": value["errors"] / value["reference_words"]}
                         for speaker, value in sorted(by_speaker.items())},
        }
    (OUT / "mucs_language_probe_qc.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({condition: {k: value[k] for k in ("WER", "arabic_script_outputs", "repeated_token_ge4")}
                      for condition, value in report["conditions"].items()}))


if __name__ == "__main__":
    main()
