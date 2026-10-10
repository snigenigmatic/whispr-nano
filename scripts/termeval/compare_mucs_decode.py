"""Compare complete MUCS auto-language and forced-Hindi decodes on the same IDs."""

from __future__ import annotations

from collections import Counter, defaultdict
import json
from pathlib import Path
import random
import unicodedata

import jiwer

from ast_asr.termeval.normalize import normalize


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/night1"
SEED = 20261011


def _pct_interval(values: list[float]) -> list[float]:
    values.sort()
    return [values[int(0.025 * (len(values) - 1))],
            values[int(0.975 * (len(values) - 1))]]


def main() -> None:
    manifest = {r["utt_id"]: r for r in map(json.loads, (ROOT / "data/termeval/manifest.jsonl").open())
                if r["dataset"] == "mucs_hien"}
    policy = json.loads((OUT / "decode_policy.json").read_text())
    specs = {"auto_language": policy["secondary"]["mucs_hien_auto_language"],
             "forced_hi": policy["primary"]["mucs_hien"]}
    files = {condition: OUT / spec["prediction_file"] for condition, spec in specs.items()}
    by_condition: dict[str, dict[str, dict]] = {}
    for condition, file in files.items():
        rows = [json.loads(line) for line in file.open()]
        if len(rows) != len(manifest) or {r["utt_id"] for r in rows} != set(manifest):
            raise ValueError(f"{condition}: predictions do not match MUCS manifest")
        if len({r["config_hash"] for r in rows}) != 1:
            raise ValueError(f"{condition}: mixed config hashes")
        if {r["condition_id"] for r in rows} != {specs[condition]["condition_id"]}:
            raise ValueError(f"{condition}: decode condition disagrees with frozen policy")
        by_condition[condition] = {r["utt_id"]: r for r in rows}
    speaker_totals: dict[str, dict[str, Counter[str]]] = {}
    report = {"purpose": "full-corpus decoder-setting QC; setting selected after a small evaluation-source probe",
              "n_utterances": len(manifest), "n_speakers": len({r["speaker"] for r in manifest.values()}),
              "bootstrap_resamples": 1000, "bootstrap_unit": "speaker ID", "conditions": {}}
    for condition, rows in by_condition.items():
        totals: Counter[str] = Counter()
        by_speaker: dict[str, Counter[str]] = defaultdict(Counter)
        for utt, row in manifest.items():
            ref = normalize(row["reference"])
            hyp = normalize(rows[utt]["text"])
            score = jiwer.process_words(ref, hyp)
            errors = score.substitutions + score.deletions + score.insertions
            ref_words = score.hits + score.substitutions + score.deletions
            arabic = any("ARABIC" in unicodedata.name(c, "") for c in hyp)
            repeated = any(n >= 4 for n in Counter(hyp.split()).values())
            for target in (totals, by_speaker[row["speaker"]]):
                target["errors"] += errors
                target["reference_words"] += ref_words
                target["arabic_script_outputs"] += arabic
                target["arabic_script_reference_words"] += ref_words if arabic else 0
                target["arabic_script_errors"] += errors if arabic else 0
                target["repeated_token_ge4"] += repeated
                target["repeated_token_errors"] += errors if repeated else 0
                target["empty_outputs"] += not hyp
                target["utterance_wer_ge200pct"] += score.wer >= 2
        speaker_totals[condition] = by_speaker
        report["conditions"][condition] = {
            "condition_id": specs[condition]["condition_id"],
            "config_hash": next(iter({r["config_hash"] for r in rows.values()})),
            **totals,
            "WER": totals["errors"] / totals["reference_words"],
        }
    auto = speaker_totals["auto_language"]
    forced = speaker_totals["forced_hi"]
    speakers = sorted(auto)
    if speakers != sorted(forced):
        raise ValueError("decoder-setting comparison has different speaker sets")
    rng = random.Random(SEED)
    rates = {"auto_language": [], "forced_hi": [], "auto_minus_forced_pp": []}
    for _ in range(1000):
        chosen = rng.choices(speakers, k=len(speakers))
        a = sum(auto[s]["errors"] for s in chosen) / sum(auto[s]["reference_words"] for s in chosen)
        f = sum(forced[s]["errors"] for s in chosen) / sum(forced[s]["reference_words"] for s in chosen)
        rates["auto_language"].append(a)
        rates["forced_hi"].append(f)
        rates["auto_minus_forced_pp"].append(100 * (a - f))
    report["conditions"]["auto_language"]["WER_ci95"] = _pct_interval(rates["auto_language"])
    report["conditions"]["forced_hi"]["WER_ci95"] = _pct_interval(rates["forced_hi"])
    report["paired_auto_minus_forced_wer_pp"] = {
        "point": 100 * (report["conditions"]["auto_language"]["WER"]
                        - report["conditions"]["forced_hi"]["WER"]),
        "ci95": _pct_interval(rates["auto_minus_forced_pp"]),
    }
    (OUT / "mucs_full_language_comparison.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({"auto_WER": report["conditions"]["auto_language"]["WER"],
                      "forced_hi_WER": report["conditions"]["forced_hi"]["WER"],
                      "auto_minus_forced_wer_pp": report["paired_auto_minus_forced_wer_pp"]}))


if __name__ == "__main__":
    main()
