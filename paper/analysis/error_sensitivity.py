"""Exploratory post hoc decomposition of normalization sensitivity.

No inference, audio correctness judgement, or attribution to numerals alone.
The English normalizer simultaneously changes numbers, spelling, contractions,
fillers, and other conventions. A raw-reference digit flag is a descriptive
partition, not an intervention on just number formatting.
"""

import argparse
import json
import re
from pathlib import Path

from jiwer import process_words

from paper.analysis.analyze import _write_csv, hash_file, load_predictions, normalize_minimal


QWEN = "Qwen3-ASR-0.6B-hf"
SMALL = "whisper-small"
# Chosen after inspecting historical transformations: short, non-sensitive text.
EXAMPLE_IDS = ("svarah_01674", "svarah_01925", "svarah_05634")


def score_rows(systems: dict, english_normalizer) -> list[dict]:
    scored = []
    for system, predictions in sorted(systems.items()):
        for utt_id, prediction in sorted(predictions.items()):
            row = {"system": system, "utt_id": utt_id,
                   "reference_has_digit": bool(re.search(r"\d", prediction["reference_text"])),
                   "primary_language": prediction["primary_language"]}
            for name, normalizer in (("minimal", normalize_minimal), ("english", english_normalizer)):
                reference = normalizer(prediction["reference_text"])
                hypothesis = normalizer(prediction["hypothesis_text"])
                counts = process_words(reference, hypothesis)
                row[f"{name}_errors"] = counts.substitutions + counts.deletions + counts.insertions
                row[f"{name}_reference_words"] = len(reference.split())
            scored.append(row)
    return scored


def decomposition(error_before: int, error_after: int, ref_before: int, ref_after: int) -> dict:
    """Exact, explicitly ordered decomposition: change errors, then denominator."""
    if ref_before <= 0 or ref_after <= 0:
        return {"status": "undefined_zero_reference_denominator"}
    before, after = 100 * error_before / ref_before, 100 * error_after / ref_after
    numerator = 100 * (error_after - error_before) / ref_before
    denominator = 100 * error_after * (1 / ref_after - 1 / ref_before)
    assert abs((after - before) - (numerator + denominator)) < 1e-9
    return {"minimal_errors": error_before, "english_errors": error_after,
            "minimal_reference_words": ref_before, "english_reference_words": ref_after,
            "minimal_wer_pct": before, "english_wer_pct": after,
            "change_pp": after - before, "error_numerator_component_pp": numerator,
            "reference_denominator_component_pp": denominator,
            "decomposition_order": "replace error count first at minimal reference length, then replace reference length"}


def summarize_pair(scored: list[dict], systems: dict) -> dict:
    paired_ids = set(systems[QWEN]) & set(systems[SMALL])
    pair_rows = [row for row in scored if row["system"] in (QWEN, SMALL) and row["utt_id"] in paired_ids]
    groups = {}
    for group in ("all", "raw_reference_contains_digit", "raw_reference_has_no_digit"):
        chosen = [row for row in pair_rows if group == "all" or
                  row["reference_has_digit"] == (group == "raw_reference_contains_digit")]
        models = {}
        for system in (QWEN, SMALL):
            rows = [row for row in chosen if row["system"] == system]
            models[system] = {"n_utterances": len(rows), **decomposition(
                sum(row["minimal_errors"] for row in rows), sum(row["english_errors"] for row in rows),
                sum(row["minimal_reference_words"] for row in rows),
                sum(row["english_reference_words"] for row in rows))}
        q, s = models[QWEN], models[SMALL]
        if q.get("minimal_reference_words") != s.get("minimal_reference_words") or q.get("english_reference_words") != s.get("english_reference_words"):
            raise ValueError("Paired systems have mismatching reference denominators")
        contrast = decomposition(q["minimal_errors"] - s["minimal_errors"],
                                 q["english_errors"] - s["english_errors"],
                                 q["minimal_reference_words"], q["english_reference_words"])
        # For this contrast the 'WER' fields denote signed differences, not a
        # standalone model WER. Rename to prevent a negative rate interpretation.
        contrast["minimal_difference_pp"] = contrast.pop("minimal_wer_pct")
        contrast["english_difference_pp"] = contrast.pop("english_wer_pct")
        groups[group] = {"systems": models, "Qwen_minus_small": contrast}
    examples = []
    for utt_id in EXAMPLE_IDS:
        if utt_id not in systems[QWEN]:
            continue
        row = systems[QWEN][utt_id]
        example = {"system": QWEN, "utt_id": utt_id, "reference_text": row["reference_text"],
                   "hypothesis_text": row["hypothesis_text"]}
        scored_row = next(item for item in pair_rows if item["system"] == QWEN and item["utt_id"] == utt_id)
        example.update(scored_row)
        examples.append(example)
    return {"n_paired_utterances": len(paired_ids), "partitions": groups, "examples": examples}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=Path("audit/svarah_fairness_audit"))
    parser.add_argument("--matched-results-dir", type=Path)
    parser.add_argument("--full-results-dir", type=Path)
    parser.add_argument("--spelling-map", type=Path, default=Path("paper/data/whisper_normalizer.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("paper/results"))
    args = parser.parse_args()
    from transformers.models.whisper.english_normalizer import EnglishTextNormalizer
    english = EnglishTextNormalizer(json.loads(args.spelling_map.read_text()))
    output = {"status": "exploratory_post_hoc_descriptive_no_inference",
              "scope": "This analysis was chosen after observing the historical Qwen/small normalization ranking change.",
              "caveat": "Digit presence is only a raw-reference partition; differences are not causally attributed to numeral handling alone.",
              "spelling_map_sha256": hash_file(args.spelling_map), "cohorts": {}, "inputs": []}
    all_scored = []
    panels = [("historical", args.results_dir)]
    if args.matched_results_dir:
        panels.append(("matched", args.matched_results_dir))
    if args.full_results_dir:
        panels.append(("full", args.full_results_dir))
    for cohort, directory in panels:
        systems, provenance = load_predictions(directory)
        if QWEN not in systems or SMALL not in systems:
            raise ValueError("Qwen and Whisper-small predictions are both required")
        scored = score_rows(systems, english)
        all_scored += [{"panel": cohort, **row} for row in scored]
        output["inputs"] += provenance
        output["cohorts"][cohort] = summarize_pair(scored, systems)
        for example in output["cohorts"][cohort]["examples"]:
            for name, normalizer in (("minimal", normalize_minimal), ("english", english)):
                example[f"{name}_reference"] = normalizer(example["reference_text"])
                example[f"{name}_hypothesis"] = normalizer(example["hypothesis_text"])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(args.output_dir / "per_utterance_scores.csv", all_scored)
    (args.output_dir / "error_sensitivity.json").write_text(json.dumps(output, indent=2) + "\n")
    print(f"Wrote {len(all_scored)} per-utterance scores and error_sensitivity.json")


if __name__ == "__main__":
    main()
