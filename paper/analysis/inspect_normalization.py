"""Post hoc examples of presentation sensitivity; no acoustic error attribution.

This saves a compact scored row for every prediction, plus deterministic examples
with the largest increases/decreases in total errors. Examples do not partition
the overall WER change into causes: the English normalizer changes multiple
features and reference lengths simultaneously.
"""

import argparse
import json
from pathlib import Path

from paper.analysis.analyze import (
    _write_csv,
    hash_file,
    load_predictions,
    normalize_minimal,
    word_edit_counts,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=Path("audit/svarah_fairness_audit"))
    parser.add_argument("--spelling-map", type=Path, default=Path("paper/data/whisper_normalizer.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("paper/results/normalization_examples"))
    parser.add_argument("--examples-per-direction", type=int, default=10)
    args = parser.parse_args()
    from transformers.models.whisper.english_normalizer import EnglishTextNormalizer
    normalizer = EnglishTextNormalizer(json.loads(args.spelling_map.read_text()))
    systems, provenance = load_predictions(args.results_dir)
    audit_rows, examples, summaries = [], [], []
    for system, source in sorted(systems.items()):
        scored = []
        for utt_id, row in sorted(source.items()):
            minimal_ref = normalize_minimal(row["reference_text"])
            minimal_hyp = normalize_minimal(row["hypothesis_text"])
            english_ref = normalizer(row["reference_text"])
            english_hyp = normalizer(row["hypothesis_text"])
            minimal = word_edit_counts(minimal_ref, minimal_hyp)
            english = word_edit_counts(english_ref, english_hyp)
            compact = {"system": system, "utt_id": utt_id, "primary_language": row["primary_language"],
                       "minimal_errors": minimal.errors, "english_errors": english.errors,
                       "error_reduction": minimal.errors - english.errors,
                       "minimal_reference_words": minimal.reference_words,
                       "english_reference_words": english.reference_words,
                       "reference_changed": minimal_ref != english_ref,
                       "hypothesis_changed": minimal_hyp != english_hyp}
            audit_rows.append(compact)
            scored.append({**compact, "reference_text": row["reference_text"],
                           "hypothesis_text": row["hypothesis_text"],
                           "minimal_reference": minimal_ref, "minimal_hypothesis": minimal_hyp,
                           "english_reference": english_ref, "english_hypothesis": english_hyp})
        for direction, eligible in (
                ("fewer_errors", [row for row in scored if row["error_reduction"] > 0]),
                ("more_errors", [row for row in scored if row["error_reduction"] < 0])):
            ordered = sorted(eligible, key=lambda row: (-abs(row["error_reduction"]), row["utt_id"]))
            examples += [{"selection": direction, **row} for row in ordered[:args.examples_per_direction]]
        summaries.append({"system": system, "n_utterances": len(scored),
                          "fewer_error_rows": sum(row["error_reduction"] > 0 for row in scored),
                          "more_error_rows": sum(row["error_reduction"] < 0 for row in scored),
                          "unchanged_error_rows": sum(row["error_reduction"] == 0 for row in scored),
                          "minimal_total_errors": sum(row["minimal_errors"] for row in scored),
                          "english_total_errors": sum(row["english_errors"] for row in scored),
                          "minimal_total_reference_words": sum(row["minimal_reference_words"] for row in scored),
                          "english_total_reference_words": sum(row["english_reference_words"] for row in scored)})
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(args.output_dir / "audit_rows.csv", audit_rows)
    _write_csv(args.output_dir / "examples.csv", examples)
    (args.output_dir / "summary.json").write_text(json.dumps({
        "status": "post_hoc_descriptive_examples_no_causal_attribution",
        "inputs": provenance, "spelling_map_sha256": hash_file(args.spelling_map),
        "systems": summaries,
    }, indent=2) + "\n")
    print(f"Wrote {len(audit_rows)} audited predictions and {len(examples)} examples to {args.output_dir}")


if __name__ == "__main__":
    main()
