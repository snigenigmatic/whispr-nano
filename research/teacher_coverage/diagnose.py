"""Descriptive teacher-selection diagnostics on frozen, previously seen data."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from jiwer import process_words

from paper.analysis.analyze import (
    hash_file, load_predictions, normalize_minimal, validate_frozen_inputs,
    validate_panel,
)


ROOT = Path(__file__).resolve().parents[2]
TEACHER = "whisper-large-v3"
STUDENT = "whisper-tiny"
ALTERNATIVE = "Qwen3-ASR-0.6B-hf"
THRESHOLDS = (10, 20, 40)


def selection_summary(rows: list[dict], threshold: float) -> dict:
    rejected = [r for r in rows if 100 * r["teacher_errors"] > threshold * r["words"]]
    total_words = sum(r["words"] for r in rows)
    total_seconds = sum(r["duration_s"] for r in rows)
    rejected_words = sum(r["words"] for r in rejected)
    rejected_hits = sum(r["teacher_hits"] for r in rejected)
    return {
        "n": len(rows), "reference_words": total_words,
        "rejected_n": len(rejected),
        "rejected_utterance_pct": 100 * len(rejected) / len(rows),
        "rejected_reference_words": rejected_words,
        "rejected_reference_word_pct": 100 * rejected_words / total_words,
        "rejected_audio_pct": 100 * sum(r["duration_s"] for r in rejected) / total_seconds,
        "correctly_aligned_reference_words_discarded": rejected_hits,
        "rejected_reference_words_aligned_correct_pct": (
            100 * rejected_hits / rejected_words if rejected_words else None),
        "teacher_wer_pct": 100 * sum(r["teacher_errors"] for r in rows) / total_words,
        "student_wer_pct": 100 * sum(r["student_errors"] for r in rows) / total_words,
        "teacher_error_free_n": sum(r["teacher_errors"] == 0 for r in rows),
        "student_correct_teacher_wrong_n": sum(
            r["student_errors"] == 0 and r["teacher_errors"] > 0 for r in rows),
        "teacher_correct_student_wrong_n": sum(
            r["teacher_errors"] == 0 and r["student_errors"] > 0 for r in rows),
        "both_wrong_n": sum(r["student_errors"] > 0 and r["teacher_errors"] > 0 for r in rows),
        "alternative_correct_teacher_wrong_n": sum(
            r["alternative_errors"] == 0 and r["teacher_errors"] > 0 for r in rows),
        "reference_oracle_teacher_alternative_wer_pct": 100 * sum(
            min(r["teacher_errors"], r["alternative_errors"]) for r in rows) / total_words,
    }


def main() -> None:
    from transformers.models.whisper.english_normalizer import EnglishTextNormalizer

    systems, provenance = load_predictions(ROOT / "audit/svarah_fairness_audit")
    manifest_path = ROOT / "paper/data/input_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    validate_frozen_inputs(systems, provenance, manifest)
    selected = {m: systems[m] for m in (TEACHER, STUDENT, ALTERNATIVE)}
    ids = sorted(selected[TEACHER])
    assert len(ids) == 2500
    assert all(set(p) == set(ids) for p in selected.values())
    validate_panel(selected, ids)
    spelling = ROOT / "paper/data/whisper_normalizer.json"
    normalizers = {
        "minimal": normalize_minimal,
        "english": EnglishTextNormalizer(json.loads(spelling.read_text())),
    }
    scored, summary = [], []
    for name, normalize in normalizers.items():
        rows = []
        for uid in ids:
            p = selected[TEACHER][uid]
            ref = normalize(p["reference_text"])
            n = len(ref.split())
            if not n:
                raise ValueError(f"Zero-word reference: {uid}, {name}")
            row = {"utt_id": uid, "normalizer": name,
                   "native_language": p["primary_language"], "words": n,
                   "duration_s": float(p["duration_s"])}
            for key, model in (("teacher", TEACHER), ("student", STUDENT),
                               ("alternative", ALTERNATIVE)):
                counts = process_words(ref, normalize(selected[model][uid]["hypothesis_text"]))
                row[f"{key}_errors"] = counts.substitutions + counts.deletions + counts.insertions
                row[f"{key}_hits"] = counts.hits
                assert counts.hits + counts.substitutions + counts.deletions == n
            rows.append(row)
        scored += rows
        groups = {"ALL": rows, "non_Bodo": [r for r in rows if r["native_language"] != "Bodo"]}
        groups.update({g: [r for r in rows if r["native_language"] == g]
                       for g in sorted({r["native_language"] for r in rows})})
        for threshold in THRESHOLDS:
            for group, group_rows in groups.items():
                summary.append({"normalizer": name, "group": group,
                                "threshold_wer_pct": threshold,
                                **selection_summary(group_rows, threshold)})
    out = ROOT / "research/teacher_coverage/results"
    out.mkdir(parents=True, exist_ok=True)
    for filename, rows in (("per_utterance.csv", scored), ("selection.csv", summary)):
        with (out / filename).open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    result = {"status": "exploratory_hypothesis_generation_no_training_no_inference",
              "teacher": TEACHER, "student": STUDENT, "alternative": ALTERNATIVE,
              "thresholds_wer_pct": THRESHOLDS,
              "filter_rule": "reject if per-utterance WER is strictly greater than threshold",
              "oracle_warning": "Reference-dependent lower bound among two fixed hypotheses, not a method result",
              "manifest_sha256": hash_file(manifest_path),
              "spelling_map_sha256": hash_file(spelling),
              "sources": provenance,
              "results": summary}
    (out / "diagnostic.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps([r for r in summary if r["threshold_wer_pct"] == 10
                      and r["group"] in ("ALL", "Bodo", "non_Bodo")], indent=2))


if __name__ == "__main__":
    main()
