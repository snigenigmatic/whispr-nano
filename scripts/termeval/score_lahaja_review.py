"""Score independent labels in the private 30-clip Lahaja review packet."""

from __future__ import annotations

from collections import Counter
import csv
import json
from pathlib import Path


ROOT = Path.home() / ".local/share/openresearch/lahaja-review-dev-v1"
FIELDS = {
    "reference": ["reference_matches_audio_yes_no_uncertain"],
    "asr": ["numeric_value_correct_yes_no_na_uncertain",
            "named_entity_correct_yes_no_na_uncertain",
            "domain_term_correct_yes_no_na_uncertain",
            "other_acoustic_error_yes_no_uncertain",
            "hallucination_yes_no_uncertain"],
}


def _read(reviewer: str, stage: str) -> dict[str, dict]:
    with (ROOT / f"reviewer_{reviewer}_{stage}.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    by_id = {r["item_id"]: r for r in rows}
    if len(rows) != 30 or len(by_id) != 30:
        raise ValueError(f"incomplete or duplicate reviewer_{reviewer}_{stage} sheet")
    return by_id


def _agreement(left: list[str], right: list[str]) -> dict:
    if len(left) != len(right) or not left:
        raise ValueError("empty or mismatched labels")
    n = len(left)
    observed = sum(a == b for a, b in zip(left, right, strict=True)) / n
    ca, cb = Counter(left), Counter(right)
    expected = sum(ca[label] * cb[label] for label in ca.keys() | cb.keys()) / n**2
    kappa = None if expected == 1 else (observed - expected) / (1 - expected)
    return {"n": n, "raw_agreement": observed, "cohens_kappa": kappa,
            "reviewer_a_counts": dict(ca), "reviewer_b_counts": dict(cb)}


def main() -> None:
    result = {"packet": str(ROOT), "items": 30, "fields": {}, "disagreement_ids": {}}
    for stage, fields in FIELDS.items():
        a, b = _read("A", stage), _read("B", stage)
        if set(a) != set(b):
            raise ValueError(f"reviewer item IDs differ in {stage}")
        ids = sorted(a)
        for field in fields:
            left = [a[item][field].strip().lower() for item in ids]
            right = [b[item][field].strip().lower() for item in ids]
            allowed = {"yes", "no", "uncertain"}
            if field.endswith("_na_uncertain"):
                allowed.add("na")
            if any(value not in allowed for value in left + right):
                raise ValueError(f"fill {field} with {sorted(allowed)} in both sheets")
            result["fields"][field] = _agreement(left, right)
            result["disagreement_ids"][field] = [
                item for item, x, y in zip(ids, left, right, strict=True) if x != y
            ]
    result["interpretation"] = "Thirty development clips only; agreement is descriptive and should accompany an adjudicated label set."
    path = ROOT / "review_agreement.json"
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"output": str(path), "items": 30,
                      "kappa_by_field": {k: v["cohens_kappa"] for k, v in result["fields"].items()}},
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
