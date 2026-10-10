"""Validate cached ASR outputs and score the public-corpus pilot."""

from __future__ import annotations

from collections import Counter, defaultdict
import csv
import json
from pathlib import Path
import random

import pandas as pd

from ast_asr.termeval.lexicon import approved_pairs, build_lexicon
from ast_asr.termeval.metrics import cluster_bootstrap, pooled_metrics, score_utterance


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data/termeval/manifest.jsonl"
OUT = ROOT / "results/night1"
SEED = 20261011
MODEL_REVISION = "06f233fe06e710322aca913c1bc4249a0d71fce1"


def load_inputs() -> tuple[dict[str, dict], list[dict], dict[str, dict]]:
    rows = [json.loads(line) for line in DATA.open()]
    manifest = {r["utt_id"]: r for r in rows}
    if len(manifest) != len(rows):
        raise ValueError("duplicate manifest utterance IDs")
    predictions = []
    for dataset in ("hiacc", "mucs_hien"):
        file = OUT / f"predictions_{dataset}.jsonl"
        if not file.exists():
            raise FileNotFoundError(file)
        subset = [json.loads(line) for line in file.open()]
        expected_ids = {r["utt_id"] for r in rows if r["dataset"] == dataset}
        if len(subset) != len(expected_ids) or {r["utt_id"] for r in subset} != expected_ids:
            raise ValueError(f"{dataset} prediction cache is incomplete or duplicated")
        hashes = {p["config_hash"] for p in subset}
        if len(hashes) != 1:
            raise ValueError(f"{dataset} has mixed config hashes")
        for p in subset:
            if p["dataset"] != dataset or p["system"] != "whisper-large-v3":
                raise ValueError("prediction identity mismatch")
            if p["model_revision"] != MODEL_REVISION:
                raise ValueError("model revision mismatch")
        predictions.extend(subset)
    annotations_file = OUT / "annotations_heuristic_v0.jsonl"
    annotations = {r["utt_id"]: r for r in map(json.loads, annotations_file.open())}
    if len(annotations) != 200:
        raise ValueError("annotation sample is not 200 unique utterances")
    return manifest, predictions, annotations


def main() -> None:
    manifest, predictions, annotations = load_inputs()
    lexicon = build_lexicon(predictions, {u: r["reference"] for u, r in manifest.items()})
    with (OUT / "lexicon_v0.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["latin", "devanagari", "romanized", "dist",
                                                       "status", "n_occurrences", "example_utt"])
        writer.writeheader()
        writer.writerows(lexicon)
    pairs = approved_pairs(lexicon)
    scored: dict[tuple[str, str], list] = defaultdict(list)
    by_utt: dict[tuple[str, str], object] = {}
    for condition, equivalences in (("A_raw", None), ("C_script_equiv_provisional", pairs)):
        for prediction in predictions:
            row = manifest[prediction["utt_id"]]
            annotated = annotations.get(prediction["utt_id"])
            result = score_utterance(
                prediction["utt_id"], row["speaker"], row["reference"], prediction["text"],
                equivalences=equivalences,
                spans=annotated["spans"] if annotated is not None else None,
            )
            scored[(row["dataset"], condition)].append(result)
            by_utt[(result.utt_id, condition)] = result

    # Per-token tables contain almost all reference text; keep them local only.
    for (dataset, condition), rows in scored.items():
        table = [token for row in rows for token in row.token_rows]
        pd.DataFrame(table, columns=["utt_id", "speaker", "ref_idx", "ref_tok", "hyp_tok",
                                     "op", "tok_class", "in_term", "term_id"]).to_parquet(
                                         OUT / f"tokens_{dataset}_whisper-large-v3_{condition}.parquet", index=False
                                     )
    term_status_rows = [
        {"dataset": dataset, "condition": condition, **term}
        for (dataset, condition), rows in scored.items()
        for row in rows for term in row.term_rows
    ]
    with (OUT / "term_status.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["dataset", "condition", "utt_id", "speaker",
                                                       "term_id", "start", "end", "status",
                                                       "boundary_ins", "internal_ins", "text"])
        writer.writeheader()
        writer.writerows(term_status_rows)

    metrics_rows = []

    def add_metrics(dataset: str, condition: str, group_field: str, group_value: str, rows: list) -> None:
        point = pooled_metrics(rows)
        intervals = cluster_bootstrap(rows, n_resamples=1000, seed=SEED)
        for metric, (value, num, den) in point.items():
            if den == 0:
                continue
            if group_field != "overall" and metric == "TermER" and den < 30:
                continue
            lower, upper = intervals.get(metric, (None, None))
            metrics_rows.append({"dataset": dataset, "system": "whisper-large-v3",
                                 "condition": condition, "group_field": group_field,
                                 "group_value": group_value, "metric": metric,
                                 "value": value, "ci_low": lower, "ci_high": upper,
                                 "numerator": num, "denominator": den,
                                 "n_utterances": len(rows),
                                 "n_speakers": len({r.speaker for r in rows}),
                                 "annotation_method": "dictionary_candidates_unverified_v0" if metric == "TermER" else "ground_truth_transcript"})

    for (dataset, condition), rows in scored.items():
        add_metrics(dataset, condition, "overall", "all", rows)
        if dataset == "hiacc":
            for field in ("group", "gender"):
                values = sorted({manifest[row.utt_id].get(field) for row in rows if manifest[row.utt_id].get(field)})
                grouped = {value: [row for row in rows if manifest[row.utt_id].get(field) == value]
                           for value in values}
                # C7 support threshold: each reported level needs 10 speaker IDs.
                if field != "group" and any(len({r.speaker for r in part}) < 10 for part in grouped.values()):
                    continue
                for value, part in grouped.items():
                    add_metrics(dataset, condition, field, value, part)
    with (OUT / "metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(metrics_rows[0]))
        writer.writeheader()
        writer.writerows(metrics_rows)

    # Random term-error examples, never selected by magnitude or narrative fit.
    candidates = []
    for dataset in ("hiacc", "mucs_hien"):
        for row in scored[(dataset, "A_raw")]:
            for term in row.term_rows:
                if term["status"] != "CORRECT":
                    candidates.append((dataset, row.utt_id, term))
    rng = random.Random(SEED)
    selected = rng.sample(candidates, min(40, len(candidates)))
    lines = ["# Seed-selected candidate term errors", "",
             "These are dictionary-generated, unverified TERM candidates. The examples are randomly selected from A-raw errors with seed 20261011.", ""]
    for i, (dataset, utt, term) in enumerate(selected, 1):
        pred = next(p for p in predictions if p["utt_id"] == utt)
        a = by_utt[(utt, "A_raw")]
        c = by_utt[(utt, "C_script_equiv_provisional")]
        cterm = next(t for t in c.term_rows if t["term_id"] == term["term_id"])
        ops_a = [e.op for e in a.edits if e.ref_idx is not None and term["start"] <= e.ref_idx < term["end"]]
        ops_c = [e.op for e in c.edits if e.ref_idx is not None and term["start"] <= e.ref_idx < term["end"]]
        lines += [f"## {i}. {dataset}/{utt} term {term['term_id']}", "",
                  f"- Reference: {manifest[utt]['reference']}", f"- Hypothesis: {pred['text']}",
                  f"- TERM: {term['text']}",
                  f"- A: {term['status']} ({', '.join(ops_a)})",
                  f"- C: {cterm['status']} ({', '.join(ops_c)})", ""]
    (OUT / "examples_term_errors.md").write_text("\n".join(lines))
    summary = {"predictions": len(predictions), "config_hashes": {
        dataset: sorted({p["config_hash"] for p in predictions if p["dataset"] == dataset})[0]
        for dataset in ("hiacc", "mucs_hien")},
        "lexicon_status": dict(Counter(r["status"] for r in lexicon)),
        "lexicon_auto_accept_pairs": len(pairs),
        "annotations": len(annotations), "candidate_term_spans": sum(
            len([s for s in a["spans"] if s["label"] == "TERM"]) for a in annotations.values()),
        "sampled_term_errors": len(selected), "available_term_errors": len(candidates),
        "bootstrap_resamples": 1000, "bootstrap_unit": "speaker ID"}
    (OUT / "score_receipt.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
