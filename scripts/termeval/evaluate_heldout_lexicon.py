"""Measure provisional script equivalence on speakers unseen by its lexicon."""

from __future__ import annotations

from collections import Counter, defaultdict
import csv
import json
from pathlib import Path
import random

from ast_asr.termeval.lexicon import approved_pairs, build_lexicon
from ast_asr.termeval.metrics import _metrics_from_totals, cluster_bootstrap, pooled_metrics, score_utterance

from .score_results import OUT, SEED, load_inputs


def _paired_delta(a_rows: list, c_rows: list, metric: str) -> tuple[float, float, float]:
    a_by = defaultdict(Counter)
    c_by = defaultdict(Counter)
    for row in a_rows:
        a_by[row.speaker].update(row.totals)
    for row in c_rows:
        c_by[row.speaker].update(row.totals)
    speakers = sorted(a_by)
    if speakers != sorted(c_by):
        raise ValueError("A/C speaker sets differ")
    point_a = pooled_metrics(a_rows)[metric][0]
    point_c = pooled_metrics(c_rows)[metric][0]
    if point_a is None or point_c is None:
        raise ValueError(f"missing denominator for {metric}")
    values = []
    rng = random.Random(SEED)
    for _ in range(1000):
        at, ct = Counter(), Counter()
        for speaker in rng.choices(speakers, k=len(speakers)):
            at.update(a_by[speaker])
            ct.update(c_by[speaker])
        av = _metrics_from_totals(at)[metric][0]
        cv = _metrics_from_totals(ct)[metric][0]
        if av is not None and cv is not None:
            values.append(100 * (av - cv))
    values.sort()
    return 100 * (point_a - point_c), values[int(0.025 * (len(values) - 1))], values[int(0.975 * (len(values) - 1))]


def main() -> None:
    manifest, predictions, _ = load_inputs()
    split = json.loads((OUT / "speaker_holdout_split.json").read_text())
    assignments = split["assignments"]
    if set(assignments) != {r["speaker"] for r in manifest.values()}:
        raise ValueError("speaker split does not cover manifest")
    metrics = []
    receipt = {"seed": SEED, "split_file": "speaker_holdout_split.json",
               "lexicon_mined_on": "lexicon_dev speakers only", "datasets": {}}
    for dataset in ("mucs_hien", "hiacc"):
        d_preds = [p for p in predictions if p["dataset"] == dataset]
        dev = [p for p in d_preds if assignments[manifest[p["utt_id"]]["speaker"]] == "lexicon_dev"]
        test = [p for p in d_preds if assignments[manifest[p["utt_id"]]["speaker"]] == "heldout_eval"]
        lexicon = build_lexicon(dev, {p["utt_id"]: manifest[p["utt_id"]]["reference"] for p in dev})
        with (OUT / f"lexicon_dev_v0_{dataset}.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["latin", "devanagari", "romanized", "dist",
                                                           "status", "n_occurrences", "example_utt"])
            writer.writeheader()
            writer.writerows(lexicon)
        pairs = approved_pairs(lexicon)
        a_rows, c_rows = [], []
        for prediction in test:
            row = manifest[prediction["utt_id"]]
            a_rows.append(score_utterance(prediction["utt_id"], row["speaker"], row["reference"], prediction["text"]))
            c_rows.append(score_utterance(prediction["utt_id"], row["speaker"], row["reference"], prediction["text"], equivalences=pairs))
        for condition, rows in (("A_raw", a_rows), ("C_dev_lexicon_provisional", c_rows)):
            point = pooled_metrics(rows)
            ci = cluster_bootstrap(rows, n_resamples=1000, seed=SEED)
            for metric in ("WER", "CER", "TokER[LAT]", "TokER[DEV]", "INS_rate"):
                value, numerator, denominator = point[metric]
                if value is None:
                    continue
                low, high = ci[metric]
                metrics.append({"dataset": dataset, "condition": condition, "metric": metric,
                                "value": value, "ci_low": low, "ci_high": high,
                                "numerator": numerator, "denominator": denominator,
                                "n_utterances": len(rows), "n_speakers": len({r.speaker for r in rows}),
                                "lexicon_pairs": len(pairs)})
        receipt["datasets"][dataset] = {
            "dev_utterances": len(dev), "heldout_utterances": len(test),
            "dev_speakers": len({manifest[p["utt_id"]]["speaker"] for p in dev}),
            "heldout_speakers": len({manifest[p["utt_id"]]["speaker"] for p in test}),
            "lexicon_status": dict(Counter(r["status"] for r in lexicon)),
            "auto_accept_pairs": len(pairs),
            "paired_delta_wer_pp": _paired_delta(a_rows, c_rows, "WER"),
            "paired_delta_latin_toker_pp": _paired_delta(a_rows, c_rows, "TokER[LAT]"),
        }
    with (OUT / "heldout_lexicon_metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(metrics[0]))
        writer.writeheader()
        writer.writerows(metrics)
    (OUT / "heldout_lexicon_receipt.json").write_text(json.dumps(receipt, indent=2))
    print(json.dumps(receipt))


if __name__ == "__main__":
    main()
