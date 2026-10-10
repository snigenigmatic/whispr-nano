"""Render the night-1 descriptive pilot figures from accepted metrics.csv only.

Regenerate: .venv/bin/python results/night1/figs/plot_pilot.py
"""

from __future__ import annotations

from collections import Counter
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from orx_figstyle import PALETTE, WIDE, figure, save, use_style


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DATASETS = [("mucs_hien", "MUCS Hindi–English"), ("hiacc", "HiACC")]
CONDITIONS = [("A_raw", "A raw"), ("C_script_equiv_provisional", "C provisional")]
METRICS = [("WER", "WER"), ("TokER[LAT]", "Latin TokER"), ("TermER", "candidate TermER")]
COLORS = {"WER": PALETTE["blue"], "TokER[LAT]": PALETTE["orange"], "TermER": PALETTE["red"]}


def load_metrics() -> list[dict]:
    with (ROOT / "metrics.csv").open(newline="") as handle:
        return list(csv.DictReader(handle))


def _get(rows: list[dict], dataset: str, condition: str, metric: str,
         group_field: str = "overall", group_value: str = "all") -> dict | None:
    return next((r for r in rows if r["dataset"] == dataset and r["condition"] == condition
                 and r["metric"] == metric and r["group_field"] == group_field
                 and r["group_value"] == group_value), None)


def forest(rows: list[dict]) -> None:
    fig, ax = figure(width=WIDE, ratio=0.79)
    entries = []
    for dataset, dlabel in DATASETS:
        for condition, clabel in CONDITIONS:
            for metric, mlabel in METRICS:
                row = _get(rows, dataset, condition, metric)
                if row and (metric != "TermER" or int(row["denominator"]) >= 30):
                    entries.append((f"{dlabel} · {clabel} · {mlabel}", metric, row))
    for y, (label, metric, row) in enumerate(reversed(entries)):
        value, lo, hi = [100 * float(row[k]) for k in ("value", "ci_low", "ci_high")]
        ax.plot([lo, hi], [y, y], color=COLORS[metric], linewidth=1.1)
        ax.plot([value], [y], "o", color=COLORS[metric], markersize=3)
    ax.set_yticks(range(len(entries)), [x[0] for x in reversed(entries)])
    ax.set_xlabel("Error rate (%) · 95% speaker-cluster bootstrap CI")
    ax.set_xlim(left=0)
    ax.grid(axis="x", alpha=0.25)
    ax.grid(axis="y", visible=False)
    ax.tick_params(axis="y", length=0)
    save(fig, str(HERE / "forest_metrics"))


def conditions(rows: list[dict]) -> None:
    fig, ax = figure(width=WIDE, ratio=0.45)
    labels, a, c = [], [], []
    for dataset, dlabel in DATASETS:
        for metric, mlabel in METRICS:
            ra = _get(rows, dataset, "A_raw", metric)
            rc = _get(rows, dataset, "C_script_equiv_provisional", metric)
            if ra and rc and (metric != "TermER" or min(int(ra["denominator"]), int(rc["denominator"])) >= 30):
                labels.append(f"{dlabel}\n{mlabel}")
                a.append(100 * float(ra["value"]))
                c.append(100 * float(rc["value"]))
    x = np.arange(len(labels))
    width = 0.38
    bars_a = ax.bar(x - width / 2, a, width, color="#888888", label="A raw", edgecolor="black", linewidth=0.3)
    bars_c = ax.bar(x + width / 2, c, width, color=PALETTE["blue"], label="C provisional", edgecolor="black", linewidth=0.3)
    ax.bar_label(bars_a, fmt="%.1f", fontsize=5, padding=1)
    ax.bar_label(bars_c, fmt="%.1f", fontsize=5, padding=1)
    ax.set_xticks(x, labels)
    ax.set_ylabel("Error rate (%)")
    ax.set_ylim(0, max(a + c) * 1.24 if a or c else 1)
    ax.legend(ncol=2, frameon=False, loc="upper right")
    ax.tick_params(axis="x", length=0)
    save(fig, str(HERE / "conditions_a_c"))


def term_error_stack() -> None:
    with (ROOT / "term_status.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    fig, ax = figure(width=WIDE, ratio=0.36)
    labels = []
    values = []
    for dataset, dlabel in DATASETS:
        for condition, clabel in CONDITIONS:
            counts = Counter(r["status"] for r in rows if r["dataset"] == dataset and r["condition"] == condition)
            if sum(counts.values()) < 30:
                continue
            wrong = sum(counts[k] for k in ("SUBSTITUTED", "DELETED", "PARTIAL"))
            labels.append(f"{dlabel} · {clabel}")
            values.append({k: 100 * counts[k] / wrong if wrong else 0 for k in ("SUBSTITUTED", "DELETED", "PARTIAL")})
    y = np.arange(len(labels))
    left = np.zeros(len(labels))
    for status, color in (("SUBSTITUTED", PALETTE["blue"]),
                          ("DELETED", PALETTE["orange"]),
                          ("PARTIAL", PALETTE["red"])):
        segment = np.array([v[status] for v in values])
        ax.barh(y, segment, left=left, label=status.lower(), color=color,
                edgecolor="black", linewidth=0.3)
        left += segment
    ax.set_yticks(y, labels)
    ax.set_xlabel("Share of wrong candidate TERM spans (%)")
    ax.set_xlim(0, 100)
    ax.legend(ncol=3, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.22))
    ax.invert_yaxis()
    save(fig, str(HERE / "term_error_types"))


def hiacc_groups(rows: list[dict]) -> None:
    fig, ax = figure(width=WIDE, ratio=0.62)
    entries = []
    for field in ("group", "gender"):
        values = sorted({r["group_value"] for r in rows if r["dataset"] == "hiacc" and r["group_field"] == field})
        for value in values:
            for metric in ("WER", "TokER[LAT]"):
                row = _get(rows, "hiacc", "A_raw", metric, field, value)
                if row:
                    entries.append((f"{field}: {value} · {'WER' if metric == 'WER' else 'Latin TokER'}", metric, row))
    for y, (label, metric, row) in enumerate(reversed(entries)):
        point, lo, hi = [100 * float(row[k]) for k in ("value", "ci_low", "ci_high")]
        ax.plot([lo, hi], [y, y], color=COLORS[metric], linewidth=1.0)
        ax.plot(point, y, "o", color=COLORS[metric], markersize=3)
    ax.set_yticks(range(len(entries)), [e[0] for e in reversed(entries)])
    ax.set_xlabel("Error rate (%) · 95% speaker-cluster bootstrap CI")
    ax.set_xlim(left=0)
    ax.grid(axis="x", alpha=0.25)
    ax.grid(axis="y", visible=False)
    ax.tick_params(axis="y", length=0)
    save(fig, str(HERE / "hiacc_groups"))


def main() -> None:
    use_style()
    rows = load_metrics()
    forest(rows)
    conditions(rows)
    term_error_stack()
    hiacc_groups(rows)
    (HERE / "captions.md").write_text(
        "# Draft captions for descriptive pilot figures\n\n"
        "- `forest_metrics`: Error rates on MUCS (forced-Hindi greedy) and pooled HiACC (auto-language greedy) for Whisper large-v3, following the frozen per-corpus decoder policy; intervals are 95% speaker-cluster bootstrap over 1,000 resamples. Candidate TermER uses unreviewed dictionary spans; C is a provisional script equivalence computed from the evaluation corpora.\n"
        "- `conditions_a_c`: Raw and provisional script-equivalent error rates. C is a scoring diagnostic, not a model improvement; its lexicon was discovered from the same corpus.\n"
        "- `term_error_types`: Distribution of substitution, deletion, and mixed errors among wrong dictionary candidate terms, shown only for datasets with at least 30 candidate spans. Denominators are in `metrics.csv`.\n"
        "- `hiacc_groups`: Adult/child and gender slices from pooled HiACC. The released train/validation/test splits share every speaker, and 119 AD63 utterances lack gender metadata. Descriptive only.\n"
    )


if __name__ == "__main__":
    main()
