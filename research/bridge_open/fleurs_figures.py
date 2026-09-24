"""Render final, evidence-bound FLEURS Turbo versus Large-v3 comparison figures.

The script intentionally refuses partial or differently pinned comparisons.  It
contains no fallback data: every plotted value and interval is read from the
completed ``analyze_fleurs.py`` comparison JSON.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np

SUPPORT = Path(__file__).with_name("figure_support")
sys.path.insert(0, str(SUPPORT))
from orx_figstyle import BASELINE, MUTED, PALETTE, TEXT, figure, save, use_style  # noqa: E402


TURBO = ("openai/whisper-large-v3-turbo", "41f01f3fe87f28c78e2fbf8b568835947dd65ed9")
LARGE = ("openai/whisper-large-v3", "06f233fe06e710322aca913c1bc4249a0d71fce1")
LANGUAGES = (("as", "Assamese", 984), ("bn", "Bengali", 920), ("hi", "Hindi", 418))


def _models(report: dict[str, Any]) -> dict[tuple[str, str], tuple[str, dict[str, Any]]]:
    models = report.get("models")
    if not isinstance(models, dict) or len(models) != 2:
        raise ValueError("comparison must contain exactly two models")
    resolved = {}
    for label, model in models.items():
        key = (model.get("model_id"), model.get("model_revision"))
        if key in resolved:
            raise ValueError(f"duplicate model identity {key}")
        resolved[key] = (label, model)
    if set(resolved) != {TURBO, LARGE}:
        raise ValueError("comparison must contain only the pinned Turbo and Large-v3 revisions")
    return resolved


def load_comparison(path: str | Path) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    """Validate the complete cohort contract and return observed plot values."""
    report = json.loads(Path(path).read_text())
    models = _models(report)
    turbo_label, turbo = models[TURBO]
    large_label, large = models[LARGE]
    rows = []
    for code, display, expected_count in LANGUAGES:
        for model_name, model in (("Turbo", turbo), ("Large-v3", large)):
            language = model.get("per_language", {}).get(code)
            combined = language.get("combined") if isinstance(language, dict) else None
            if not isinstance(combined, dict) or combined.get("scored_count") != expected_count or combined.get("utterances") != expected_count:
                raise ValueError(f"{model_name} {code}: incomplete cohort; expected {expected_count} scored utterances")
            value = combined.get("micro_wer")
            if not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                raise ValueError(f"{model_name} {code}: invalid observed micro WER")
            rows.append({"language": code, "display": display, "model": model_name, "wer": float(value)})

    pairs = report.get("paired_differences")
    if not isinstance(pairs, list):
        raise ValueError("comparison lacks paired differences")
    by_language = {}
    for pair in pairs:
        if pair.get("left") != turbo_label or pair.get("right") != large_label or pair.get("metric") != "micro_wer":
            continue
        language = pair.get("language")
        if language in by_language:
            raise ValueError(f"duplicate paired interval for {language}")
        ci = pair.get("ci_95")
        if pair.get("draws") != 2000 or pair.get("seed") != 12 or not isinstance(pair.get("cluster_count"), int) or pair["cluster_count"] < 1:
            raise ValueError(f"{language}: paired interval lacks the fixed bootstrap contract")
        if not isinstance(ci, list) or len(ci) != 2 or not all(isinstance(value, (int, float)) and math.isfinite(value) for value in [pair.get("difference"), *ci]):
            raise ValueError(f"{language}: malformed paired interval")
        if ci[0] > pair["difference"] or pair["difference"] > ci[1]:
            raise ValueError(f"{language}: observed difference lies outside its interval")
        by_language[language] = pair
    expected_languages = {code for code, _, _ in LANGUAGES}
    if set(by_language) != expected_languages:
        raise ValueError("paired intervals must cover exactly Assamese, Bengali, and Hindi")
    return {"turbo_label": turbo_label, "large_label": large_label}, rows, [by_language[code] for code, _, _ in LANGUAGES]


def _wer_bar_figure(observed: dict[str, dict[str, Any]], output: Path) -> None:
    use_style()
    fig, ax = figure(width=TEXT, ratio=.43)
    x = np.arange(len(LANGUAGES))
    width = .36
    # Convert the flat validated rows into fixed-order method series.
    by_method = {"Turbo": [], "Large-v3": []}
    for code, _, _ in LANGUAGES:
        for row in observed[code]["rows"]:
            by_method[row["model"]].append(row["wer"] * 100)
    bars_turbo = ax.bar(x - width / 2, by_method["Turbo"], width, label="Turbo", color=PALETTE["blue"], edgecolor="black", linewidth=.4)
    bars_large = ax.bar(x + width / 2, by_method["Large-v3"], width, label="Large-v3", color=MUTED, edgecolor="black", linewidth=.4)
    for bars in (bars_turbo, bars_large):
        ax.bar_label(bars, fmt="%.1f", fontsize=5, padding=1.5)
    highest = max(by_method["Turbo"] + by_method["Large-v3"])
    top = max(10, 10 * math.ceil(highest * 1.12 / 10))
    ax.set_ylim(0, top)
    ax.set_yticks(np.arange(0, top + 1, 10))
    ax.set_ylabel("Micro WER (%)")
    ax.set_xticks(x, [display for _, display, _ in LANGUAGES])
    ax.tick_params(axis="x", length=0)
    ax.legend(loc="upper left", ncol=2, frameon=True, framealpha=1, edgecolor="black", borderpad=.4)
    save(fig, str(output / "fleurs_wer_grouped"))


def _paired_figure(pairs: list[dict[str, Any]], output: Path) -> None:
    use_style()
    fig, ax = figure(width=TEXT, ratio=.32)
    ax.axvline(0, color=BASELINE, linewidth=.8, zorder=1)
    values = []
    for y, ((_, display, _), pair) in enumerate(zip(LANGUAGES, pairs)):
        # This deliberately reads the paired difference interval rather than
        # deriving it from two model-level intervals.
        estimate = pair["difference"] * 100
        low, high = (value * 100 for value in pair["ci_95"])
        values.extend((low, high))
        ax.plot([low, high], [y, y], color=PALETTE["blue"], linewidth=1.1, solid_capstyle="round", zorder=2)
        ax.plot(estimate, y, "o", color=PALETTE["blue"], zorder=3)
        ax.annotate(f"{estimate:+.1f}", xy=(high, y), xytext=(4, 0), textcoords="offset points", va="center", fontsize=7)
    ax.set_yticks(range(len(LANGUAGES)), [display for _, display, _ in LANGUAGES])
    ax.set_ylim(-.6, len(LANGUAGES) - .4)
    ax.set_xlabel("Turbo minus Large-v3 micro WER (percentage points)")
    ax.grid(axis="x")
    ax.grid(axis="y", visible=False)
    ax.tick_params(axis="y", length=0)
    extent = max(abs(value) for value in values + [0])
    ax.set_xlim(-extent * 1.3 - .1, extent * 1.3 + .1)
    save(fig, str(output / "fleurs_paired_wer_difference"))


def render(comparison_path: str | Path, output_dir: str | Path) -> None:
    labels, rows, pairs = load_comparison(comparison_path)
    observed = {code: {"rows": [row for row in rows if row["language"] == code]} for code, _, _ in LANGUAGES}
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    _wer_bar_figure(observed, output)
    _paired_figure(pairs, output)
    caption = (
        "FLEURS official-test micro WER for the pinned Whisper Large-v3 Turbo and Large-v3 recipes. "
        "Bars begin at zero and include all scored utterances, including capped and normalized-empty outputs. "
        "Dots show Turbo minus Large-v3 WER; horizontal lines are 95% intervals from 2,000 seed-12 bootstrap draws over source-sentence clusters within language. "
        "The intervals describe source-sentence cluster resampling for deterministic seed-12 decoding, not multiple training seeds, speaker uncertainty, or population uncertainty."
    )
    (output / "fleurs_figures_caption.txt").write_text(caption + "\n")
    print(json.dumps({"event": "fleurs_figures_written", "comparison": str(comparison_path), "output_dir": str(output), **labels}), flush=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comparison", required=True, help="completed analyze_fleurs comparison.json")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args(argv)
    render(args.comparison, args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
