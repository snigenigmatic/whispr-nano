"""Results assembler for the Svarah multi-system audit -- Aditi's AT1.

Reads every already-scored per-utterance CSV this audit produced
(`results_<system>_svarah_clean.csv`, written by `modal_whisper_family_audit.py`
/ `modal_qwen3_asr_audit.py` / `run_saaras_audit.py`), hard-validates the
schema so column drift is caught the moment a new file lands (not discovered
Sunday night), recomputes every reported number directly from the
per-utterance rows (overall WER, per-family WER, ΔDP, Poisson
drop-in-deviance p-value), and emits:

  - `audit/master_audit_table.csv`  -- one row per system, machine-readable
  - `audit/master_audit_table.md`   -- the same table, ready to paste into a deck

If a `summary_<tag>_svarah.json` already exists next to a results CSV (the
per-run script writes one), this also cross-checks the recomputed numbers
against it and hard-fails on any mismatch beyond a small floating-point
tolerance -- so a stale or hand-edited summary can never silently diverge
from the per-utterance source of truth.

This is deliberately independent of `ast_asr`'s package `__init__` (which
pulls in torch via `.objectives`) so it runs anywhere pandas + statsmodels
are installed, no GPU stack required -- it is pure CPU glue over CSVs that
are already fully scored.

Usage:
    python audit/assemble.py                 # assemble everything in audit/
    python audit/assemble.py --strict         # also hard-fail if any expected
                                                # results_*.csv glob finds zero files
    python audit/assemble.py --dir some/path  # assemble from a different directory
    pytest audit/test_assemble.py             # schema + math regression tests
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path

REQUIRED_COLUMNS = [
    "utt_id", "primary_language", "family", "gender", "duration_s",
    "reference_text", "reference_normalized", "hypothesis_text", "hypothesis_normalized",
    "wer", "n_sub", "n_del", "n_ins", "ref_len_words",
]
KNOWN_FAMILIES = {"Indo-Aryan", "Dravidian", "Sino-Tibetan"}
RESULTS_GLOB = "results_*_svarah_clean.csv"
RESULTS_RE = re.compile(r"^results_(?P<system>.+)_svarah_clean\.csv$")
SUMMARY_TOLERANCE = 1e-6


class SchemaError(ValueError):
    """Raised when a results CSV's columns don't match the frozen schema."""


@dataclass
class SystemResult:
    system: str
    n_utterances: int
    overall_wer: float
    per_family_wer: dict[str, float]
    family_n: dict[str, int]
    delta_dp_pp: float | None
    poisson_p: float | None
    verdict: str
    source_csv: str


def _validate_schema(header: list[str], source: str) -> None:
    missing = [c for c in REQUIRED_COLUMNS if c not in header]
    extra = [c for c in header if c not in REQUIRED_COLUMNS]
    if missing or extra:
        parts = []
        if missing:
            parts.append(f"missing columns {missing}")
        if extra:
            parts.append(f"unexpected columns {extra}")
        raise SchemaError(
            f"{source}: schema drift -- {'; '.join(parts)}. "
            f"Expected exactly: {REQUIRED_COLUMNS}"
        )
    if header != REQUIRED_COLUMNS:
        raise SchemaError(
            f"{source}: columns present but out of order -- got {header}, "
            f"expected {REQUIRED_COLUMNS}. Fix the writer, don't reorder here."
        )


def _read_rows(csv_path: Path) -> list[dict[str, str]]:
    import csv

    with csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        _validate_schema(header, csv_path.name)
        return [dict(zip(header, row)) for row in reader]


def _per_family_wer(rows: list[dict]) -> tuple[dict[str, float], dict[str, int]]:
    totals: dict[str, list[int]] = {}
    for r in rows:
        family = r["family"]
        if family not in KNOWN_FAMILIES:
            raise SchemaError(
                f"unknown family {family!r} (utt_id={r['utt_id']!r}); "
                f"expected one of {sorted(KNOWN_FAMILIES)}. "
                f"Check ast_asr.taxonomy.SVARAH_LANGUAGE_FAMILIES for drift."
            )
        n_err = int(r["n_sub"]) + int(r["n_del"]) + int(r["n_ins"])
        ref_len = int(r["ref_len_words"])
        totals.setdefault(family, [0, 0, 0])
        totals[family][0] += n_err
        totals[family][1] += ref_len
        totals[family][2] += 1
    per_family_wer = {f: (err / ref if ref else float("nan")) for f, (err, ref, _) in totals.items()}
    family_n = {f: n for f, (_, _, n) in totals.items()}
    return per_family_wer, family_n


def _fit_poisson_drop_in_deviance(rows: list[dict]) -> dict | None:
    if len({r["family"] for r in rows}) < 2 or len(rows) < 10:
        return None
    import numpy as np
    import pandas as pd
    import statsmodels.api as sm
    import statsmodels.formula.api as smf
    from scipy.stats import chi2

    df = pd.DataFrame(rows)
    df["n_errors"] = df["n_sub"].astype(int) + df["n_del"].astype(int) + df["n_ins"].astype(int)
    df["ref_len_words"] = df["ref_len_words"].astype(int)
    offset = np.log(df["ref_len_words"].astype(float).clip(lower=1))

    null_model = smf.glm("n_errors ~ 1", data=df, family=sm.families.Poisson(), offset=offset).fit()
    alt_model = smf.glm("n_errors ~ C(family)", data=df, family=sm.families.Poisson(), offset=offset).fit()

    deviance_diff = float(null_model.deviance - alt_model.deviance)
    df_diff = int(null_model.df_resid - alt_model.df_resid)
    return {"deviance_diff": deviance_diff, "df_diff": df_diff, "p_value": float(chi2.sf(deviance_diff, df_diff))}


def _check_against_summary(result: SystemResult, csv_path: Path) -> list[str]:
    """Cross-check the recomputed numbers against summary_<tag>_svarah.json,
    if one exists. Returns a list of human-readable mismatch descriptions
    (empty if everything agrees, or if there's no summary to check)."""
    tag = RESULTS_RE.match(csv_path.name).group("system")
    summary_path = csv_path.parent / f"summary_{tag}_svarah.json"
    if not summary_path.exists():
        return []
    summary = json.loads(summary_path.read_text())
    problems = []

    def close(a: float, b: float, label: str) -> None:
        if a is None or b is None:
            return
        if abs(a - b) > SUMMARY_TOLERANCE:
            problems.append(f"{label}: recomputed={a!r} vs summary={b!r}")

    close(result.overall_wer, summary.get("overall_wer"), "overall_wer")
    if result.n_utterances != summary.get("n_utterances"):
        problems.append(f"n_utterances: recomputed={result.n_utterances} vs summary={summary.get('n_utterances')}")
    for family, wer in result.per_family_wer.items():
        close(wer, (summary.get("per_family_wer") or {}).get(family), f"per_family_wer[{family}]")
    return problems


def assemble(directory: Path, strict: bool = False) -> list[SystemResult]:
    csv_paths = sorted(directory.glob(RESULTS_GLOB))
    if not csv_paths:
        msg = f"no files matching {RESULTS_GLOB!r} found in {directory}"
        if strict:
            raise SchemaError(msg)
        print(f"[assemble] warning: {msg}", file=sys.stderr)
        return []

    results: list[SystemResult] = []
    for csv_path in csv_paths:
        m = RESULTS_RE.match(csv_path.name)
        if not m:
            raise SchemaError(f"{csv_path.name} matched the glob but not the naming convention {RESULTS_RE.pattern!r}")
        system = m.group("system")

        rows = _read_rows(csv_path)
        total_err = sum(int(r["n_sub"]) + int(r["n_del"]) + int(r["n_ins"]) for r in rows)
        total_ref = sum(int(r["ref_len_words"]) for r in rows)
        overall_wer = total_err / total_ref if total_ref else float("nan")
        per_family_wer, family_n = _per_family_wer(rows)

        poisson = _fit_poisson_drop_in_deviance(rows)
        delta_dp_pp = (max(per_family_wer.values()) - min(per_family_wer.values())) * 100 if per_family_wer else None
        poisson_p = poisson["p_value"] if poisson else None
        verdict = (
            "structural" if poisson_p is not None and poisson_p < 0.05
            else "not distinguishable from noise" if poisson_p is not None
            else "too few utterances/families for a Poisson fit"
        )

        result = SystemResult(
            system=system, n_utterances=len(rows), overall_wer=overall_wer,
            per_family_wer=per_family_wer, family_n=family_n,
            delta_dp_pp=delta_dp_pp, poisson_p=poisson_p, verdict=verdict,
            source_csv=csv_path.name,
        )

        mismatches = _check_against_summary(result, csv_path)
        if mismatches:
            raise SchemaError(
                f"{csv_path.name}: recomputed numbers disagree with "
                f"summary_{system}_svarah.json -- {'; '.join(mismatches)}"
            )

        results.append(result)
        print(f"[assemble] {system}: {len(rows)} utterances, overall WER "
              f"{overall_wer * 100:.2f}%, verdict={verdict}", flush=True)

    return results


def _write_outputs(results: list[SystemResult], directory: Path) -> None:
    families = sorted({f for r in results for f in r.per_family_wer})

    csv_path = directory / "master_audit_table.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        import csv as csv_mod

        columns = ["system", "n_utterances", "overall_wer_pct"] + \
            [f"{fam}_wer_pct" for fam in families] + ["delta_dp_pp", "poisson_p", "verdict", "source_csv"]
        writer = csv_mod.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for r in results:
            row = {
                "system": r.system,
                "n_utterances": r.n_utterances,
                "overall_wer_pct": round(r.overall_wer * 100, 4),
                "delta_dp_pp": round(r.delta_dp_pp, 4) if r.delta_dp_pp is not None else "",
                "poisson_p": r.poisson_p if r.poisson_p is not None else "",
                "verdict": r.verdict,
                "source_csv": r.source_csv,
            }
            for fam in families:
                row[f"{fam}_wer_pct"] = round(r.per_family_wer[fam] * 100, 4) if fam in r.per_family_wer else ""
            writer.writerow(row)
    print(f"\nWrote {csv_path}")

    md_path = directory / "master_audit_table.md"
    header = ["System", "Overall WER", *families, "ΔDP (pp)", "Poisson p", "n", "Verdict"]
    lines = [
        "| " + " | ".join(header) + " |",
        "|" + "|".join(["---"] * len(header)) + "|",
    ]
    for r in results:
        cells = [
            r.system,
            f"{r.overall_wer * 100:.2f}%",
            *[f"{r.per_family_wer.get(fam, float('nan')) * 100:.2f}%" for fam in families],
            f"{r.delta_dp_pp:.2f}" if r.delta_dp_pp is not None else "n/a",
            f"{r.poisson_p:.4g}" if r.poisson_p is not None else "n/a",
            str(r.n_utterances),
            r.verdict,
        ]
        lines.append("| " + " | ".join(cells) + " |")
    md_path.write_text("\n".join(lines) + "\n")
    print(f"Wrote {md_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dir", type=Path, default=Path(__file__).parent,
                         help="Directory to glob results_*_svarah_clean.csv from (default: this script's directory)")
    parser.add_argument("--strict", action="store_true",
                         help="Hard-fail (instead of warn) if no results files are found")
    args = parser.parse_args()

    results = assemble(args.dir, strict=args.strict)
    if results:
        _write_outputs(results, args.dir)
        print(f"\n{len(results)} system(s) assembled cleanly, no schema drift, no summary/CSV mismatches.")


if __name__ == "__main__":
    main()
