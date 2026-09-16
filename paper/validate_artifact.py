"""Final artifact integrity checks; no remote calls or paid work."""
from pathlib import Path
import csv
import hashlib
import json
import re

ROOT = Path(__file__).resolve().parents[1]


def main():
    manifest = json.loads((ROOT/"paper/data/input_manifest.json").read_text())
    for name, source in manifest["source_files"].items():
        path=ROOT/"audit/svarah_fairness_audit"/name
        if hashlib.sha256(path.read_bytes()).hexdigest()!=source["sha256"]:
            raise ValueError(f"Historical source modified: {name}")
    expected=set(manifest["saaras_cohort_ids"])
    paths=sorted((ROOT/"paper/data/matched_predictions").glob("results_*_svarah_clean.csv"))
    if len(paths)!=6:raise ValueError("Matched panel must have exactly six result files")
    for path in paths:
        rows=list(csv.DictReader(path.open()))
        if len(rows)!=450 or {r["utt_id"] for r in rows}!=expected:
            raise ValueError(f"Incomplete or altered matched cohort: {path.name}")
    for directory in (ROOT/"paper/results", ROOT/"paper/results/demographic_profile"):
        analysis=json.loads((directory/"analysis.json").read_text(), parse_constant=lambda x: (_ for _ in ()).throw(ValueError(f"Nonfinite JSON value {x}")))
        matched=[p for p in analysis["cohorts"].values() if p["n_utterances"]==450 and len(p["systems"])==6]
        if len(matched)!=1:raise ValueError("Expected exactly one complete six-model matched cohort")
        for panel in analysis["cohorts"].values():
            if panel["cluster_status"]["inference_supported"]:
                raise ValueError("Unverified grouping must not claim speaker inference")
            for scores in panel["normalizations"].values():
                for contrast in scores["subgroup_contrasts"]+scores["model_contrasts"]:
                    if contrast["approximate_bootstrap_p"] is not None or contrast["holm_adjusted_p"] is not None:
                        raise ValueError("Unverified grouping must not report inferential p-values")
    manuscript=(ROOT/"paper/manuscript/manuscript.md").read_text()
    if re.search(r"<!--\s*(INSERT|INTERNAL DRAFT)|\b(TODO|TBD)\b",manuscript):
        raise ValueError("Unresolved manuscript marker")
    for src in re.findall(r"!\[[^\]]*\]\(([^)]+)\)",manuscript):
        if not (ROOT/"paper/manuscript"/src).resolve().is_file():
            raise ValueError(f"Missing manuscript figure: {src}")
    ledger=json.loads((ROOT/"paper/inference/budget_ledger.json").read_text())
    reserved=sum(j["reserved_usd"] for j in ledger["jobs"])
    if reserved>ledger["authorized_limit"]:raise ValueError("Budget reservations exceed authorization")
    incomplete=[j["id"] for j in ledger["jobs"] if j["status"]=="reserved"]
    if incomplete:raise ValueError(f"Unclosed jobs: {incomplete}")
    print(json.dumps({"historical_sources_unchanged":True,"matched_models":6,"matched_utterances":450,
                      "speaker_inference_withheld":True,"draft_markers":0,
                      "budget_reserved_usd":reserved,"budget_limit_usd":ledger["authorized_limit"]},indent=2))


if __name__=="__main__":main()
