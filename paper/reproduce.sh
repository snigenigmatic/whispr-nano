#!/usr/bin/env bash
# Reproduce the paper from stored predictions. This script starts no paid jobs.
set -euo pipefail
paper_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
paper_python="${PAPER_PYTHON:-${paper_root}/.venv/bin/python}"
cd "$paper_root"
"$paper_python" paper/metadata/validate_metadata.py
"$paper_python" -m paper.analysis.analyze \
  --matched-results-dir paper/data/matched_predictions \
  --metadata paper/data/svarah_metadata.csv \
  --cluster-column recording_cluster_id --cluster-kind recording \
  --output-dir paper/results
"$paper_python" -m paper.analysis.analyze \
  --matched-results-dir paper/data/matched_predictions \
  --metadata paper/data/svarah_metadata.csv \
  --cluster-column demographic_profile_id --cluster-kind demographic_profile \
  --output-dir paper/results/demographic_profile
"$paper_python" -m paper.analysis.error_sensitivity \
  --matched-results-dir paper/data/matched_predictions --output-dir paper/results
"$paper_python" paper/figures/build_figures.py
"$paper_python" -m pytest paper/tests audit/svarah_fairness_audit/test_assemble.py -q
"$paper_python" paper/manuscript/build_pdf.py
