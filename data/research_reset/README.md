# Frozen Days 1–3 artifacts

Generated from `audit/svarah_fairness_audit/results_whisper-large-v3_svarah_clean.csv` by `PYTHONPATH=src python -m research_reset`. Do not hand-edit IDs.

| File | What |
|---|---|
| `listening_set.json` | 120 clips with sealed reference/teacher text (researchers only) |
| `listening_set.blind.json` | Same IDs, no transcripts — this is what reviewers see |
| `reviewer_a_worksheet.csv` / `reviewer_b_worksheet.csv` | Empty transcript columns |
| `gate_split.json` | Listening / train / eval IDs for the 12-run gate |
| `design.json` | Frozen 2×2×3 design, 2,400 primary, stop rule |
| `run_specs.json` | Twelve paired run specs including recording order |
| `reconciliation.json` | **Not here until both reviewers finish** |

Rebuild:

```bash
PYTHONPATH=src python -m research_reset
```
