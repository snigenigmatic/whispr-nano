"""Assemble a conservative, traceable night-1 report from local result files."""

from __future__ import annotations

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/night1"


def _pct(value: str | float | None) -> str:
    return "—" if value in (None, "") else f"{100 * float(value):.2f}"


def _row(rows: list[dict], dataset: str, condition: str, metric: str) -> dict | None:
    return next((r for r in rows if r["dataset"] == dataset and r["condition"] == condition
                 and r["group_field"] == "overall" and r["metric"] == metric), None)


def main() -> None:
    with (OUT / "metrics.csv").open(newline="") as handle:
        metrics = list(csv.DictReader(handle))
    with (OUT / "lexicon_v0.csv").open(newline="") as handle:
        lexicon = list(csv.DictReader(handle))
    data = json.loads((OUT / "data_summary.json").read_text())
    audio_qc = json.loads((OUT / "audio_qc.json").read_text())
    prediction_qc = json.loads((OUT / "prediction_qc.json").read_text())
    coverage = json.loads((OUT / "annotation_coverage.json").read_text())
    score = json.loads((OUT / "score_receipt.json").read_text())
    heldout = json.loads((OUT / "heldout_lexicon_receipt.json").read_text())
    with (OUT / "heldout_lexicon_metrics.csv").open(newline="") as handle:
        heldout_metrics = list(csv.DictReader(handle))
    qc = json.loads((OUT / "language_conditioning_qc.json").read_text())
    receipts = {d: json.loads((OUT / f"receipt_{d}.json").read_text()) for d in ("hiacc", "mucs_hien")}
    if any(not r["complete"] for r in receipts.values()):
        raise ValueError("cannot write final report with an incomplete GPU receipt")
    figures = ["forest_metrics.pdf", "conditions_a_c.pdf", "term_error_types.pdf", "hiacc_groups.pdf"]
    missing_figures = [name for name in figures if not (OUT / "figs" / name).exists()]
    if missing_figures:
        raise FileNotFoundError(f"missing C8 figures: {missing_figures}")
    forced = qc["matched_forced_hi"]["WER"]
    auto = qc["matched_auto"]["WER"]
    lines = [
        "# Night 1 term-level ASR pilot — evidence report",
        "",
        "[OBSERVED] This is a public-corpus feasibility audit for C. Kaustubh's 16 October 2026 review. It is **not a submission-ready fairness or domain-term result**. All numbers below come from the linked local artifacts; the two restricted datasets remain unexamined.",
        "",
        "## Status",
        "",
        "| Step | Status | Evidence / reason |",
        "|---|---|---|",
        "| C1 code inventory | Done | [`C1_inventory.md`](C1_inventory.md) |",
        "| C2 datasets | Partial | MUCS and HiACC verified; Lahaja and VITB-HEBiC returned HTTP 401 without accepted access. [`data_card.md`](data_card.md) |",
        "| C3 normalization/scoring | Done for public data | Devanagari-safe normalizer, jiwer-matched raw alignment, pooled metrics and 1,000-resample speaker-cluster CIs; `tests/termeval/`. |",
        "| C4 ASR decode | Done for public data | Whisper large-v3 auto-language greedy over every public-corpus clip; cache and runtime receipts. Forced-Hindi diagnostic stopped after QC failure. Saaras skipped because vendor permission and a call cap were unavailable. |",
        "| C5 script lexicon | Provisional | Corpus-derived candidates, explicit collision/low-support guards, no human approval. [`lexicon_v0.csv`](lexicon_v0.csv) |",
        "| C6 TERM annotation | Partial | 200 speaker-balanced utterances receive reproducible dictionary candidate spans, **not LLM or human labels**. [`annotation_guideline.md`](annotation_guideline.md) |",
        "| C7 group slices | Descriptive | HiACC adult/child and gender only where speaker support is adequate; no hypothesis tests. [`metrics.csv`](metrics.csv) |",
        "| C8 plots/examples | Done with support limits | Vector plots in [`figs/`](figs/) and seed-selected candidate errors in [`examples_term_errors.md`](examples_term_errors.md). |",
        "",
        "## Data card summary",
        "",
        f"[OBSERVED] MUCS Hindi-English test: {data['mucs_hien']['utterances']:,} sentence segments from {data['mucs_hien']['recordings']} long recordings and {data['mucs_hien']['speakers']} speaker IDs; {data['mucs_hien']['duration_hours']:.3f} hours by segment timestamps. The full archive SHA-256 and source license are in [`data_card.md`](data_card.md).",
        f"[OBSERVED] HiACC: {data['hiacc']['total_utterances']:,} clips, {data['hiacc']['total_speakers']} speaker IDs, {data['hiacc']['total_hours']:.3f} hours by WAV headers. Adult and child speaker IDs recur in **all three** released splits. `AD63` has 119 utterances but no demographic row; `AD65` has a metadata row but no audio. The released code-switch labels are utterance-level.",
        "[OBSERVED] Lahaja and VITB-HEBiC content could not be accessed in this runtime. Their metadata claims were not converted into measured counts.",
        f"[OBSERVED] Every local clip was reread before scoring: all {data['mucs_hien']['utterances'] + data['hiacc']['total_utterances']:,} are mono 16 kHz with manifest duration agreement and no exact duplicate audio hashes. MUCS has {audio_qc['mucs_hien']['counts'].get('near_silent_rms_lt_0.001', 0)} near-silent clips and {audio_qc['mucs_hien']['counts'].get('saturated_samples_ge_0.1_percent', 0)} clip with ≥0.1% saturated samples; all remain in the denominator. [`audio_qc.json`](audio_qc.json)",
        "",
        "## Headline metrics",
        "",
        "[OBSERVED] Rates are pooled by reference-token or character counts. CER counts spaces as characters. Brackets are 95% percentile CIs from 1,000 speaker-cluster resamples. `C` is a **provisional scoring condition**, not a different ASR model. `TermER` uses only the 200-item dictionary-candidate sample and is omitted when fewer than 30 candidate spans exist.",
        "",
        "| Dataset | Scoring | Metric | Rate % [95% CI] | Errors / denominator | Utterances / speakers |",
        "|---|---|---|---:|---:|---:|",
    ]
    for dataset in ("mucs_hien", "hiacc"):
        for condition in ("A_raw", "C_script_equiv_provisional"):
            for metric in ("WER", "CER", "TokER[LAT]", "TokER[DEV]", "TokER[NUM]", "TokER[MIX]", "TokER[OTHER]", "INS_rate", "TermER"):
                row = _row(metrics, dataset, condition, metric)
                if row is None or (metric == "TermER" and int(row["denominator"]) < 30):
                    continue
                lines.append(f"| {dataset} | {condition} | {metric} | {_pct(row['value'])} [{_pct(row['ci_low'])}, {_pct(row['ci_high'])}] | {row['numerator']} / {row['denominator']} | {row['n_utterances']} / {row['n_speakers']} |")
    lines += [
        "", "## Change from A to C", "",
        "[OBSERVED] Absolute percentage-point differences (A minus C); these quantify spelling-equivalence accounting only. The candidate lexicon was discovered from this evaluation material, so they are optimistic diagnostics and not held-out gains.",
        "", "| Dataset | Metric | A % | C % | A − C points |", "|---|---|---:|---:|---:|",
    ]
    for dataset in ("mucs_hien", "hiacc"):
        for metric in ("WER", "CER", "TokER[LAT]", "TokER[DEV]", "TokER[NUM]", "TokER[MIX]", "TokER[OTHER]", "INS_rate", "TermER"):
            a = _row(metrics, dataset, "A_raw", metric)
            c = _row(metrics, dataset, "C_script_equiv_provisional", metric)
            if a is None or c is None or (metric == "TermER" and int(a["denominator"]) < 30):
                continue
            lines.append(f"| {dataset} | {metric} | {_pct(a['value'])} | {_pct(c['value'])} | {100 * (float(a['value']) - float(c['value'])):+.2f} |")
    lines += [
        "", "## Provisional lexicon and annotation support", "",
        f"[OBSERVED] Same-corpus candidate lexicon: {len(lexicon)} Latin–Devanagari pairs; {score['lexicon_auto_accept_pairs']} automatically accepted. Status counts: {score['lexicon_status']}. It has no human approvals. The first 30 pairs by observed substitution count follow; complete list and examples are in [`lexicon_v0.csv`](lexicon_v0.csv).",
        "", "| Latin | Devanagari | Substitutions | Status |", "|---|---|---:|---|",
    ]
    for pair in sorted(lexicon, key=lambda p: -int(p["n_occurrences"]))[:30]:
        lines.append(f"| {pair['latin']} | {pair['devanagari']} | {pair['n_occurrences']} | {pair['status']} |")
    lines += [
        "", f"[OBSERVED] The 200 sampled utterances contain {score['candidate_term_spans']} dictionary-generated TERM spans. Across the full reference corpora, this dictionary matches {coverage['mucs_hien']['candidate_TERM_spans']} MUCS and only {coverage['hiacc']['candidate_TERM_spans']} HiACC candidate TERM spans. The HiACC sample cannot support a term-error comparison. No LLM annotation service or human reviewer was used. The exact candidate guideline, coverage, and sample IDs are in [`annotation_guideline.md`](annotation_guideline.md), [`annotation_coverage.json`](annotation_coverage.json), and [`annotation_sample_ids.json`](annotation_sample_ids.json).",
    ]
    lines += [
        "", "## Speaker-heldout script-equivalence control", "",
        "[OBSERVED] A speaker-disjoint lexicon-development/evaluation split was frozen before the full decode completed. Only development-speaker references and predictions generate pairs; no heldout reference enters its lexicon. The pairs remain automatic and unreviewed. This control is more defensible than the same-corpus C diagnostic above, but it is still a **scoring protocol**, not a trained ASR improvement.",
        "", "| Dataset | Dev / heldout speakers | Accepted pairs | Heldout WER A % | Heldout WER C % | Paired A − C points [95% CI] |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for dataset in ("mucs_hien", "hiacc"):
        r = heldout["datasets"][dataset]
        a = next(x for x in heldout_metrics if x["dataset"] == dataset and x["condition"] == "A_raw" and x["metric"] == "WER")
        c = next(x for x in heldout_metrics if x["dataset"] == dataset and x["condition"] == "C_dev_lexicon_provisional" and x["metric"] == "WER")
        delta, low, high = r["paired_delta_wer_pp"]
        lines.append(f"| {dataset} | {r['dev_speakers']} / {r['heldout_speakers']} | {r['auto_accept_pairs']} | {_pct(a['value'])} | {_pct(c['value'])} | {delta:+.2f} [{low:+.2f}, {high:+.2f}] |")
    lines += ["[OBSERVED] Complete paired rates and denominators are in [`heldout_lexicon_metrics.csv`](heldout_lexicon_metrics.csv); the frozen speaker assignment is in [`speaker_holdout_split.json`](speaker_holdout_split.json).", ""]
    lines += [
        "## HiACC descriptive group slices", "",
        "[OBSERVED] These are pooled A-raw rates by released group metadata, with speakers as the bootstrap unit. Adult/child and gender are not causal or population-fairness comparisons. The full A/C matrix is in [`metrics.csv`](metrics.csv).",
        "", "| Field | Level | Metric | Rate % [95% CI] | Errors / denominator | Utterances / speakers |",
        "|---|---|---|---:|---:|---:|",
    ]
    for field in ("group", "gender"):
        values = sorted({r["group_value"] for r in metrics if r["dataset"] == "hiacc" and r["group_field"] == field})
        for value in values:
            for metric in ("WER", "TokER[LAT]", "TokER[DEV]", "INS_rate", "TermER"):
                row = next((r for r in metrics if r["dataset"] == "hiacc" and r["condition"] == "A_raw"
                            and r["group_field"] == field and r["group_value"] == value
                            and r["metric"] == metric), None)
                if row is None or (metric == "TermER" and int(row["denominator"]) < 30):
                    continue
                lines.append(f"| {field} | {value} | {metric} | {_pct(row['value'])} [{_pct(row['ci_low'])}, {_pct(row['ci_high'])}] | {row['numerator']} / {row['denominator']} | {row['n_utterances']} / {row['n_speakers']} |")
    lines += ["[OBSERVED] Group TermER is not reported where fewer than 30 candidate spans exist; missing HiACC gender metadata for AD63 excludes its 119 utterances from the gender slice only.", ""]
    lines += [
        "", "## Decoder QC decision", "",
        f"[OBSERVED] The forced-Hindi condition was stopped after {qc['forced_hi_rows']} cached HiACC clips. In {qc['matched_rows']} matched early clips, forced Hindi had WER {_pct(forced['value'])}% ({forced['errors']}/{forced['denominator']}) and auto-language decoding had WER {_pct(auto['value'])}% ({auto['errors']}/{auto['denominator']}). These order-biased clips are an engineering QC comparison, not a benchmark result. The forced condition generated outputs with fourfold token repetition in {qc['forced_hi_outputs_with_token_repeated_4_or_more']} of its cached clips. [`language_conditioning_qc.json`](language_conditioning_qc.json)",
        f"[OBSERVED] The full auto-language output QC flags {prediction_qc['hiacc']['flagged_utterances']} HiACC and {prediction_qc['mucs_hien']['flagged_utterances']} MUCS utterances for at least one of: blank output, >3× reference length, ≥4 token repetitions, or ≥200% utterance WER. All remain in the corpus denominator. See [`prediction_qc.json`](prediction_qc.json) and [`prediction_qc_flags.csv`](prediction_qc_flags.csv).",
        "[ASSUMED] The runbook's forced `language=hi` setting was replaced with auto language detection before the full run because the public HiACC corpus includes English-only and mixed utterances. Model, revision, greedy decode, and 30-second chunk policy remained fixed.",
        "",
        "## Assumptions and reproducibility",
        "",
        "- [ASSUMED] NFC text, zero-width removal, Devanagari-to-ASCII digits, nukta removal, Latin-only lowercasing, punctuation removal, and collapsed whitespace are applied identically to references and hypotheses. Number words are unchanged.",
        "- [ASSUMED] HiACC filenames' four-character `PID` prefixes are speaker IDs. This matches `speaker_info.csv` for all but the documented AD63/AD65 mismatch.",
        "- [ASSUMED] C uses only `AUTO_ACCEPT` pairs with ITRANS-based folded distance ≤0.34, at least two observed substitutions, and no static or observed function-word collision. The threshold and guard need human audit.",
        "- [OBSERVED] ASR model `openai/whisper-large-v3` is pinned to revision `06f233fe06e710322aca913c1bc4249a0d71fce1`; per-dataset config hashes, archive hashes, package versions, audio sample-rate checks, predictions, and token tables are local. The exact launch source commit and app are in [`decode_launch.json`](decode_launch.json). Full corpus text/audio and full predictions are intentionally not committed.",
        "- [OBSERVED] Raw word alignments match jiwer on 20 fixed random utterances in tests; the C relation is realigned with zero-cost approved pairs. Insertions are reported separately because TokER is reference-anchored.",
        "",
        "## Questions for the team",
        "",
        "1. Can the team obtain approved Lahaja and VITB-HEBiC access, and does VITB actually release speaker-region metadata?",
        "2. Which HiACC group comparisons are scientifically justified after its split and metadata issues? Adult/child may still be descriptive; gender is confounded with collection context.",
        "3. Can a bilingual reviewer audit all accepted script pairs and label domain terms in a fixed 200-item sample? The present dictionary candidates are not ground truth.",
        "4. How should the speaker-heldout script-equivalence analysis be externally validated, given the lexicon is still automatically generated?",
        "5. Which target paper venue and claim fit the data without implying a model improvement or population fairness result?",
        "",
        "## Cost and time",
        "",
        "[OBSERVED] Modal receipts report function wall time and estimated resource cost; provider billing is not verified and image build, startup, storage, and data transfer may add charges.",
        "", "| Dataset | GPU | Function wall seconds | Estimated function-resource USD | Clips >30 s | Clips |", "|---|---|---:|---:|---:|---:|",
    ]
    for dataset, receipt in receipts.items():
        lines.append(f"| {dataset} | {receipt['gpu']} | {receipt['gpu_wall_seconds']:.1f} | ${receipt['estimated_function_resource_usd']:.2f} | {receipt['longer_than_30s']} | {receipt['target']} |")
    total = sum(r["estimated_function_resource_usd"] for r in receipts.values())
    lines += [
        f"| **Main decode sum** | | | **${total:.2f}** | | **{score['predictions']}** |",
        "[UNVERIFIED] Actual Modal invoice charge and the aborted forced-Hindi diagnostic charge are not included in this estimate.",
        "",
        "## Known limitations",
        "",
        "- Training-data contamination is unknown for the model on both public corpora.",
        "- The C lexicon was mined on evaluation references and must not be used as a held-out claim; candidate TERM labels are unverified and HiACC has insufficient candidate terms for TermER.",
        "- Number words are not normalized. CER under C maps only aligned equivalent token pairs to a common spelling before character alignment.",
        "- HiACC's released splits share speakers; it cannot support an unseen-speaker claim without a new speaker-disjoint split.",
        "- MUCS supplies only 30 speaker IDs and no demographics; HiACC adult/child and gender cluster counts limit group precision.",
        "- This is a one-model, one-decode-condition pilot, not a comparison of state-of-the-art open models or an intervention improving ASR.",
        "- ASR output quality on very short clips and transcript errors need human listening review before publication.",
        "",
        "## C1 code inventory",
        "",
        (OUT / "C1_inventory.md").read_text().split("## C1 Code inventory", 1)[1].strip(),
        "",
    ]
    (OUT / "RUN_REPORT.md").write_text("\n".join(lines))
    print(OUT / "RUN_REPORT.md")


if __name__ == "__main__":
    main()
