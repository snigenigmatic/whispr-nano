"""Post hoc checks prompted by the initial teacher-coverage diagnostic.

Length standardization and shared-text comparisons are descriptive controls,
not identification of an accent effect. Also reconstruct the old PoC split
from its saved configuration and current pinned row order; no training occurs.
"""

import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from audit.opd_bias_extension.family_mapping import FAMILIES, get_family
from paper.analysis.analyze import hash_file, normalize_minimal


ROOT = Path(__file__).resolve().parents[2]


def length_bin(n):
    return "1-4" if n < 5 else "5-9" if n < 10 else "10-19" if n < 20 else "20+"


def main():
    metadata_path = ROOT / "paper/data/svarah_metadata.csv"
    meta = list(csv.DictReader(metadata_path.open(newline="")))
    by_id = {r["utt_id"]: r for r in meta}
    scores = list(csv.DictReader((ROOT / "research/teacher_coverage/results/per_utterance.csv").open()))
    controls = []
    for norm in ("minimal", "english"):
        rows = [r for r in scores if r["normalizer"] == norm]
        for r in rows:
            r["length_bin"] = length_bin(int(r["words"]))
            r["reject"] = 100 * int(r["teacher_errors"]) > 10 * int(r["words"])
        bins = []
        bodo_n = sum(r["native_language"] == "Bodo" for r in rows)
        non_standardized = 0
        for b in ("1-4", "5-9", "10-19", "20+"):
            a = [r for r in rows if r["length_bin"] == b and r["native_language"] == "Bodo"]
            c = [r for r in rows if r["length_bin"] == b and r["native_language"] != "Bodo"]
            assert a and c, "Standardization requires common support"
            ar, cr = np.mean([r["reject"] for r in a]), np.mean([r["reject"] for r in c])
            non_standardized += cr * len(a) / bodo_n
            bins.append({"length_bin": b, "Bodo_n": len(a), "non_Bodo_n": len(c),
                         "Bodo_rejected_pct": 100 * float(ar), "non_Bodo_rejected_pct": 100 * float(cr)})
        actual = sum(r["reject"] for r in rows if r["native_language"] == "Bodo") / bodo_n
        # Match only exact Minimal-normalized transcripts; do not silently
        # convert this to semantic matching or an accent intervention.
        prompts = defaultdict(list)
        for r in rows:
            prompts[normalize_minimal(by_id[r["utt_id"]]["reference_text"])].append(r)
        common = [group for group in prompts.values()
                  if any(r["native_language"] == "Bodo" for r in group)
                  and any(r["native_language"] != "Bodo" for r in group)]
        shared = {"n_shared_minimal_normalized_reference_strings": len(common),
                  "Bodo_n": sum(r["native_language"] == "Bodo" for g in common for r in g),
                  "non_Bodo_n": sum(r["native_language"] != "Bodo" for g in common for r in g)}
        for key, is_bodo in (("Bodo", True), ("non_Bodo", False)):
            prompt_rates = [np.mean([r["reject"] for r in g
                                    if (r["native_language"] == "Bodo") == is_bodo]) for g in common]
            shared[f"equal_reference_string_weight_rejected_pct_{key}"] = (
                100 * float(np.mean(prompt_rates)) if prompt_rates else None)
        controls.append({"normalizer": norm, "threshold_wer_pct": 10,
                         "Bodo_observed_rejected_pct": 100 * actual,
                         "non_Bodo_reweighted_to_Bodo_length_bins_rejected_pct": 100 * non_standardized,
                         "length_standardized_difference_pp": 100 * (actual - non_standardized),
                         "length_bins": bins, "shared_text_sensitivity": shared})
    saved_path = ROOT / "audit/opd_bias_extension/svarah_opd_run_20260828-172221.json"
    saved = json.loads(saved_path.read_text())
    cfg = saved["arms"]["svarah_poc_on_policy"]["config"]
    rng = np.random.default_rng(cfg["seed"])
    train, test = [], []
    for family in FAMILIES:
        indices = [i for i, r in enumerate(meta) if get_family(r["primary_language"]) == family]
        ntrain, ntest = saved["train_per_family"], saved["eval_per_family"]
        selected = rng.choice(indices, size=ntrain + ntest, replace=False)
        train += [meta[i] for i in selected[:ntrain]]
        test += [meta[i] for i in selected[ntrain:]]
    sample_ids = {x["clip_id"] for x in saved["arms"]["svarah_poc_on_policy"]["baseline_eval"]["samples"]}
    assert sample_ids <= {r["utt_id"] for r in test}, "Reconstructed order disagrees with saved examples"
    overlap = {}
    for column in ("utt_id", "recording_cluster_id", "demographic_profile_id"):
        seen = {r[column] for r in train}
        overlap[column] = {"distinct_shared_values": len(seen & {r[column] for r in test}),
                           "eval_utterances_with_value_in_train": sum(r[column] in seen for r in test)}
    train_texts = {normalize_minimal(r["reference_text"]) for r in train}
    overlap["minimal_normalized_reference"] = {
        "eval_utterances_with_value_in_train": sum(
            normalize_minimal(r["reference_text"]) in train_texts for r in test)}
    result = {
        "status": "post_hoc_descriptive_controls_no_causal_claim",
        "metadata_sha256": hash_file(metadata_path),
        "controls": controls,
        "old_poc_split_reconstruction": {
            "saved_run_sha256": hash_file(saved_path), "train_n": len(train), "eval_n": len(test),
            "matches_all_saved_example_ids": True, "overlap": overlap,
            "caveat": "Reconstructed from seed and pinned row order, not saved complete split IDs. Profiles are not verified speakers. The old loader samples utterances from Svarah test; it never enforces speaker independence.",
            "train_ids": [r["utt_id"] for r in train], "eval_ids": [r["utt_id"] for r in test]},
    }
    out = ROOT / "research/teacher_coverage/results/confounders.json"
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"controls": controls, "old_poc_split_overlap": overlap}, indent=2))


if __name__ == "__main__":
    main()
