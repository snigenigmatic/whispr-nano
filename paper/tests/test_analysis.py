"""Behavioral checks for pairing, normalization, and audit arithmetic."""

import json
import unittest
from pathlib import Path

import numpy as np

from paper.analysis.analyze import (
    analyze_panel,
    corpus_wer,
    holm_adjust,
    interval,
    make_bootstrap_plan,
    normalize_minimal,
    poisson_diagnostic,
    validate_frozen_inputs,
    validate_full_panel,
    validate_unchanged_saaras,
    word_edit_counts,
)


class TestScoring(unittest.TestCase):
    def test_full_census_requires_every_pinned_row_and_reference(self):
        row = {"reference_text": "hello there", "primary_language": "Hindi", "family": "Indo-Aryan",
               "gender": "Unknown", "duration_s": "1"}
        metadata = {f"svarah_{i:05d}": dict(row) for i in range(6656)}
        systems = {name: dict(metadata) for name in ("a", "b", "c", "d", "e")}
        validate_full_panel(systems, set(systems), metadata)
        with self.assertRaises(ValueError):
            validate_full_panel({key: value for key, value in systems.items() if key != "a"}, set(systems), metadata)
        systems["a"] = dict(systems["a"])
        systems["a"].pop("svarah_00001")
        with self.assertRaises(ValueError):
            validate_full_panel(systems, set(systems), metadata)
        systems["a"]["svarah_00001"] = {**row, "reference_text": "wrong benchmark reference"}
        with self.assertRaises(ValueError):
            validate_full_panel(systems, set(systems), metadata)

    def test_frozen_study_cannot_silently_lose_systems_or_change_cohorts(self):
        open_ids = [f"u{i}" for i in range(2500)]
        saaras_ids = [f"u{i}" for i in range(450)]
        models = ["a", "b", "c", "d", "e", "saarasv3"]
        systems, provenance, files = {}, [], {}
        for model in models:
            ids = saaras_ids if model == "saarasv3" else open_ids
            systems[model] = {key: {} for key in ids}
            filename = f"results_{model}_svarah_clean.csv"
            files[filename] = {"sha256": model, "n_rows": len(ids)}
            provenance.append({"path": filename, "sha256": model, "rows": len(ids)})
        manifest = {"source_files": files, "open_cohort_ids": open_ids, "saaras_cohort_ids": saaras_ids}
        validate_frozen_inputs(systems, provenance, manifest)
        with self.assertRaises(ValueError):
            validate_frozen_inputs(systems, provenance[:-1], manifest)
        systems["a"].pop("u1")
        with self.assertRaises(ValueError):
            validate_frozen_inputs(systems, provenance, manifest)

    def test_saaras_predictions_cannot_change_in_matched_panel(self):
        original = {"saarasv3": {"u1": {"hypothesis_text": "frozen"}}}
        validate_unchanged_saaras(original, original)
        with self.assertRaises(ValueError):
            validate_unchanged_saaras(original, {"saarasv3": {"u1": {"hypothesis_text": "changed"}}})

    def test_edit_arithmetic_and_empty_reference(self):
        counts = word_edit_counts("one two three", "one four three five")
        self.assertEqual((counts.substitutions, counts.deletions, counts.insertions), (1, 0, 1))
        self.assertEqual(counts.reference_words, 3)
        self.assertEqual(word_edit_counts("", "extra words").errors, 2)
        self.assertEqual(word_edit_counts("one two", "").deletions, 2)

    def test_corpus_wer_is_word_weighted_and_preserves_insertions(self):
        errors, lengths = np.array([1, 0, 2]), np.array([1, 9, 0])
        self.assertEqual(corpus_wer(errors, lengths), 30.0)
        self.assertIsNone(corpus_wer(np.array([2]), np.array([0])))

    def test_minimal_normalization_preserves_documented_features(self):
        text = "  WOMEN'S  15-17, PHONE_NUMBER! "
        self.assertEqual(normalize_minimal(text), "women's 15 17 phone_number")
        self.assertEqual(normalize_minimal("café"), "café")

    def test_common_normalization_applied_to_both_sides(self):
        from transformers.models.whisper.english_normalizer import EnglishTextNormalizer
        path = Path(__file__).parents[1] / "data" / "whisper_normalizer.json"
        normalizer = EnglishTextNormalizer(json.loads(path.read_text()))
        pairs = [("I have twenty one apples.", "I have 21 apples"),
                 ("That is my colour.", "That is my color."),
                 ("I can not go", "I can't go")]
        for reference, hypothesis in pairs:
            self.assertEqual(word_edit_counts(normalizer(reference), normalizer(hypothesis)).errors, 0)
        self.assertGreater(word_edit_counts(normalize_minimal(pairs[0][0]),
                                           normalize_minimal(pairs[0][1])).errors, 0)


class TestBootstrap(unittest.TestCase):
    def test_pairing_and_whole_cluster_weights(self):
        plan = make_bootstrap_plan(["a", "a", "b", "c", "d", "d"],
                                   ["L1", "L1", "L1", "L2", "L2", "L2"], 2000, 8)
        errors = np.array([1, 2, 0, 3, 1, 0])
        references = np.array([5, 4, 6, 9, 4, 6])
        np.testing.assert_array_equal(plan.aggregate(errors), [3, 0, 3, 1])
        np.testing.assert_array_equal(plan.weights[:, :2].sum(axis=1), 2)
        np.testing.assert_array_equal(plan.weights[:, 2:].sum(axis=1), 2)
        # An identical second system must have a zero paired difference in EVERY
        # resample, not merely similar marginal confidence intervals.
        a = plan.ratios(errors, references)
        b = plan.ratios(errors.copy(), references.copy())
        np.testing.assert_array_equal(a - b, np.zeros(2000))
        self.assertGreater(np.std(a), 0)
        repeated = make_bootstrap_plan(["a", "a", "b", "c", "d", "d"],
                                      ["L1", "L1", "L1", "L2", "L2", "L2"], 2000, 8)
        np.testing.assert_array_equal(plan.weights, repeated.weights)

    def test_invalid_or_missing_cluster_ids_are_rejected(self):
        with self.assertRaises(ValueError):
            make_bootstrap_plan(["a", "a"], ["Hindi", "Bodo"], 2000, 1)
        with self.assertRaises(ValueError):
            make_bootstrap_plan(["a", ""], ["Hindi", "Hindi"], 2000, 1)
        low, high, status = interval(None, None, None)
        self.assertIsNone(low)
        self.assertIsNone(high)
        self.assertIn("unsupported", status)
        self.assertIsNone(interval(np.ones(2000), 1, "speaker")[0])

    def test_holm_stepdown_includes_withheld_contrasts(self):
        got = holm_adjust([0.04, 0.01, 0.03, None])
        np.testing.assert_allclose(got[:3], [0.09, 0.04, 0.09])
        self.assertIsNone(got[3])

    def test_unverified_groupings_never_produce_speaker_pvalues(self):
        systems = {"whisper-large-v3": {}, "other": {}}
        metadata = {}
        for i in range(36):
            language, family = [("Bodo", "Sino-Tibetan"), ("Kannada", "Dravidian"),
                                ("Hindi", "Indo-Aryan")][i // 12]
            key = f"u{i}"
            row = {"utt_id": key, "reference_text": "one two three",
                   "hypothesis_text": "one four three" if i < 12 else "one two three",
                   "primary_language": language, "family": family,
                   "gender": "Unknown", "duration_s": "2"}
            systems["whisper-large-v3"][key] = row
            systems["other"][key] = dict(row)
            metadata[key] = {"recording": f"rec{i}"}
        for kind, status in (("recording", "diagnostic_filename_block_only"),
                             ("demographic_profile", "exploratory_unverified_demographic_profile")):
            result = analyze_panel("test", systems, sorted(metadata), {"minimal": normalize_minimal},
                                   metadata, "recording", kind, 200, 8, 10)
            outputs = result["normalizations"]["minimal"]
            for row in outputs["subgroup_contrasts"] + outputs["model_contrasts"]:
                self.assertIsNone(row["approximate_bootstrap_p"])
                self.assertIsNone(row["holm_adjusted_p"])
                self.assertEqual(row["ci_status"], status)
            self.assertFalse(result["cluster_status"]["inference_supported"])
            contrast = outputs["model_contrasts"][0]
            self.assertEqual((contrast["difference_pp"], contrast["ci_low_pp"], contrast["ci_high_pp"]), (0, 0, 0))

    def test_missing_metadata_does_not_drop_observations(self):
        rows = {}
        for i, (language, family) in enumerate((("Bodo", "Sino-Tibetan"),
                                               ("Tamil", "Dravidian"), ("Hindi", "Indo-Aryan"))):
            rows[str(i)] = {"utt_id": str(i), "reference_text": "a b c", "hypothesis_text": "a b c",
                            "primary_language": language, "family": family, "duration_s": "1"}
        result = analyze_panel("test", {"whisper-large-v3": rows}, sorted(rows),
                               {"minimal": normalize_minimal}, {"0": {"speaker_id": "real-id"}},
                               "speaker_id", "speaker", 200, 8, 10)
        self.assertEqual(result["n_utterances"], 3)
        self.assertFalse(result["cluster_status"]["complete"])
        self.assertEqual(result["normalizations"]["minimal"]["metrics"][0]["wer_pct"], 0)
        self.assertIsNone(result["normalizations"]["minimal"]["metrics"][0]["ci_low_pct"])


class TestDiagnostic(unittest.TestCase):
    def test_poisson_equal_group_rates_have_zero_deviance_difference(self):
        result = poisson_diagnostic(np.array([1, 3, 2, 2, 0, 4]), np.full(6, 10),
                                    np.array(["A", "A", "B", "B", "C", "C"]))
        self.assertAlmostEqual(result["deviance_difference"], 0)
        self.assertAlmostEqual(result["unadjusted_poisson_p"], 1)
        self.assertAlmostEqual(result["pearson_dispersion"], 5 / 3)


if __name__ == "__main__":
    unittest.main()
