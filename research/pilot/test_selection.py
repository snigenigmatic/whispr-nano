"""Scientific control and leakage checks for the teacher-selection intervention."""

import copy
from collections import Counter

import numpy as np
import pytest

from research.pilot.selection import build, duration_matched_random, length_bin, validate_manifest


def normalizer(value):
    return " ".join(value.lower().split())


def row(uid, speaker="S1", language="Hindi", nwords=10, duration=5.0, split="train"):
    return {"utt_id": uid, "speaker_id": speaker, "native_language": language,
            "prompt_id": "prompt_" + uid, "reference_text": " ".join(f"word{i}" for i in range(nwords)),
            "audio_path": f"{speaker}/{uid}.wav", "duration_s": duration, "split": split}


def teacher_prediction(source, errors=0):
    words = source["reference_text"].split()
    for index in range(errors):
        words[index] = "wrong" + str(index)
    return {"utt_id": source["utt_id"], "reference_text": source["reference_text"],
            "hypothesis_text": " ".join(words)}


def representative_inputs():
    manifest, predictions = [], []
    for speaker, language in (("S1", "Hindi"), ("S2", "Hindi"), ("S3", "Arabic")):
        for nwords in (4, 8, 12, 22):
            for index in range(8):
                source = row(f"{speaker}_{nwords}_{index}", speaker, language, nwords=nwords)
                manifest.append(source)
                predictions.append(teacher_prediction(source, errors=0 if index < 4 else min(4, nwords)))
    return manifest, predictions


def test_exact_speaker_word_bin_and_language_counts_match():
    manifest, predictions = representative_inputs()
    arms, report, scored = build(manifest, predictions, normalizer)
    filtered = arms["filtered10_gold"]
    control = arms["random_matched_gold"]
    strata = lambda rows: Counter((r["speaker_id"], r["length_bin"]) for r in rows)
    assert strata(filtered) == strata(control)
    assert len(filtered) == len(control) == 48
    assert Counter(r["native_language"] for r in filtered) == Counter(r["native_language"] for r in control)
    assert len({r["utt_id"] for r in control}) == len(control)
    assert len(scored) == len(manifest) == 96
    assert report["target_source"] == "provided_reference_in_every_arm"


def test_control_can_select_teacher_rejected_examples():
    manifest, predictions = representative_inputs()
    arms, report, _ = build(manifest, predictions, normalizer)
    assert any(not r["accepted"] for r in arms["random_matched_gold"])
    assert report["arms"]["random_matched_gold"]["teacher_rejected_included_n"] > 0
    assert all(r["accepted"] for r in arms["filtered10_gold"])


def test_all_arms_keep_exact_raw_reference_targets():
    manifest, predictions = representative_inputs()
    manifest[0]["reference_text"] = "  WORD0 word1 word2 WORD3  "
    predictions[0] = teacher_prediction(manifest[0], errors=0)
    before = copy.deepcopy(manifest)
    arms, _, _ = build(manifest, predictions, normalizer)
    originals = {r["utt_id"]: r["reference_text"] for r in manifest}
    for arm in arms.values():
        assert all(r["reference_text"] == originals[r["utt_id"]] for r in arm)
    assert manifest == before


def test_ten_percent_threshold_is_inclusive_and_uses_error_counts():
    manifest, predictions = [], []
    for uid, words, errors in [("at", 10, 1), ("above", 9, 1), ("below", 11, 1), ("twice", 20, 2)]:
        source = row(uid, nwords=words)
        manifest.append(source)
        predictions.append(teacher_prediction(source, errors=errors))
    _, report, scored = build(manifest, predictions, normalizer, threshold=10)
    accepted = {r["utt_id"]: r["accepted"] for r in scored}
    assert accepted == {"at": True, "above": False, "below": True, "twice": True}
    assert report["threshold_pct"] == 10


def test_both_reference_and_hypothesis_are_normalized_before_thresholding():
    source = row("case")
    prediction = teacher_prediction(source)
    prediction["hypothesis_text"] = "  " + prediction["hypothesis_text"].upper().replace(" ", "  ") + " "
    _, _, scores = build([source], [prediction], normalizer)
    assert scores[0]["teacher_errors"] == 0
    assert scores[0]["accepted"]


@pytest.mark.parametrize("words,expected", [(1, 0), (4, 0), (5, 1), (9, 1), (10, 2), (19, 2), (20, 3), (50, 3)])
def test_reference_length_bins_have_registered_boundaries(words, expected):
    assert length_bin(words) == expected


def test_selection_is_reproducible_and_invariant_to_input_order():
    manifest, predictions = representative_inputs()
    arms, report, scored = build(manifest, predictions, normalizer, seed=17)
    other_arms, other_report, other_scored = build(list(reversed(manifest)), list(reversed(predictions)), normalizer, seed=17)
    assert arms == other_arms
    assert report == other_report
    assert scored == other_scored
    changed_arms, _, _ = build(manifest, predictions, normalizer, seed=18)
    assert arms["random_matched_gold"] != changed_arms["random_matched_gold"]
    assert arms["filtered10_gold"] == changed_arms["filtered10_gold"]


class DurationOnlyRow(dict):
    """Any attempt to inspect acceptance/errors while optimizing must fail."""

    def __getitem__(self, key):
        assert key == "duration_s", f"Random duration optimization read forbidden field {key}"
        return super().__getitem__(key)

    def get(self, key, default=None):
        assert key == "duration_s", f"Random duration optimization read forbidden field {key}"
        return super().get(key, default)


def test_duration_optimizer_does_not_read_teacher_labels_or_errors():
    pool = [DurationOnlyRow(duration_s=duration, teacher_errors=99 - index, accepted=index % 2 == 0)
            for index, duration in enumerate([1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5])]
    selected = duration_matched_random(pool, 4, 11.0, np.random.default_rng(23))
    assert len(selected) == 4
    assert abs(sum(r["duration_s"] for r in selected) - 11.0) <= 0.055


def test_duration_matching_preserves_count_and_reports_actual_sum():
    manifest, predictions = representative_inputs()
    for index, source in enumerate(manifest):
        source["duration_s"] = 3.0 + 0.2 * (index % 8)
    arms, report, _ = build(manifest, predictions, normalizer)
    selected = arms["random_matched_gold"]
    for balance in report["stratum_balance"]:
        actual = [r for r in selected if r["speaker_id"] == balance["speaker_id"] and r["length_bin"] == balance["length_bin"]]
        assert len(actual) == balance["random_n"] == balance["filtered_n"]
        assert sum(r["duration_s"] for r in actual) == pytest.approx(balance["random_seconds"])


def test_small_stratum_can_recover_filtered_subset_so_rejected_coverage_is_not_assumed():
    manifest = [row("short", duration=1), row("long", duration=20)]
    predictions = [teacher_prediction(manifest[0], errors=0), teacher_prediction(manifest[1], errors=2)]
    arms, report, _ = build(manifest, predictions, normalizer, seed=9)
    assert [r["utt_id"] for r in arms["filtered10_gold"]] == [r["utt_id"] for r in arms["random_matched_gold"]] == ["short"]
    assert report["arms"]["random_matched_gold"]["teacher_rejected_included_n"] == 0


def test_zero_accepted_stratum_is_allowed_when_another_stratum_has_support():
    manifest = [row("good", speaker="S1"), row("bad", speaker="S2")]
    predictions = [teacher_prediction(manifest[0]), teacher_prediction(manifest[1], errors=3)]
    arms, report, _ = build(manifest, predictions, normalizer)
    assert len(arms["filtered10_gold"]) == len(arms["random_matched_gold"]) == 1
    assert any(balance["filtered_n"] == balance["random_n"] == 0 for balance in report["stratum_balance"])


def test_zero_retained_global_pool_fails_clearly_before_training():
    manifest = [row("bad")]
    with pytest.raises(ValueError, match="(?i)(empty|retained|filter|accept)"):
        build(manifest, [teacher_prediction(manifest[0], errors=3)], normalizer)


def test_empty_normalized_reference_is_rejected():
    manifest = [row("x")]
    with pytest.raises(ValueError, match="Zero-word"):
        build(manifest, [teacher_prediction(manifest[0])], lambda _: "")


def primary_partitions():
    result = [row("train", speaker="S1"), row("dev", speaker="S2", split="dev"),
              row("test", speaker="S3", split="test")]
    for source in result:
        source["reference_text"] = source["split"] + " content exclusively"
    return result


@pytest.mark.parametrize("field", ["speaker_id", "prompt_id", "reference_text"])
def test_train_dev_speaker_content_leakage_fails(field):
    manifest = primary_partitions()
    manifest[1][field] = manifest[0][field]
    with pytest.raises(ValueError, match="leakage"):
        validate_manifest(manifest)


def test_reference_normalization_cannot_hide_content_leakage():
    manifest = primary_partitions()
    manifest[1]["reference_text"] = manifest[0]["reference_text"].upper() + "!"
    with pytest.raises(ValueError, match="reference leakage"):
        validate_manifest(manifest)


def test_conflicting_native_language_for_a_speaker_is_rejected():
    manifest = [row("a", language="Hindi"), row("b", language="Arabic")]
    with pytest.raises(ValueError, match="(?i)(language|speaker)"):
        validate_manifest(manifest)


@pytest.mark.parametrize("change", ["missing_train", "dev_prediction", "duplicate", "changed_reference"])
def test_teacher_predictions_must_match_training_provenance_exactly(change):
    manifest = primary_partitions()
    predictions = [teacher_prediction(manifest[0])]
    if change == "missing_train":
        predictions = []
    elif change == "dev_prediction":
        predictions.append(teacher_prediction(manifest[1]))
    elif change == "duplicate":
        predictions.append(copy.deepcopy(predictions[0]))
    else:
        predictions[0]["reference_text"] += " changed"
    with pytest.raises(ValueError):
        build(manifest, predictions, normalizer)
def test_small_stratum_local_minimum_uses_feasible_duration_match():
    import numpy as np
    from research.pilot.selection import duration_matched_random
    pool = [{'duration_s': d, 'utt_id': str(i)} for i, d in enumerate([10., 10., .1, 20.9])]
    chosen = duration_matched_random(pool, 2, 21., np.random.default_rng(1))
    assert len(chosen) == 2
    assert abs(sum(r['duration_s'] for r in chosen) - 21.) <= .105
