"""Tests for the Days 1–3 research reset. CPU only; no GPU, no HF downloads
except the optional Whisper normalizer already present in this environment.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from research_reset.constants import (
    AUDIT_CSV,
    L1_GROUPS,
    N_LISTENING,
    PAIRED_SEEDS,
    PER_CELL,
)
from research_reset.io_utils import read_scored_csv
from research_reset.listening import sample_listening_set
from research_reset.partitions import split_gate_rows
from research_reset.pools import construct_filtered_pool, construct_matched_random_pool
from research_reset.runs import build_run_specs
from research_reset.scoring import (
    corpus_wer,
    macro_wer,
    normalize_capstone,
    word_edit_counts,
)
from research_reset.stop_rule import apply_stop_rule
from research_reset.train_sft import pair_fingerprint, target_text


@pytest.fixture(scope="module")
def scored_rows():
    return read_scored_csv(AUDIT_CSV)


def test_capstone_normalizer_matches_audit_csv(scored_rows):
    for row in scored_rows:
        assert normalize_capstone(row["reference_text"]) == row["reference_normalized"]
        assert normalize_capstone(row["hypothesis_text"]) == row["hypothesis_normalized"]


def test_word_edit_counts_simple():
    counts = word_edit_counts("the cat sat", "the dog sat", normalizer="capstone")
    assert counts.substitutions == 1
    assert counts.deletions == 0
    assert counts.insertions == 0
    assert counts.wer == pytest.approx(1 / 3)


def test_listening_set_is_120_balanced_and_deterministic(scored_rows):
    a = sample_listening_set(scored_rows)
    b = sample_listening_set(scored_rows)
    assert len(a) == N_LISTENING
    assert [x["utt_id"] for x in a] == [x["utt_id"] for x in b]
    by_cell: dict[str, int] = {}
    for item in a:
        by_cell[item["cell"]] = by_cell.get(item["cell"], 0) + 1
    assert len(by_cell) == 12
    assert set(by_cell.values()) == {PER_CELL}
    assert {item["primary_language"] for item in a} == set(L1_GROUPS)
    accepted = [x for x in a if x["accepted"]]
    rejected = [x for x in a if not x["accepted"]]
    assert len(accepted) == 60 and len(rejected) == 60
    assert all(x["capstone_wer"] == 0.0 for x in accepted)
    assert all(x["capstone_wer"] > 0.0 for x in rejected)


def test_gate_split_is_disjoint(scored_rows):
    listening = sample_listening_set(scored_rows)
    parts = split_gate_rows(scored_rows, {i["utt_id"] for i in listening})
    ids = {name: {r["utt_id"] for r in rows} for name, rows in parts.items()}
    assert ids["listening"].isdisjoint(ids["train"])
    assert ids["listening"].isdisjoint(ids["eval"])
    assert ids["train"].isdisjoint(ids["eval"])
    assert len(ids["listening"]) + len(ids["train"]) + len(ids["eval"]) == len(scored_rows)


def test_twelve_runs_are_paired(scored_rows):
    listening = sample_listening_set(scored_rows)
    parts = split_gate_rows(scored_rows, {i["utt_id"] for i in listening})
    specs = build_run_specs(parts["train"])
    assert len(specs) == 12
    by_pair: dict[str, list[dict]] = {}
    for spec in specs:
        by_pair.setdefault(spec["pair_id"], []).append(spec)
    assert len(by_pair) == 6
    for pair_id, arms in by_pair.items():
        assert {a["label_source"] for a in arms} == {"reference", "teacher"}
        assert pair_fingerprint(arms[0]) == pair_fingerprint(arms[1])
        assert arms[0]["recording_ids"] == arms[1]["recording_ids"]
        assert len(set(arms[0]["recording_ids"])) == len(arms[0]["recording_ids"])
        # Paired arms must not train on listening or eval ids.
        banned = {r["utt_id"] for r in parts["listening"] + parts["eval"]}
        assert banned.isdisjoint(arms[0]["recording_ids"])
    assert {s["seed"] for s in specs} == set(PAIRED_SEEDS)


def test_matched_random_same_size_as_filtered(scored_rows):
    listening = sample_listening_set(scored_rows)
    parts = split_gate_rows(scored_rows, {i["utt_id"] for i in listening})
    filtered = construct_filtered_pool(parts["train"])
    matched = construct_matched_random_pool(parts["train"], filtered, seed=12)
    assert len(matched) == len(filtered)
    assert len({r["utt_id"] for r in matched}) == len(matched)
    # Language counts match.
    from collections import Counter

    assert Counter(r["primary_language"] for r in filtered) == Counter(
        r["primary_language"] for r in matched
    )


def test_target_text_switches_label_source():
    row = {"reference_text": "hello", "hypothesis_text": "hallo"}
    assert target_text(row, "reference") == "hello"
    assert target_text(row, "teacher") == "hallo"


def test_stop_rule_closes_when_teacher_is_not_worse():
    def macros(ref: float, teacher: float) -> dict:
        out = {}
        for seed in PAIRED_SEEDS:
            out[f"matched_random__reference__seed{seed}"] = {
                "macro_wer@2400:capstone": ref,
                "macro_wer@2400:whisper": ref,
            }
            out[f"matched_random__teacher__seed{seed}"] = {
                "macro_wer@2400:capstone": teacher,
                "macro_wer@2400:whisper": teacher,
            }
        return out

    closed = apply_stop_rule(macros(0.20, 0.201))
    assert closed["continue_days_4_10"] is False
    opened = apply_stop_rule(macros(0.20, 0.21))  # +1.0 pp
    assert opened["continue_days_4_10"] is True
    assert opened["mean_delta_pp_primary"] == pytest.approx(1.0)


def test_stop_rule_requires_two_seeds_and_no_reversal():
    results = {}
    # Two seeds worse by 1 pp, one seed better by 0.4 pp → mean 0.533, 2 worse.
    deltas = {12: 0.01, 34: 0.01, 56: -0.004}
    alt = {12: 0.008, 34: 0.008, 56: -0.02}  # alternate mean negative → reversal
    for seed in PAIRED_SEEDS:
        results[f"matched_random__reference__seed{seed}"] = {
            "macro_wer@2400:capstone": 0.20,
            "macro_wer@2400:whisper": 0.20,
        }
        results[f"matched_random__teacher__seed{seed}"] = {
            "macro_wer@2400:capstone": 0.20 + deltas[seed],
            "macro_wer@2400:whisper": 0.20 + alt[seed],
        }
    decision = apply_stop_rule(results)
    assert decision["n_seeds_with_deterioration"] == 2
    assert decision["mean_delta_pp_primary"] >= 0.5
    assert decision["continue_days_4_10"] is False  # reversal under whisper


def test_macro_wer_is_unweighted(tmp_path: Path):
    rows = [
        {"primary_language": "Hindi", "reference_text": "a b", "hypothesis_text": "a b"},
        {"primary_language": "Hindi", "reference_text": "a b", "hypothesis_text": "a b"},
        {"primary_language": "Bodo", "reference_text": "a b", "hypothesis_text": "x y"},
    ]
    # Hindi 0, Bodo 1 → macro 0.5, not the micro 0.33.
    result = macro_wer(rows, ("Hindi", "Bodo"))
    assert result["macro_wer"] == pytest.approx(0.5)
    assert corpus_wer([(r["reference_text"], r["hypothesis_text"]) for r in rows]) == pytest.approx(0.5 / 1.5)


def test_macro_wer_errors_on_missing_group():
    rows = [{"primary_language": "Hindi", "reference_text": "a", "hypothesis_text": "a"}]
    with pytest.raises(ValueError, match="missing groups"):
        macro_wer(rows, ("Hindi", "Bodo"))


def test_committed_listening_set_matches_sampler(scored_rows):
    committed = Path("data/research_reset/listening_set.json")
    if not committed.exists():
        pytest.skip("artifacts not generated yet")
    sampled = [i["utt_id"] for i in sample_listening_set(scored_rows)]
    stored = [i["utt_id"] for i in json.loads(committed.read_text())]
    assert sampled == stored
