"""Tests for audit/assemble.py -- schema hard-fail + math regression, per
WEEKEND_PLAN.md's AT1 brief ("Include a --strict pytest with a fixture CSV").

CPU-only: pandas + statsmodels + scipy, no torch, no ast_asr package import.
Run with: pytest audit/test_assemble.py -v
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from assemble import REQUIRED_COLUMNS, SchemaError, assemble

FIXTURE_ROWS = [
    # (utt_id, primary_language, family, gender, duration_s, ref, ref_norm, hyp, hyp_norm, wer, n_sub, n_del, n_ins, ref_len_words)
    ("u1", "Hindi", "Indo-Aryan", "Female", "3.0", "a b c d", "a b c d", "a b c d", "a b c d", "0.0", "0", "0", "0", "4"),
    ("u2", "Hindi", "Indo-Aryan", "Male", "2.5", "e f g h", "e f g h", "e f x h", "e f x h", "0.25", "1", "0", "0", "4"),
    ("u3", "Tamil", "Dravidian", "Female", "3.1", "i j k l", "i j k l", "i j k l", "i j k l", "0.0", "0", "0", "0", "4"),
    ("u4", "Tamil", "Dravidian", "Male", "2.9", "m n o p", "m n o p", "m n o p", "m n o p", "0.0", "0", "0", "0", "4"),
    ("u5", "Bodo", "Sino-Tibetan", "Female", "3.4", "q r s t", "q r s t", "q x s x", "q x s x", "0.5", "2", "0", "0", "4"),
    ("u6", "Bodo", "Sino-Tibetan", "Male", "2.7", "u v w x", "u v w x", "u v w x", "u v w x", "0.0", "0", "0", "0", "4"),
    ("u7", "Bodo", "Sino-Tibetan", "Female", "3.0", "y z a b", "y z a b", "y z x b", "y z x b", "0.25", "1", "0", "0", "4"),
    ("u8", "Hindi", "Indo-Aryan", "Male", "2.8", "c d e f", "c d e f", "c d e f", "c d e f", "0.0", "0", "0", "0", "4"),
    ("u9", "Tamil", "Dravidian", "Female", "3.2", "g h i j", "g h i j", "g h i j", "g h i j", "0.0", "0", "0", "0", "4"),
    ("u10", "Hindi", "Indo-Aryan", "Male", "2.6", "k l m n", "k l m n", "k l m n", "k l m n", "0.0", "0", "0", "0", "4"),
]


def _write_fixture_csv(path: Path, rows=FIXTURE_ROWS, columns=REQUIRED_COLUMNS) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(columns)
        for row in rows:
            writer.writerow(row)


def test_assemble_computes_expected_overall_wer(tmp_path: Path):
    csv_path = tmp_path / "results_fixture-system_svarah_clean.csv"
    _write_fixture_csv(csv_path)

    results = assemble(tmp_path, strict=True)
    assert len(results) == 1
    r = results[0]
    assert r.system == "fixture-system"
    assert r.n_utterances == 10

    # 4 substitution errors total (u2:1, u5:2, u7:1) over 40 reference words.
    assert r.overall_wer == pytest.approx(4 / 40)
    assert r.family_n == {"Indo-Aryan": 4, "Dravidian": 3, "Sino-Tibetan": 3}
    # Indo-Aryan: 1 err / 16 words; Dravidian: 0 err / 12 words; Sino-Tibetan: 3 err / 12 words.
    assert r.per_family_wer["Indo-Aryan"] == pytest.approx(1 / 16)
    assert r.per_family_wer["Dravidian"] == pytest.approx(0.0)
    assert r.per_family_wer["Sino-Tibetan"] == pytest.approx(3 / 12)
    assert r.delta_dp_pp == pytest.approx((3 / 12 - 0.0) * 100)
    assert r.poisson_p is not None


def test_assemble_writes_master_table(tmp_path: Path):
    csv_path = tmp_path / "results_fixture-system_svarah_clean.csv"
    _write_fixture_csv(csv_path)

    from assemble import _write_outputs

    results = assemble(tmp_path, strict=True)
    _write_outputs(results, tmp_path)

    assert (tmp_path / "master_audit_table.csv").exists()
    md = (tmp_path / "master_audit_table.md").read_text()
    assert "fixture-system" in md
    assert "Sino-Tibetan" in md


def test_missing_column_hard_fails(tmp_path: Path):
    """Schema drift (a dropped column) must raise, not silently proceed."""
    csv_path = tmp_path / "results_broken-system_svarah_clean.csv"
    columns = [c for c in REQUIRED_COLUMNS if c != "n_ins"]  # drop a required column
    _write_fixture_csv(csv_path, columns=columns, rows=[row[:-2] + row[-1:] for row in FIXTURE_ROWS])

    with pytest.raises(SchemaError, match="missing columns"):
        assemble(tmp_path, strict=True)


def test_extra_column_hard_fails(tmp_path: Path):
    """Schema drift (an added column) must also raise, not silently pass through."""
    csv_path = tmp_path / "results_broken-system_svarah_clean.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(REQUIRED_COLUMNS + ["unexpected_extra_column"])
        for row in FIXTURE_ROWS:
            writer.writerow(row + ("surprise",))

    with pytest.raises(SchemaError, match="unexpected columns"):
        assemble(tmp_path, strict=True)


def test_unknown_family_hard_fails(tmp_path: Path):
    """A family label outside the three-way Svarah taxonomy must raise, not
    get silently grouped or dropped (guards against taxonomy drift)."""
    csv_path = tmp_path / "results_broken-system_svarah_clean.csv"
    bad_rows = list(FIXTURE_ROWS)
    bad_rows[0] = ("u1", "Klingon", "Klingon-family", "Female", "3.0", "a b c d", "a b c d", "a b c d", "a b c d", "0.0", "0", "0", "0", "4")
    _write_fixture_csv(csv_path, rows=bad_rows)

    with pytest.raises(SchemaError, match="unknown family"):
        assemble(tmp_path, strict=True)


def test_strict_hard_fails_on_no_files(tmp_path: Path):
    with pytest.raises(SchemaError, match="no files matching"):
        assemble(tmp_path, strict=True)


def test_non_strict_warns_on_no_files(tmp_path: Path):
    assert assemble(tmp_path, strict=False) == []


def test_summary_mismatch_hard_fails(tmp_path: Path):
    """If a stale summary_<tag>_svarah.json disagrees with the per-utterance
    CSV it's supposed to summarize, that must be caught, not trusted."""
    csv_path = tmp_path / "results_fixture-system_svarah_clean.csv"
    _write_fixture_csv(csv_path)
    summary_path = tmp_path / "summary_fixture-system_svarah.json"
    summary_path.write_text(json.dumps({"overall_wer": 0.999, "n_utterances": 10}))

    with pytest.raises(SchemaError, match="disagree with"):
        assemble(tmp_path, strict=True)


def test_summary_agreement_passes(tmp_path: Path):
    """A summary that DOES match the recomputed numbers should not raise."""
    csv_path = tmp_path / "results_fixture-system_svarah_clean.csv"
    _write_fixture_csv(csv_path)
    summary_path = tmp_path / "summary_fixture-system_svarah.json"
    summary_path.write_text(json.dumps({
        "overall_wer": 4 / 40,
        "n_utterances": 10,
        "per_family_wer": {"Indo-Aryan": 1 / 16, "Dravidian": 0.0, "Sino-Tibetan": 3 / 12},
    }))

    results = assemble(tmp_path, strict=True)
    assert len(results) == 1
