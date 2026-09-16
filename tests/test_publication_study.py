from pathlib import Path

import numpy as np
import pytest

from publication_study.arithmetic import apply_global_correction
from publication_study.contract import BudgetLedger, StudyContract, atomic_json
from publication_study.integrity import balanced_target_rows, validate_speaker_boundary


ROOT = Path(__file__).resolve().parents[1]


def test_committed_contract_is_frozen_and_valid():
    contract = StudyContract.load(ROOT / "research/publication/contract.json")
    assert contract.seed == 12
    assert contract.hard_cap_usd == 12.0
    assert len(contract.sha256) == 64


def test_budget_refuses_duplicate_and_over_cap(tmp_path):
    path = tmp_path / "budget.json"
    atomic_json(path, {"schema_version": 1, "hard_cap_usd": 12.0, "jobs": []})
    ledger = BudgetLedger(path)
    ledger.reserve("smoke", 0.5, "a" * 64)
    with pytest.raises(ValueError, match="cannot be reused"):
        ledger.reserve("smoke", 0.5, "a" * 64)
    with pytest.raises(RuntimeError, match="exceed"):
        ledger.reserve("too-much", 11.51, "b" * 64)


def test_balanced_target_manifest_rejects_labels_and_is_deterministic():
    rows = [
        {"utt_id": f"{g}{i}", "speaker_id": f"{g}s{i//2}", "accent": g, "audio": f"{g}{i}.wav"}
        for g in ("Hindi", "Tamil") for i in range(8)
    ]
    a, report = balanced_target_rows(rows, total=8)
    b, _ = balanced_target_rows(list(reversed(rows)), total=8)
    assert a == b
    assert report["selected_by_accent"] == {"Hindi": 4, "Tamil": 4}
    with pytest.raises(ValueError, match="reference-like"):
        balanced_target_rows([{**rows[0], "text": "leak"}, *rows[1:]], total=8)


def test_spire_speaker_boundary_fails_closed():
    train = [{"utt_id": "tr1", "speaker_id": "s1"}]
    test = [{"utt_id": "te1", "speaker_id": "s2"}]
    assert validate_speaker_boundary(train, test)["speaker_overlap"] == 0
    with pytest.raises(ValueError, match="speaker overlap"):
        validate_speaker_boundary(train, [{"utt_id": "te1", "speaker_id": "s1"}])


def test_global_pseudo2real_arithmetic_and_shape_guard():
    real = {"w": np.array([3.0, 5.0], dtype=np.float32)}
    pseudo = {"w": np.array([2.0, 1.0], dtype=np.float32)}
    target = {"w": np.array([10.0, 20.0], dtype=np.float32)}
    out, report = apply_global_correction(real, pseudo, target, 0.5)
    assert np.allclose(out["w"], [10.5, 22.0])
    assert report["formula"].startswith("target_pseudo")
    with pytest.raises(ValueError, match="signature"):
        apply_global_correction(real, {"w": np.ones((2, 1), dtype=np.float32)}, target, 1.0)
