from pathlib import Path

import numpy as np
import pytest

from publication_study.arithmetic import apply_global_correction, apply_weighted_correction
from publication_study.contract import BudgetLedger, StudyContract, atomic_json
from publication_study.integrity import (
    TARGET_ACCENTS,
    balanced_target_rows,
    validate_speaker_boundary,
    validate_target_teacher_artifacts,
)


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


def test_budget_releases_unused_terminal_reservation_headroom(tmp_path):
    path = tmp_path / "budget.json"
    atomic_json(path, {"schema_version": 1, "hard_cap_usd": 12.0, "jobs": []})
    ledger = BudgetLedger(path)
    ledger.reserve("first", 5.0, "a" * 64)
    ledger.complete("first", 1.0, "b" * 64)
    assert ledger.reserved_total() == 5.0
    assert ledger.guarded_total() == 1.0
    ledger.reserve("second", 10.9, "c" * 64)


def test_budget_records_measured_cancellation(tmp_path):
    path = tmp_path / "budget.json"
    atomic_json(path, {"schema_version": 1, "hard_cap_usd": 12.0, "jobs": []})
    ledger = BudgetLedger(path)
    ledger.reserve("stopped", 2.0, "a" * 64)
    ledger.cancel("stopped", 0.25, "bounded worker was stopped")
    assert ledger.guarded_total() == 0.25
    with pytest.raises(ValueError, match="not uniquely reserved"):
        ledger.cancel("stopped", 0.25, "again")


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


def test_weighted_correction_bank_uses_frozen_mixture_weights():
    real = [
        {"w": np.array([3.0, 5.0], dtype=np.float32)},
        {"w": np.array([8.0, 2.0], dtype=np.float32)},
    ]
    pseudo = [
        {"w": np.array([2.0, 1.0], dtype=np.float32)},
        {"w": np.array([4.0, 4.0], dtype=np.float32)},
    ]
    target = {"w": np.array([10.0, 20.0], dtype=np.float32)}
    out, report = apply_weighted_correction(real, pseudo, target, [0.25, 0.75], 0.5)
    assert np.allclose(out["w"], [11.625, 19.75])
    assert report["weights"] == [0.25, 0.75]
    with pytest.raises(ValueError, match="sum to one"):
        apply_weighted_correction(real, pseudo, target, [0.5, 0.6], 1.0)
    with pytest.raises(ValueError, match="non-negative"):
        apply_weighted_correction(real, pseudo, target, [-0.1, 1.1], 1.0)


def test_target_teacher_artifacts_require_exact_blind_confidence_extremes():
    targets = []
    predictions = []
    for accent in TARGET_ACCENTS:
        for index in range(800):
            uid = f"{accent}-{index:04d}"
            target = {
                "utt_id": uid, "speaker_id": f"{accent}-speaker-{index}",
                "accent": accent, "audio_path": f"wav/{uid}.wav",
                "audio_sha256": f"{index:064x}", "duration_s": 2.0,
                "source_sample_rate": 48000, "source_dataset_split": "train",
            }
            targets.append(target)
            predictions.append({
                **target, "pseudo_text": f"prediction {index}",
                "teacher_mean_logprob": -float(800 - index),
                "teacher_min_logprob": -float(801 - index),
                "teacher_token_count": 2, "teacher_generated_token_ids": [1, 2],
            })
    audit = []
    for accent in TARGET_ACCENTS:
        group = sorted(
            (row for row in predictions if row["accent"] == accent),
            key=lambda row: (row["teacher_mean_logprob"], row["utt_id"]),
        )
        for stratum, selected in (("low", group[:10]), ("high", group[-10:])):
            for row in selected:
                audit.append({
                    "utt_id": row["utt_id"], "accent": accent,
                    "teacher_mean_logprob": row["teacher_mean_logprob"],
                    "confidence_stratum": stratum,
                })
    report = validate_target_teacher_artifacts(targets, predictions, audit)
    assert report["pseudo_id_coverage"] == 1.0
    audit[0]["pseudo_text"] = "leak"
    with pytest.raises(ValueError, match="leaks"):
        validate_target_teacher_artifacts(targets, predictions, audit)
