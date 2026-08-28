"""Pure-Python tests for the cost guardrail -- no torch, no network, no GPU."""

import pytest

from onpolicy_distill.config import (
    BudgetExceededError,
    PRESETS,
    SMOKE,
    affordable_steps,
    enforce_budget,
    estimate_cost,
    gpu_second_rate,
)


def test_gpu_second_rate_matches_hourly():
    assert gpu_second_rate("T4") == pytest.approx(0.59 / 3600, rel=1e-6)


def test_gpu_second_rate_unknown_gpu_raises():
    with pytest.raises(ValueError):
        gpu_second_rate("H42000")


def test_estimate_cost_arithmetic():
    est = estimate_cost("T4", num_steps=60, seconds_per_step=10.0, max_cost_usd=1.0)
    assert est.est_seconds == 600.0
    assert est.est_usd == pytest.approx(600 * (0.59 / 3600), rel=1e-6)
    assert est.within_budget


def test_enforce_budget_raises_when_over():
    est = estimate_cost("A100-80GB", num_steps=1000, seconds_per_step=10.0, max_cost_usd=0.5)
    assert not est.within_budget
    with pytest.raises(BudgetExceededError):
        enforce_budget(est)


def test_enforce_budget_passes_when_under():
    est = estimate_cost("T4", num_steps=5, seconds_per_step=1.0, max_cost_usd=1.0)
    enforce_budget(est)  # should not raise


def test_affordable_steps_shrinks_plan_within_budget():
    steps = affordable_steps("T4", measured_seconds_per_step=10.0, remaining_budget_usd=0.10)
    max_cost = steps * 10.0 * gpu_second_rate("T4")
    assert max_cost <= 0.10 + 1e-9
    assert steps >= 0


def test_affordable_steps_zero_when_no_budget_left():
    assert affordable_steps("T4", 10.0, remaining_budget_usd=0.0) == 0


def test_presets_are_all_sub_dollar_by_design():
    for cfg in PRESETS.values():
        assert cfg.max_cost_usd <= 1.0


def test_smoke_preset_cheapest():
    assert SMOKE.max_cost_usd <= 0.20
