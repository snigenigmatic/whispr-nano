"""Run configuration and Modal-cost accounting.

This module intentionally has zero heavy dependencies (no torch, no
transformers) so it can be imported instantly to size a run and enforce a
dollar budget *before* a GPU container is ever started, and so it can be unit
tested without touching Modal, CUDA, or the network at all.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

# Modal's published per-second GPU rates (https://modal.com/pricing, checked
# 2026-08-28). Used only for our own pre-flight cost estimate; Modal is the
# source of truth for actual billing.
GPU_HOURLY_USD: dict[str, float] = {
    "T4": 0.59,
    "L4": 0.80,
    "A10": 1.10,
    "A10G": 1.10,
    "L40S": 1.95,
    "A100-40GB": 2.10,
    "A100-80GB": 2.50,
    "H100": 3.95,
}


def gpu_second_rate(gpu_type: str) -> float:
    try:
        return GPU_HOURLY_USD[gpu_type] / 3600.0
    except KeyError as exc:
        raise ValueError(
            f"Unknown GPU type {gpu_type!r}; known types: {sorted(GPU_HOURLY_USD)}"
        ) from exc


@dataclass(frozen=True)
class DistillConfig:
    """Hyperparameters for one on-policy (or off-policy) distillation arm."""

    name: str
    student_model: str = "openai/whisper-tiny"
    teacher_model: str = "openai/whisper-large-v3"
    gpu_type: str = "T4"

    # Data: a small, real, standard ASR benchmark subset (LibriSpeech
    # `clean`/`validation`, 73 clips total) is used instead of streaming the
    # full LibriSpeech-960 corpus -- see PROPOSAL.md for the scale-up plan.
    dataset_id: str = "hf-internal-testing/librispeech_asr_dummy"
    dataset_config: str = "clean"
    dataset_split: str = "validation"
    num_train_clips: int = 50
    num_eval_clips: int = 20
    language: str = "en"
    task: str = "transcribe"

    # Optimization
    num_steps: int = 60
    batch_size: int = 4
    learning_rate: float = 1e-5
    max_new_tokens: int = 64
    temperature: float = 1.0
    seed: int = 0

    # "on_policy": rollouts sampled from the student, scored by the teacher
    #   (Agarwal et al. 2023 / GKD; Thinking Machines Lab 2025).
    # "off_policy": rollouts sampled from the teacher (pseudo-labels), scored
    #   with the *same* reverse-KL loss -- the controlled ablation from
    #   Agarwal et al. 2023, Table 1, isolating rollout source from loss choice.
    policy: str = "on_policy"

    # Eval / logging
    eval_every: int = 20
    log_every: int = 5

    # Cost guardrail. `assumed_seconds_per_step` is a deliberately conservative
    # prior used for the *pre-flight* estimate before any GPU spend; the
    # in-run guardrail replaces it with measured timings after a short warmup.
    assumed_seconds_per_step: float = 15.0
    max_cost_usd: float = 0.60
    hard_timeout_s: int = 900

    def clone(self, **overrides) -> "DistillConfig":
        return replace(self, **overrides)

    def as_dict(self) -> dict:
        return {k: getattr(self, k) for k in self.__dataclass_fields__}


@dataclass(frozen=True)
class CostEstimate:
    gpu_type: str
    seconds_per_step: float
    num_steps: int
    est_seconds: float
    est_usd: float
    max_cost_usd: float
    basis: str  # "assumed" (pre-flight) or "measured" (in-run)

    @property
    def within_budget(self) -> bool:
        return self.est_usd <= self.max_cost_usd

    def report(self) -> str:
        return (
            f"[cost-guardrail:{self.basis}] {self.gpu_type} @ "
            f"{self.seconds_per_step:.2f}s/step x {self.num_steps} steps = "
            f"{self.est_seconds:.0f}s (~{self.est_seconds / 60:.1f} min) -> "
            f"${self.est_usd:.4f} projected vs ${self.max_cost_usd:.2f} budget "
            f"[{'OK' if self.within_budget else 'OVER BUDGET'}]"
        )


class BudgetExceededError(RuntimeError):
    """Raised when a projected or measured cost would exceed --max-cost-usd."""


def estimate_cost(
    gpu_type: str,
    num_steps: int,
    seconds_per_step: float,
    max_cost_usd: float,
    basis: str = "assumed",
) -> CostEstimate:
    est_seconds = num_steps * seconds_per_step
    est_usd = est_seconds * gpu_second_rate(gpu_type)
    return CostEstimate(
        gpu_type=gpu_type,
        seconds_per_step=seconds_per_step,
        num_steps=num_steps,
        est_seconds=est_seconds,
        est_usd=est_usd,
        max_cost_usd=max_cost_usd,
        basis=basis,
    )


def enforce_budget(estimate: CostEstimate, raise_on_exceed: bool = True) -> CostEstimate:
    print(estimate.report(), flush=True)
    if raise_on_exceed and not estimate.within_budget:
        raise BudgetExceededError(
            f"Projected cost ${estimate.est_usd:.4f} exceeds --max-cost-usd "
            f"${estimate.max_cost_usd:.2f}. Reduce steps/batch size, pick a "
            f"cheaper GPU, or raise the budget explicitly."
        )
    return estimate


def affordable_steps(
    gpu_type: str,
    measured_seconds_per_step: float,
    remaining_budget_usd: float,
) -> int:
    """How many more steps fit in the remaining dollar budget, given a measured rate."""
    if measured_seconds_per_step <= 0:
        return 0
    rate = gpu_second_rate(gpu_type)
    if rate <= 0:
        return 10**9
    remaining_seconds = remaining_budget_usd / rate
    return max(0, int(remaining_seconds // measured_seconds_per_step))


# ---------------------------------------------------------------------------
# Named presets. `poc` deliberately targets total spend well under $1: see
# PROPOSAL.md for the arithmetic and the scale-up plan real credits unlock.
# ---------------------------------------------------------------------------

SMOKE = DistillConfig(
    name="smoke_on_policy",
    policy="on_policy",
    num_train_clips=16,
    num_eval_clips=8,
    num_steps=16,
    batch_size=2,
    max_new_tokens=32,
    eval_every=8,
    log_every=4,
    assumed_seconds_per_step=10.0,  # fp32 on T4; see modal_app.py for why fp16 is avoided
    max_cost_usd=0.10,
    hard_timeout_s=300,
)

SMOKE_OFF_POLICY = SMOKE.clone(
    name="smoke_off_policy",
    policy="off_policy",
    assumed_seconds_per_step=14.0,
    max_cost_usd=0.10,
)

POC_ON_POLICY = DistillConfig(
    name="poc_on_policy",
    policy="on_policy",
    num_train_clips=50,
    num_eval_clips=20,
    num_steps=60,
    batch_size=4,
    max_new_tokens=64,
    eval_every=20,
    log_every=5,
    assumed_seconds_per_step=14.0,  # fp32 on T4; see modal_app.py for why fp16 is avoided
    max_cost_usd=0.30,
    hard_timeout_s=900,
)

POC_OFF_POLICY = POC_ON_POLICY.clone(
    name="poc_off_policy",
    policy="off_policy",
    assumed_seconds_per_step=20.0,  # teacher (1.55B) must also autoregressively generate
    max_cost_usd=0.30,
)

PRESETS: dict[str, DistillConfig] = {
    c.name: c for c in [SMOKE, SMOKE_OFF_POLICY, POC_ON_POLICY, POC_OFF_POLICY]
}
