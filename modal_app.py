"""On-policy distillation of whisper-tiny from whisper-large-v3, on Modal.

    modal run modal_app.py::hello   # near-zero-cost auth + image-build check (no GPU)
    modal run modal_app.py::smoke   # ~$0.05 on a T4, validates the full loop end-to-end
    modal run modal_app.py::poc     # ~$0.30-0.60 on a T4, the on-policy vs off-policy comparison

Every entrypoint that touches a GPU computes a *pre-flight* cost estimate
(steps x assumed seconds/step x Modal's published $/GPU-second) and refuses
to call `.remote()` at all if that projection exceeds `--max-cost-usd`. Once
a GPU function is running, `train_arm` (see `onpolicy_distill/distill.py`)
re-estimates from *measured* per-step latency after a short warmup and caps
the remaining step count if needed. The Modal function `timeout=` is the
final, unconditional backstop against runaway spend. See PROPOSAL.md for the
full budget accounting.
"""

from __future__ import annotations

import base64
import json
import time
from dataclasses import asdict
from pathlib import Path

import modal

from onpolicy_distill.config import (
    PRESETS,
    DistillConfig,
    enforce_budget,
    estimate_cost,
    gpu_second_rate,
)

# ---------------------------------------------------------------------------
# App, image, volumes
# ---------------------------------------------------------------------------

app = modal.App("onpolicy-whisper-distill")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("ffmpeg")
    .pip_install(
        "torch==2.13.0",
        "transformers==5.16.1",
        "accelerate==1.11.0",
        "datasets==4.4.1",
        "torchcodec==0.16.0",
        "soundfile==0.14.0",
        "librosa==1.0.0",
        "jiwer==4.0.0",
        "numpy==2.5.2",
        "matplotlib==3.11.1",
    )
    .env({"HF_HOME": "/cache/huggingface", "HF_HUB_ENABLE_HF_TRANSFER": "0"})
    .add_local_python_source("onpolicy_distill")
)

hf_cache_volume = modal.Volume.from_name("onpolicy-distill-hf-cache", create_if_missing=True)
results_volume = modal.Volume.from_name("onpolicy-distill-results", create_if_missing=True)

HF_CACHE_PATH = "/cache/huggingface"
RESULTS_PATH = "/results"

LOCAL_RESULTS_DIR = Path(__file__).parent / "results"


# ---------------------------------------------------------------------------
# Remote functions
# ---------------------------------------------------------------------------


@app.function(image=image, timeout=120, scaledown_window=15)
def hello_check() -> dict:
    """No GPU requested -- confirms Modal auth and that the image builds,
    at CPU-container rates (a small fraction of a cent), before any GPU
    spend is risked."""
    import torch

    return {
        "modal_auth_ok": True,
        "torch_version": torch.__version__,
        "cuda_available_in_this_cpu_container": torch.cuda.is_available(),
    }


@app.function(
    image=image,
    gpu="T4",
    timeout=1400,
    scaledown_window=15,
    volumes={HF_CACHE_PATH: hf_cache_volume, RESULTS_PATH: results_volume},
)
def run_poc(config_dicts: list[dict]) -> dict:
    """Load the teacher once, then run each config in `config_dicts` as one
    training arm against the same held-out eval set, for a fair on-policy
    vs. off-policy comparison. Writes JSON + plots to the results Volume in
    addition to returning them, and commits the HF cache Volume so later
    runs (e.g. `poc` after `smoke`) skip re-downloading whisper-large-v3."""
    import torch

    from onpolicy_distill.data import load_clips
    from onpolicy_distill.distill import build_vocab_map, load_model_bundle, train_arm
    from onpolicy_distill.evaluate import evaluate
    from onpolicy_distill.plotting import loss_curve_png, wer_bar_png

    configs = [DistillConfig(**d) for d in config_dicts]
    base = configs[0]
    shared_data_key = lambda c: (  # noqa: E731
        c.dataset_id, c.dataset_config, c.dataset_split, c.num_train_clips, c.num_eval_clips, c.seed
    )
    for c in configs[1:]:
        assert shared_data_key(c) == shared_data_key(base), (
            "all configs in one run_poc call must share the same data settings "
            "so the arms are directly comparable"
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # fp16 sampling in HF's Whisper `generate()` is known to occasionally
    # produce NaN/inf in the softmax over logits on some GPUs/checkpoints
    # (a documented transformers/Whisper issue, not specific to this code --
    # confirmed here too: identical code path is stable in fp32 on CPU but
    # crashed with a CUDA "probability tensor contains inf/nan" assertion in
    # fp16 on a T4). Both models comfortably fit a T4's 16GB in fp32, so we
    # trade some speed for correctness rather than chase fp16 numerics.
    dtype = torch.float32
    run_started = time.monotonic()

    print(f"[run_poc] device={device} dtype={dtype}", flush=True)
    print(f"[run_poc] loading teacher {base.teacher_model} ...", flush=True)
    teacher = load_model_bundle(
        base.teacher_model, base.language, base.task, device, dtype, trainable=False
    )

    n_clips = base.num_train_clips + base.num_eval_clips
    print(f"[run_poc] loading {n_clips} clips from {base.dataset_id} ...", flush=True)
    train_clips, eval_clips = load_clips(
        base.dataset_id, base.dataset_config, base.dataset_split,
        base.num_train_clips, base.num_eval_clips, seed=base.seed,
    )
    hf_cache_volume.commit()

    arms: dict[str, dict] = {}
    already_spent_usd = 0.0
    for cfg in configs:
        print(f"\n[run_poc] === arm: {cfg.name} (policy={cfg.policy}) ===", flush=True)
        student = load_model_bundle(
            cfg.student_model, cfg.language, cfg.task, device, dtype, trainable=True
        )
        vocab_map = build_vocab_map(teacher, student)
        result = train_arm(
            cfg, student, teacher, vocab_map, train_clips, eval_clips,
            already_spent_usd=already_spent_usd, evaluate_fn=evaluate,
        )
        already_spent_usd += result.est_cost_usd
        arms[cfg.name] = {
            "config": cfg.as_dict(),
            "baseline_eval": result.baseline_eval,
            "final_eval": result.final_eval,
            "step_log": [asdict(s) for s in result.step_log],
            "steps_run": result.steps_run,
            "steps_planned": result.steps_planned,
            "stopped_early": result.stopped_early,
            "stop_reason": result.stop_reason,
            "wall_seconds": result.wall_seconds,
            "est_cost_usd": result.est_cost_usd,
        }
        del student
        if device.type == "cuda":
            torch.cuda.empty_cache()

    total_wall = time.monotonic() - run_started
    total_cost = total_wall * gpu_second_rate(base.gpu_type)

    summary = {
        "gpu_type": base.gpu_type,
        "teacher_model": base.teacher_model,
        "student_model": base.student_model,
        "device": str(device),
        "dtype": str(dtype),
        "arms": arms,
        "total_wall_seconds": total_wall,
        "total_est_cost_usd": total_cost,
    }

    plots: dict[str, str] = {}
    try:
        plots["loss_curve.png"] = base64.b64encode(loss_curve_png(arms)).decode()
        plots["wer_comparison.png"] = base64.b64encode(wer_bar_png(arms)).decode()
    except Exception as e:  # pragma: no cover - plotting must never fail the run
        print(f"[run_poc] plotting failed (non-fatal): {e}", flush=True)

    run_id = time.strftime("%Y%m%d-%H%M%S")
    out_dir = Path(RESULTS_PATH) / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    for name, b64 in plots.items():
        with open(out_dir / name, "wb") as f:
            f.write(base64.b64decode(b64))
    results_volume.commit()

    summary["run_id"] = run_id
    summary["plots"] = plots
    return summary


# ---------------------------------------------------------------------------
# Local entrypoints
# ---------------------------------------------------------------------------


def _save_summary(summary: dict, run_kind: str) -> None:
    LOCAL_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    run_id = summary.get("run_id", time.strftime("%Y%m%d-%H%M%S"))
    prefix = f"{run_kind}_{run_id}"

    plots = summary.pop("plots", {})
    json_path = LOCAL_RESULTS_DIR / f"{prefix}_summary.json"
    json_path.write_text(json.dumps(summary, indent=2))
    print(f"\nWrote {json_path}")

    for name, b64 in plots.items():
        png_path = LOCAL_RESULTS_DIR / f"{prefix}_{name}"
        png_path.write_bytes(base64.b64decode(b64))
        print(f"Wrote {png_path}")

    print(f"\n=== {run_kind} report ===")
    print(f"gpu={summary['gpu_type']} device={summary['device']} "
          f"total_wall={summary['total_wall_seconds']:.1f}s "
          f"total_est_cost=${summary['total_est_cost_usd']:.4f}")
    for name, arm in summary["arms"].items():
        b = arm["baseline_eval"]
        f_ = arm["final_eval"]
        print(f"\n[{name}] policy={arm['config']['policy']} "
              f"steps={arm['steps_run']}/{arm['steps_planned']} "
              f"({'STOPPED EARLY: ' + str(arm['stop_reason']) if arm['stopped_early'] else 'completed'})")
        print(f"  wall={arm['wall_seconds']:.1f}s est_cost=${arm['est_cost_usd']:.4f}")
        print(f"  student WER  before -> after : {b['student_wer']*100:.2f}% -> {f_['student_wer']*100:.2f}%")
        print(f"  teacher WER (reference)       : {b['teacher_wer']*100:.2f}%")
        print(f"  mean reverse KL before -> after: {b['mean_reverse_kl']:.4f} -> {f_['mean_reverse_kl']:.4f}")


@app.local_entrypoint()
def hello():
    """`modal run modal_app.py::hello` -- near-zero-cost auth/build check."""
    print("Checking Modal auth + image build (no GPU requested)...")
    result = hello_check.remote()
    print(json.dumps(result, indent=2))


@app.local_entrypoint()
def smoke(max_cost_usd: float = 0.20):
    """`modal run modal_app.py::smoke` -- ~$0.05 end-to-end check of *both*
    the on-policy and off-policy code paths on a real GPU, before committing
    to the larger `poc` run."""
    per_arm_budget = max_cost_usd / 2
    on_cfg = PRESETS["smoke_on_policy"].clone(max_cost_usd=per_arm_budget)
    off_cfg = PRESETS["smoke_off_policy"].clone(max_cost_usd=per_arm_budget)

    projected_total = 0.0
    for cfg in (on_cfg, off_cfg):
        est = estimate_cost(cfg.gpu_type, cfg.num_steps, cfg.assumed_seconds_per_step, cfg.max_cost_usd)
        enforce_budget(est)  # raises BudgetExceededError before any GPU is requested
        projected_total += est.est_usd

    print(f"Pre-flight projected total across both smoke arms: ${projected_total:.4f} "
          f"(budget ${max_cost_usd:.2f})")
    print(f"Launching smoke run on Modal ({on_cfg.gpu_type})...")
    summary = run_poc.remote([on_cfg.as_dict(), off_cfg.as_dict()])
    _save_summary(summary, "smoke")


@app.local_entrypoint()
def poc(max_cost_usd: float = 0.60):
    """`modal run modal_app.py::poc` -- the real on-policy vs off-policy
    comparison, budget split evenly across the two arms."""
    per_arm_budget = max_cost_usd / 2
    on_cfg = PRESETS["poc_on_policy"].clone(max_cost_usd=per_arm_budget)
    off_cfg = PRESETS["poc_off_policy"].clone(max_cost_usd=per_arm_budget)

    projected_total = 0.0
    for cfg in (on_cfg, off_cfg):
        est = estimate_cost(cfg.gpu_type, cfg.num_steps, cfg.assumed_seconds_per_step, cfg.max_cost_usd)
        enforce_budget(est)
        projected_total += est.est_usd

    print(f"Pre-flight projected total across both arms: ${projected_total:.4f} "
          f"(budget ${max_cost_usd:.2f})")
    print(f"Launching poc run on Modal ({on_cfg.gpu_type}, on-policy + off-policy arms)...")
    summary = run_poc.remote([on_cfg.as_dict(), off_cfg.as_dict()])
    _save_summary(summary, "poc")
