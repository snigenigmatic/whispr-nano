"""On-policy distillation on real Indian-accented speech, with a per-family
(Indo-Aryan / Dravidian / Sino-Tibetan) breakdown of reverse KL and WER
before and after training -- the direct extension toward "does distillation
preserve or amplify accent bias?" using real accented audio instead of the
core PoC's generic LibriSpeech clips.

This is a standalone add-on, deliberately reusing without modification:
  - `onpolicy_distill.distill.{load_model_bundle, build_vocab_map, train_arm}`
  - `onpolicy_distill.evaluate.evaluate` (wrapped per family below, not changed)
  - `onpolicy_distill.config.{DistillConfig, estimate_cost, enforce_budget,
    affordable_steps, gpu_second_rate}` -- same two-layer cost guardrail as
    `modal_app.py` and `audit/modal_svarah_audit.py`
  - the same image-build pattern, `onpolicy-distill-hf-cache` Volume (so the
    already-downloaded whisper-large-v3 fp32 checkpoint is reused), and the
    `huggingface` Modal secret needed for the gated `ai4bharat/svarah` dataset
  - `audit/family_mapping.py`'s Svarah-accent -> family table

The only new pieces are a Svarah clip loader (stratified per family, so a
small train/eval pool still covers the minority Sino-Tibetan/Bodo group) and
a thin per-family wrapper around the existing `evaluate()` function.

Usage:
    modal run audit/modal_svarah_opd.py::smoke   # ~$0.05-0.10, both arms, small
    modal run audit/modal_svarah_opd.py::run     # ~$0.30-0.60, both arms, the real comparison

Honest scope: this is a PoC-scale extension of an already-PoC-scale repo.
Small stratified pools (tens of clips per family), one seed, one run. It is
evidence that the mechanism is worth pointing at real accented data, not a
publication-grade fairness result.
"""

from __future__ import annotations

import sys
import time
from dataclasses import asdict
from pathlib import Path

import modal

from onpolicy_distill.config import (
    DistillConfig,
    enforce_budget,
    estimate_cost,
    gpu_second_rate,
)

sys.path.insert(0, str(Path(__file__).parent))

# ---------------------------------------------------------------------------
# App, image, volume, secret -- same pattern as modal_app.py / modal_svarah_audit.py
# ---------------------------------------------------------------------------

app = modal.App("svarah-opd")  # separate from "onpolicy-whisper-distill" and "svarah-family-audit"

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
    .add_local_file(str(Path(__file__).parent / "family_mapping.py"), "/root/family_mapping.py")
)

hf_cache_volume = modal.Volume.from_name("onpolicy-distill-hf-cache", create_if_missing=True)
HF_CACHE_PATH = "/cache/huggingface"
hf_secret = modal.Secret.from_name("huggingface")  # ai4bharat/svarah is gated

GPU_TYPE = "T4"
HARD_TIMEOUT_S = 1800  # 30 min hard backstop per arm-pair call, generous over measured smoke throughput
LOCAL_RESULTS_DIR = Path(__file__).parent

# Svarah utterances run longer than the LibriSpeech-dummy clips the core PoC
# was tuned on (up to ~25s vs a few seconds); max_new_tokens is raised
# accordingly, and the assumed pre-flight seconds/step is doubled versus the
# core PoC's LibriSpeech presets as a conservative prior -- the in-run
# guardrail replaces this with the *measured* rate after 3 steps, same as
# `train_arm` always does.
SVARAH_SMOKE_ON = DistillConfig(
    name="svarah_smoke_on_policy",
    policy="on_policy",
    dataset_id="ai4bharat/svarah",
    num_train_clips=9,   # 3 per family
    num_eval_clips=9,    # 3 per family
    num_steps=12,
    batch_size=3,
    max_new_tokens=96,
    eval_every=6,
    log_every=3,
    assumed_seconds_per_step=20.0,
    max_cost_usd=0.10,
    hard_timeout_s=HARD_TIMEOUT_S,
)
SVARAH_SMOKE_OFF = SVARAH_SMOKE_ON.clone(name="svarah_smoke_off_policy", policy="off_policy", assumed_seconds_per_step=28.0)

SVARAH_POC_ON = DistillConfig(
    name="svarah_poc_on_policy",
    policy="on_policy",
    dataset_id="ai4bharat/svarah",
    num_train_clips=45,  # 15 per family
    num_eval_clips=24,   # 8 per family
    num_steps=60,
    batch_size=3,
    max_new_tokens=112,
    eval_every=20,
    log_every=5,
    assumed_seconds_per_step=24.0,
    max_cost_usd=0.35,
    hard_timeout_s=HARD_TIMEOUT_S,
)
SVARAH_POC_OFF = SVARAH_POC_ON.clone(name="svarah_poc_off_policy", policy="off_policy", assumed_seconds_per_step=34.0, max_cost_usd=0.35)


# ---------------------------------------------------------------------------
# Svarah clip loading, stratified per family so a small pool still covers
# the minority Sino-Tibetan/Bodo group (only ~6% of Svarah's utterances)
# ---------------------------------------------------------------------------


def _load_svarah_opd_clips(num_train_per_family: int, num_eval_per_family: int, seed: int):
    import numpy as np
    from datasets import Audio, load_dataset

    from family_mapping import FAMILIES, get_family
    from onpolicy_distill.data import Clip, _decode_audio

    ds = load_dataset("ai4bharat/svarah", split="test")
    ds = ds.cast_column("audio_filepath", Audio(sampling_rate=16_000))

    by_family: dict[str, list[int]] = {f: [] for f in FAMILIES}
    for i, lang in enumerate(ds["primary_language"]):
        by_family[get_family(lang)].append(i)

    rng = np.random.default_rng(seed)
    train_clips, eval_clips = [], []

    def to_clip(i: int, family: str):
        row = ds[int(i)]
        array, sr = _decode_audio(row["audio_filepath"])
        return Clip(
            clip_id=f"svarah_{int(i):05d}",
            audio=array,
            sampling_rate=sr,
            reference_text=row["text"],
            family=family,
        )

    counts = {}
    for family, indices in by_family.items():
        needed = num_train_per_family + num_eval_per_family
        if len(indices) < needed:
            raise ValueError(f"family {family!r} has only {len(indices)} Svarah utterances, need {needed}")
        chosen = rng.choice(indices, size=needed, replace=False)
        train_clips.extend(to_clip(i, family) for i in chosen[:num_train_per_family])
        eval_clips.extend(to_clip(i, family) for i in chosen[num_train_per_family:])
        counts[family] = len(indices)

    print(f"[svarah-opd] family pool sizes in full dataset: {counts}", flush=True)
    return train_clips, eval_clips


def _evaluate_with_family_breakdown(student, teacher, vocab_map, clips, cfg):
    from onpolicy_distill.evaluate import evaluate

    overall = evaluate(student, teacher, vocab_map, clips, cfg, num_samples=2)
    families = sorted({c.family for c in clips if c.family})
    by_family = {}
    for family in families:
        family_clips = [c for c in clips if c.family == family]
        if len(family_clips) < 2:
            continue
        fe = evaluate(student, teacher, vocab_map, family_clips, cfg, num_samples=0)
        by_family[family] = {
            "n": len(family_clips),
            "student_wer": fe["student_wer"],
            "teacher_wer": fe["teacher_wer"],
            "mean_reverse_kl": fe["mean_reverse_kl"],
        }
    overall["by_family"] = by_family
    return overall


# ---------------------------------------------------------------------------
# Remote GPU function -- mirrors modal_app.py's run_poc, with a Svarah loader
# and the per-family eval wrapper in place of the LibriSpeech loader / plain evaluate()
# ---------------------------------------------------------------------------


@app.function(
    image=image,
    gpu=GPU_TYPE,
    timeout=HARD_TIMEOUT_S,
    scaledown_window=15,
    volumes={HF_CACHE_PATH: hf_cache_volume},
    secrets=[hf_secret],
)
def run_svarah_opd(config_dicts: list[dict], train_per_family: int, eval_per_family: int) -> dict:
    import torch

    from onpolicy_distill.distill import build_vocab_map, load_model_bundle, train_arm

    configs = [DistillConfig(**d) for d in config_dicts]
    base = configs[0]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32  # see distill.py / README.md -- fp16 sampling hit a CUDA NaN bug in this repo's history
    run_started = time.monotonic()

    print(f"[svarah-opd] device={device} dtype={dtype}", flush=True)
    print(f"[svarah-opd] loading teacher {base.teacher_model} ...", flush=True)
    teacher = load_model_bundle(base.teacher_model, base.language, base.task, device, dtype, trainable=False)

    print(f"[svarah-opd] loading Svarah ({train_per_family}/{eval_per_family} clips per family)...", flush=True)
    train_clips, eval_clips = _load_svarah_opd_clips(train_per_family, eval_per_family, seed=base.seed)
    print(f"[svarah-opd] loaded {len(train_clips)} train / {len(eval_clips)} eval clips", flush=True)

    arms: dict[str, dict] = {}
    already_spent_usd = 0.0
    for cfg in configs:
        print(f"\n[svarah-opd] === arm: {cfg.name} (policy={cfg.policy}) ===", flush=True)
        student = load_model_bundle(cfg.student_model, cfg.language, cfg.task, device, dtype, trainable=True)
        vocab_map = build_vocab_map(teacher, student)
        result = train_arm(
            cfg, student, teacher, vocab_map, train_clips, eval_clips,
            already_spent_usd=already_spent_usd,
            evaluate_fn=_evaluate_with_family_breakdown,
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
    return {
        "gpu_type": base.gpu_type,
        "teacher_model": base.teacher_model,
        "student_model": base.student_model,
        "device": str(device),
        "arms": arms,
        "total_wall_seconds": total_wall,
        "total_est_cost_usd": total_cost,
        "train_per_family": train_per_family,
        "eval_per_family": eval_per_family,
    }


# ---------------------------------------------------------------------------
# Local reporting
# ---------------------------------------------------------------------------


def _print_and_save(summary: dict, run_kind: str) -> None:
    import json

    run_id = time.strftime("%Y%m%d-%H%M%S")
    path = LOCAL_RESULTS_DIR / f"svarah_opd_{run_kind}_{run_id}.json"
    path.write_text(json.dumps(summary, indent=2))
    print(f"\nWrote {path}")

    print(f"\n=== svarah-opd {run_kind} report ===")
    print(f"gpu={summary['gpu_type']} device={summary['device']} "
          f"train/eval per family={summary['train_per_family']}/{summary['eval_per_family']} "
          f"total_wall={summary['total_wall_seconds']:.1f}s total_est_cost=${summary['total_est_cost_usd']:.4f}")
    for name, arm in summary["arms"].items():
        b, f_ = arm["baseline_eval"], arm["final_eval"]
        print(f"\n[{name}] policy={arm['config']['policy']} steps={arm['steps_run']}/{arm['steps_planned']} "
              f"({'STOPPED EARLY: ' + str(arm['stop_reason']) if arm['stopped_early'] else 'completed'}) "
              f"wall={arm['wall_seconds']:.1f}s cost=${arm['est_cost_usd']:.4f}")
        print(f"  overall student WER  before -> after : {b['student_wer']*100:.2f}% -> {f_['student_wer']*100:.2f}%")
        print(f"  overall teacher WER (reference)       : {b['teacher_wer']*100:.2f}%")
        print(f"  overall mean reverse KL before -> after: {b['mean_reverse_kl']:.4f} -> {f_['mean_reverse_kl']:.4f}")
        print(f"  per-family breakdown:")
        for family in sorted(set(b["by_family"]) | set(f_["by_family"])):
            bf, ff = b["by_family"].get(family), f_["by_family"].get(family)
            if not bf or not ff:
                continue
            print(f"    {family:14s} n={bf['n']:2d}  WER {bf['student_wer']*100:6.2f}% -> {ff['student_wer']*100:6.2f}%  "
                  f"KL {bf['mean_reverse_kl']:.3f} -> {ff['mean_reverse_kl']:.3f}")


@app.local_entrypoint()
def smoke(max_cost_usd: float = 0.20):
    """`modal run audit/modal_svarah_opd.py::smoke` -- both arms, tiny, on real Svarah audio."""
    per_arm_budget = max_cost_usd / 2
    on_cfg = SVARAH_SMOKE_ON.clone(max_cost_usd=per_arm_budget)
    off_cfg = SVARAH_SMOKE_OFF.clone(max_cost_usd=per_arm_budget)

    total_est = 0.0
    for cfg in (on_cfg, off_cfg):
        est = estimate_cost(cfg.gpu_type, cfg.num_steps, cfg.assumed_seconds_per_step, cfg.max_cost_usd)
        enforce_budget(est)
        total_est += est.est_usd
    print(f"Pre-flight projected total: ${total_est:.4f} (budget ${max_cost_usd:.2f})")

    print("Launching svarah-opd smoke run on Modal...")
    summary = run_svarah_opd.remote(
        [on_cfg.as_dict(), off_cfg.as_dict()],
        train_per_family=on_cfg.num_train_clips // 3,
        eval_per_family=on_cfg.num_eval_clips // 3,
    )
    _print_and_save(summary, "smoke")


@app.local_entrypoint()
def run(max_cost_usd: float = 0.70):
    """`modal run audit/modal_svarah_opd.py::run` -- the real comparison on Svarah."""
    per_arm_budget = max_cost_usd / 2
    on_cfg = SVARAH_POC_ON.clone(max_cost_usd=per_arm_budget)
    off_cfg = SVARAH_POC_OFF.clone(max_cost_usd=per_arm_budget)

    total_est = 0.0
    for cfg in (on_cfg, off_cfg):
        est = estimate_cost(cfg.gpu_type, cfg.num_steps, cfg.assumed_seconds_per_step, cfg.max_cost_usd)
        enforce_budget(est)
        total_est += est.est_usd
    print(f"Pre-flight projected total: ${total_est:.4f} (budget ${max_cost_usd:.2f})")

    print("Launching svarah-opd run on Modal...")
    summary = run_svarah_opd.remote(
        [on_cfg.as_dict(), off_cfg.as_dict()],
        train_per_family=on_cfg.num_train_clips // 3,
        eval_per_family=on_cfg.num_eval_clips // 3,
    )
    _print_and_save(summary, "run")
