"""Single fixed entrypoint for every orx experiment node.

Reads committed `experiments/run.json`, trains one arm, prints a RESULT
SUMMARY block that `orx logs` can judge. Adds this snapshot's `src/` to
sys.path so isolated orx run directories use the node's code, not whatever
happens to be editable-installed on the machine.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

os.environ.setdefault("HF_HOME", str(Path.home() / ".cache" / "huggingface"))
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

CONFIG_PATH = ROOT / "experiments" / "run.json"


def load_config():
    from onpolicy_distill.config import DistillConfig

    raw = json.loads(CONFIG_PATH.read_text())
    allowed = set(DistillConfig.__dataclass_fields__)
    unknown = sorted(set(raw) - allowed)
    if unknown:
        raise SystemExit(f"experiments/run.json has unknown keys: {unknown}")
    return DistillConfig(**raw)


def pick_device():
    import torch

    if torch.cuda.is_available():
        return torch.device("cuda"), torch.float32
    return torch.device("cpu"), torch.float32


def print_summary(cfg, result, device) -> None:
    base_wer = float(result.baseline_eval["student_wer"])
    final_wer = float(result.final_eval["student_wer"])
    delta_pp = (final_wer - base_wer) * 100.0
    if delta_pp <= -1.0:
        verdict = "improved"
    elif delta_pp >= 1.0:
        verdict = "regressed"
    else:
        verdict = "flat"

    print("===== RESULT SUMMARY =====", flush=True)
    print(f"variant: {cfg.name}", flush=True)
    print(f"policy: {cfg.policy}", flush=True)
    print(f"scheduled_sampling_prob: {cfg.scheduled_sampling_prob}", flush=True)
    print(f"student: {cfg.student_model}", flush=True)
    print(f"teacher: {cfg.teacher_model}", flush=True)
    print(f"device: {device}", flush=True)
    print(f"dataset: {cfg.dataset_id} {cfg.dataset_config}/{cfg.dataset_split}", flush=True)
    print(
        f"clips: train={cfg.num_train_clips} eval={cfg.num_eval_clips} "
        f"steps={result.steps_run}/{result.steps_planned} seed={cfg.seed}",
        flush=True,
    )
    print(f"learning_rate: {cfg.learning_rate}", flush=True)
    print(f"baseline_student_wer: {base_wer:.4f}", flush=True)
    print(f"final_student_wer: {final_wer:.4f}", flush=True)
    print(f"delta_wer_pp: {delta_pp:+.2f}", flush=True)
    print(f"baseline_teacher_wer: {result.baseline_eval['teacher_wer']:.4f}", flush=True)
    print(f"baseline_mean_kl: {result.baseline_eval['mean_reverse_kl']:.4f}", flush=True)
    print(f"final_mean_kl: {result.final_eval['mean_reverse_kl']:.4f}", flush=True)
    print(f"stopped_early: {result.stopped_early} reason={result.stop_reason}", flush=True)
    print(f"wall_seconds: {result.wall_seconds:.1f}", flush=True)
    print(f"verdict: {verdict}", flush=True)
    print("==========================", flush=True)


def main() -> int:
    import torch

    from onpolicy_distill.data import load_clips
    from onpolicy_distill.distill import build_vocab_map, load_model_bundle, train_arm
    from onpolicy_distill.evaluate import evaluate

    cfg = load_config()
    device, dtype = pick_device()
    print(f"[experiments.run] loaded {CONFIG_PATH}", flush=True)
    print(f"[experiments.run] {cfg.as_dict()}", flush=True)
    print(f"[experiments.run] device={device} dtype={dtype}", flush=True)

    torch.manual_seed(cfg.seed)
    student = load_model_bundle(
        cfg.student_model, cfg.language, cfg.task, device, dtype, trainable=True
    )
    teacher = load_model_bundle(
        cfg.teacher_model, cfg.language, cfg.task, device, dtype, trainable=False
    )
    vocab_map = build_vocab_map(teacher, student)
    train_clips, eval_clips = load_clips(
        cfg.dataset_id,
        cfg.dataset_config,
        cfg.dataset_split,
        cfg.num_train_clips,
        cfg.num_eval_clips,
        seed=cfg.seed,
    )
    result = train_arm(
        cfg,
        student,
        teacher,
        vocab_map,
        train_clips,
        eval_clips,
        already_spent_usd=0.0,
        evaluate_fn=evaluate,
    )
    print_summary(cfg, result, device)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
