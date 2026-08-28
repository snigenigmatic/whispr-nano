"""CPU integration test exercising the *real* Hugging Face Whisper stack end
to end (model loading, feature extraction, rollout sampling, cross-model
teacher-forced scoring, reverse-KL loss, backward, optimizer step).

Uses `openai/whisper-tiny` in both the student and "teacher" role so it runs
in seconds on CPU with no GPU and no Modal spend -- this is exactly the gate
we run before paying for anything on Modal. It downloads ~150 MB (the tiny
checkpoint) plus a few KB of dummy audio on first run.
"""

import torch

from onpolicy_distill.config import DistillConfig
from onpolicy_distill.data import ClipCycler, load_clips
from onpolicy_distill.distill import build_vocab_map, load_model_bundle, train_arm
from onpolicy_distill.evaluate import evaluate


def _tiny_cfg(policy: str) -> DistillConfig:
    return DistillConfig(
        name=f"cpu_test_{policy}",
        student_model="openai/whisper-tiny",
        teacher_model="openai/whisper-tiny",  # stand-in "teacher" for a fast CPU check
        policy=policy,
        num_train_clips=4,
        num_eval_clips=2,
        num_steps=2,
        batch_size=2,
        max_new_tokens=8,
        eval_every=2,
        log_every=1,
        max_cost_usd=100.0,  # irrelevant on CPU; guardrail is exercised in test_config.py
    )


def test_end_to_end_on_policy_cpu():
    cfg = _tiny_cfg("on_policy")
    device = torch.device("cpu")

    student = load_model_bundle(cfg.student_model, cfg.language, cfg.task, device, torch.float32, trainable=True)
    teacher = load_model_bundle(cfg.teacher_model, cfg.language, cfg.task, device, torch.float32, trainable=False)
    vocab_map = build_vocab_map(teacher, student)
    assert vocab_map.shape == (student.vocab_size,)
    # Same checkpoint on both sides -> vocab map must be the identity.
    assert torch.equal(vocab_map, torch.arange(student.vocab_size))

    train_clips, eval_clips = load_clips(
        cfg.dataset_id, cfg.dataset_config, cfg.dataset_split, cfg.num_train_clips, cfg.num_eval_clips, seed=0
    )
    assert len(train_clips) == 4 and len(eval_clips) == 2

    result = train_arm(
        cfg, student, teacher, vocab_map, train_clips, eval_clips,
        already_spent_usd=0.0, evaluate_fn=evaluate,
    )

    assert result.steps_run == 2
    assert not result.stopped_early
    assert "student_wer" in result.baseline_eval
    assert "student_wer" in result.final_eval
    assert all(torch.isfinite(torch.tensor(s.loss)) for s in result.step_log)
    assert all(s.loss >= 0 for s in result.step_log)


def test_end_to_end_off_policy_cpu():
    cfg = _tiny_cfg("off_policy")
    device = torch.device("cpu")

    student = load_model_bundle(cfg.student_model, cfg.language, cfg.task, device, torch.float32, trainable=True)
    teacher = load_model_bundle(cfg.teacher_model, cfg.language, cfg.task, device, torch.float32, trainable=False)
    vocab_map = build_vocab_map(teacher, student)

    train_clips, eval_clips = load_clips(
        cfg.dataset_id, cfg.dataset_config, cfg.dataset_split, cfg.num_train_clips, cfg.num_eval_clips, seed=1
    )

    result = train_arm(
        cfg, student, teacher, vocab_map, train_clips, eval_clips,
        already_spent_usd=0.0, evaluate_fn=evaluate,
    )
    assert result.steps_run == 2
    assert all(s.loss >= 0 for s in result.step_log)


def test_clip_cycler_cycles_without_replacement_within_epoch():
    train_clips, _ = load_clips(
        "hf-internal-testing/librispeech_asr_dummy", "clean", "validation", num_train=6, num_eval=1, seed=0
    )
    cycler = ClipCycler(train_clips, batch_size=4, seed=0)
    first = cycler.next_batch()
    second = cycler.next_batch()
    seen_ids = {c.clip_id for c in first} | {c.clip_id for c in second}
    assert len(seen_ids) == 6  # exactly one full epoch across two batches of 4 (with wraparound)
