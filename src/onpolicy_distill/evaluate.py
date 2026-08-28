"""Evaluation: WER (greedy decode vs. reference) and mean reverse KL between
student and teacher, computed before and after distillation.

WER follows Whisper's own evaluation protocol -- normalize text with the
English text normalizer from Radford et al. (2022, Appendix C.1) before
scoring with `jiwer`, which is also how Distil-Whisper (Gandhi et al., 2023)
and most Whisper follow-up work reports numbers, so results are comparable.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import jiwer
import torch

from .config import DistillConfig
from .data import Clip
from .distill import (
    ModelBundle,
    build_loss_mask,
    extract_features,
    reverse_kl_loss,
    retarget_prefix,
    sample_rollout,
    score_sequences,
)

if TYPE_CHECKING:
    pass


def _normalizer():
    from transformers.models.whisper.english_normalizer import EnglishTextNormalizer

    return EnglishTextNormalizer({})


def _chunks(clips: list[Clip], size: int) -> list[list[Clip]]:
    return [clips[i : i + size] for i in range(0, len(clips), size)]


def transcribe_greedy(
    bundle: ModelBundle, clips: list[Clip], language: str, task: str, max_new_tokens: int, batch_size: int = 4
) -> list[str]:
    """Greedy-decode `clips` in small sub-batches. Whisper always pads audio
    to a fixed 30s window, so activation memory scales with batch size
    regardless of actual clip length -- a large-v3 forward pass over more
    than a handful of clips at once can exceed a T4's 16GB in fp32, so we
    never pass the full eval set through in one shot."""
    predictions: list[str] = []
    for chunk in _chunks(clips, batch_size):
        feats = extract_features(bundle, chunk)
        seqs = sample_rollout(
            bundle,
            feats,
            language=language,
            task=task,
            max_new_tokens=max_new_tokens,
            temperature=0.0,
            do_sample=False,
        )
        predictions.extend(bundle.tokenizer.decode(s, skip_special_tokens=True).strip() for s in seqs)
    return predictions


def compute_wer(predictions: list[str], references: list[str]) -> float:
    norm = _normalizer()
    norm_preds = [norm(p) or " " for p in predictions]
    norm_refs = [norm(r) or " " for r in references]
    return float(jiwer.wer(norm_refs, norm_preds))


@torch.no_grad()
def mean_reverse_kl_on_rollouts(
    student: ModelBundle,
    teacher: ModelBundle,
    vocab_map: torch.Tensor,
    clips: list[Clip],
    cfg: DistillConfig,
    batch_size: int = 4,
) -> float:
    """Reverse KL(student || teacher) on the student's own greedy rollouts --
    the same quantity minimized during on-policy training, used here purely
    as a read-only diagnostic before/after training. Processed in small
    sub-batches for the same memory reason as `transcribe_greedy`."""
    prefix_len = student.prefix_ids.numel()
    total_kl = 0.0
    total_count = 0.0

    for chunk in _chunks(clips, batch_size):
        student_feats = extract_features(student, chunk)
        seqs = sample_rollout(
            student,
            student_feats,
            language=cfg.language,
            task=cfg.task,
            max_new_tokens=cfg.max_new_tokens,
            temperature=0.0,
            do_sample=False,
        )
        teacher_feats = extract_features(teacher, chunk)
        teacher_input_ids = retarget_prefix(seqs[:, :-1], prefix_len, teacher.prefix_ids)

        student_logits = score_sequences(student, student_feats, seqs[:, :-1], grad=False)
        teacher_logits = score_sequences(teacher, teacher_feats, teacher_input_ids, grad=False)

        mask = build_loss_mask(seqs, prefix_len, student.eos_id)
        _, per_token_kl = reverse_kl_loss(student_logits, teacher_logits, vocab_map, mask)
        total_kl += float((per_token_kl * mask).sum())
        total_count += float(mask.sum())

    return total_kl / max(total_count, 1.0)


def evaluate(
    student: ModelBundle,
    teacher: ModelBundle,
    vocab_map: torch.Tensor,
    clips: list[Clip],
    cfg: DistillConfig,
    num_samples: int = 3,
) -> dict:
    """Full before/after eval snapshot: student WER, teacher WER (as a
    ceiling reference), mean reverse KL on student rollouts, and a few
    example transcripts for qualitative inspection."""
    references = [c.reference_text for c in clips]
    eval_batch_size = min(cfg.batch_size, 4)  # eval uses greedy decoding on the larger teacher; keep it small

    student_preds = transcribe_greedy(student, clips, cfg.language, cfg.task, cfg.max_new_tokens, eval_batch_size)
    teacher_preds = transcribe_greedy(teacher, clips, cfg.language, cfg.task, cfg.max_new_tokens, eval_batch_size)

    student_wer = compute_wer(student_preds, references)
    teacher_wer = compute_wer(teacher_preds, references)
    mean_kl = mean_reverse_kl_on_rollouts(student, teacher, vocab_map, clips, cfg, eval_batch_size)

    samples = [
        {
            "clip_id": clips[i].clip_id,
            "reference": references[i],
            "student": student_preds[i],
            "teacher": teacher_preds[i],
        }
        for i in range(min(num_samples, len(clips)))
    ]

    return {
        "student_wer": student_wer,
        "teacher_wer": teacher_wer,
        "mean_reverse_kl": mean_kl,
        "num_eval_clips": len(clips),
        "samples": samples,
        "summary": f"student_wer={student_wer:.4f} teacher_wer={teacher_wer:.4f} mean_KL={mean_kl:.4f}",
    }
