"""Core on-policy distillation logic for Whisper ASR models.

Method (mapped from the LLM on-policy distillation literature onto
autoregressive ASR; see README.md for full citations):

1. Sample a transcript from the *student* at temperature 1.0 -- its own
   distribution, mistakes included (Agarwal et al., 2023, "GKD"; Thinking
   Machines Lab, 2025).
2. Teacher-force the *teacher* over that exact token sequence to get its
   next-token distribution at every position the student actually visited.
3. Loss = mean per-token reverse KL(student || teacher), computed exactly
   over the full shared vocabulary (not a single-sample / top-k estimate),
   since both models are local and fully observable to us.
4. Backprop through a second, gradient-enabled forward pass of the student
   over its own rollout; the teacher is always frozen (`torch.no_grad`).

For the off-policy comparison arm, step 1 instead samples from the *teacher*
(classic pseudo-labelling, as in Distil-Whisper), holding the loss function
(reverse KL) fixed -- this isolates the on-policy-vs-off-policy effect from
the choice of divergence, mirroring the ablation in Agarwal et al. (2023).

Two Whisper-specific subtleties are handled explicitly (see README.md):
  - whisper-large-v3 uses 128 log-mel filterbanks vs. 80 for smaller
    checkpoints, so student and teacher need separate feature extractors
    over the same raw audio.
  - whisper-large-v3 inserts a new `<|yue|>` (Cantonese) language token into
    the vocabulary, which shifts every subsequent special-token id by one.
    Real transcript content tokens are unaffected, but the fixed forced
    prefix (`<|en|><|transcribe|><|notimestamps|>`) is not -- so we align
    vocabularies by token string, not by naive index slicing.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Optional

import torch
import torch.nn.functional as F

from .config import DistillConfig, affordable_steps
from .data import Clip, ClipCycler

# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------


@dataclass
class ModelBundle:
    name: str
    model: "torch.nn.Module"
    feature_extractor: object
    tokenizer: object
    prefix_ids: torch.Tensor  # shape (prefix_len,), this model's own token ids
    eos_id: int
    vocab_size: int
    device: torch.device


def load_model_bundle(
    model_name: str,
    language: str,
    task: str,
    device: torch.device,
    dtype: torch.dtype,
    trainable: bool,
    freeze_encoder: bool = True,
) -> ModelBundle:
    from transformers import (
        WhisperFeatureExtractor,
        WhisperForConditionalGeneration,
        WhisperTokenizerFast,
    )

    model = WhisperForConditionalGeneration.from_pretrained(model_name, dtype=dtype)
    model.to(device)
    # We pass language/task/max_new_tokens explicitly on every generate() call
    # instead of relying on a baked-in generation_config, so behavior is
    # identical and reproducible across checkpoints (and so transformers
    # doesn't warn about a stale `max_length`/`forced_decoder_ids` clashing
    # with the arguments we pass).
    model.generation_config.forced_decoder_ids = None
    model.generation_config.max_length = None
    model.generation_config.suppress_tokens = None
    model.generation_config.begin_suppress_tokens = None

    feature_extractor = WhisperFeatureExtractor.from_pretrained(model_name)
    tokenizer = WhisperTokenizerFast.from_pretrained(model_name)

    prompt = tokenizer.get_decoder_prompt_ids(language=language, task=task, no_timestamps=True)
    sot_id = tokenizer.convert_tokens_to_ids("<|startoftranscript|>")
    prefix_ids = torch.tensor([sot_id] + [tok_id for _, tok_id in prompt], dtype=torch.long)

    if not trainable:
        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)
    else:
        model.train()
        if freeze_encoder:
            for p in model.get_encoder().parameters():
                p.requires_grad_(False)

    return ModelBundle(
        name=model_name,
        model=model,
        feature_extractor=feature_extractor,
        tokenizer=tokenizer,
        prefix_ids=prefix_ids.to(device),
        eos_id=int(tokenizer.eos_token_id),
        vocab_size=len(tokenizer),
        device=device,
    )


def extract_features(bundle: ModelBundle, clips: list[Clip]) -> torch.Tensor:
    arrays = [c.audio for c in clips]
    feats = bundle.feature_extractor(arrays, sampling_rate=16_000, return_tensors="pt")
    return feats.input_features.to(bundle.device, bundle.model.dtype)


# ---------------------------------------------------------------------------
# Cross-vocabulary alignment
# ---------------------------------------------------------------------------

# whisper-tiny/base/small/medium call the "no speech detected" special token
# `<|nocaptions|>`; whisper-large-v3's tokenizer renames the same functional
# slot to `<|nospeech|>`. Same concept, different string label -- treated as
# equivalent when aligning vocabularies by token string.
_TOKEN_ALIASES = {"<|nocaptions|>": "<|nospeech|>", "<|nospeech|>": "<|nocaptions|>"}


def build_vocab_map(teacher: ModelBundle, student: ModelBundle) -> torch.Tensor:
    """Return a LongTensor `m` of shape (student_vocab_size,) such that
    `teacher_logits.index_select(-1, m)` re-expresses the teacher's logits in
    the student's index space, token-for-token (not merely index-for-index).

    Built by matching token *strings*, not ids, precisely because
    whisper-large-v3's extra `<|yue|>` token shifts every subsequent
    special-token id by one relative to smaller Whisper checkpoints.
    """
    student_vocab = student.tokenizer.get_vocab()
    teacher_vocab = teacher.tokenizer.get_vocab()

    mapping = [-1] * len(student_vocab)
    for token_str, student_id in student_vocab.items():
        teacher_id = teacher_vocab.get(token_str)
        if teacher_id is None:
            teacher_id = teacher_vocab.get(_TOKEN_ALIASES.get(token_str, ""))
        if teacher_id is None:
            raise ValueError(
                f"Student token {token_str!r} (id {student_id}) has no counterpart "
                f"in the teacher vocabulary, even after alias lookup. Student "
                f"({student.name}) and teacher ({teacher.name}) tokenizers may not "
                f"be from the same Whisper family."
            )
        mapping[student_id] = teacher_id

    assert all(t >= 0 for t in mapping), "vocab map has unmatched entries"
    assert len(set(mapping)) == len(mapping), "vocab map is not injective"
    return torch.tensor(mapping, dtype=torch.long)


def retarget_prefix(seqs: torch.Tensor, prefix_len: int, new_prefix_ids: torch.Tensor) -> torch.Tensor:
    """Swap in `new_prefix_ids` for the first `prefix_len` positions of
    `seqs`, leaving the generated content (positions >= prefix_len)
    untouched. Needed whenever a sequence generated under one model's
    forced-prefix ids is teacher-forced through the *other* model, since the
    two Whisper checkpoints assign different ids to the same task/timestamp
    prompt tokens (see module docstring), even though real transcript
    content tokens share identical ids across the whole Whisper family.
    """
    out = seqs.clone()
    out[:, :prefix_len] = new_prefix_ids.to(device=seqs.device, dtype=seqs.dtype)
    return out


# ---------------------------------------------------------------------------
# Rollouts and scoring
# ---------------------------------------------------------------------------


@torch.no_grad()
def sample_rollout(
    bundle: ModelBundle,
    input_features: torch.Tensor,
    language: str,
    task: str,
    max_new_tokens: int,
    temperature: float,
    do_sample: bool,
) -> torch.Tensor:
    gen_kwargs = dict(
        input_features=input_features,
        language=language,
        task=task,
        max_new_tokens=max_new_tokens,
        return_dict_in_generate=True,
    )
    if do_sample:
        gen_kwargs.update(do_sample=True, temperature=temperature, top_k=0)
    else:
        gen_kwargs.update(do_sample=False, num_beams=1)
    out = bundle.model.generate(**gen_kwargs)
    return out.sequences


def score_sequences(
    bundle: ModelBundle,
    input_features: torch.Tensor,
    decoder_input_ids: torch.Tensor,
    grad: bool,
) -> torch.Tensor:
    ctx = torch.enable_grad() if grad else torch.no_grad()
    with ctx:
        out = bundle.model(input_features=input_features, decoder_input_ids=decoder_input_ids)
    return out.logits


def build_loss_mask(seqs: torch.Tensor, prefix_len: int, eos_id: int) -> torch.Tensor:
    """Mask (over `seqs[:, 1:]`, i.e. the target positions) that is 1 for
    generated content positions (including the first end-of-transcript token)
    and 0 for the forced prompt prefix and any post-eos padding."""
    targets = seqs[:, 1:]
    is_eos = targets.eq(eos_id)
    cumulative_eos = is_eos.cumsum(dim=1)
    not_padding = cumulative_eos <= 1
    position_idx = torch.arange(targets.size(1), device=seqs.device).unsqueeze(0)
    after_prefix = position_idx >= (prefix_len - 1)
    return (not_padding & after_prefix).float()


def reverse_kl_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    vocab_map: torch.Tensor,
    mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Full-vocabulary reverse KL(student || teacher) at every masked
    position. Returns (scalar loss for backprop, per-position KL for logging).
    """
    teacher_logits = teacher_logits.index_select(-1, vocab_map.to(teacher_logits.device))
    student_logp = F.log_softmax(student_logits.float(), dim=-1)
    teacher_logp = F.log_softmax(teacher_logits.float(), dim=-1)
    student_p = student_logp.exp()
    per_token_kl = (student_p * (student_logp - teacher_logp.detach())).sum(-1)
    denom = mask.sum().clamp_min(1.0)
    loss = (per_token_kl * mask).sum() / denom
    return loss, per_token_kl


# ---------------------------------------------------------------------------
# One training step
# ---------------------------------------------------------------------------


def training_step(
    cfg: DistillConfig,
    student: ModelBundle,
    teacher: ModelBundle,
    vocab_map: torch.Tensor,
    batch: list[Clip],
    optimizer: torch.optim.Optimizer,
) -> float:
    """Run one on-policy or off-policy distillation step; returns the mean
    reverse KL for logging (equal to the loss for on/off policy as defined
    here, since the loss *is* the mean reverse KL)."""
    prefix_len = student.prefix_ids.numel()

    if cfg.policy == "on_policy":
        rollout_bundle, score_bundle = student, teacher
    elif cfg.policy == "off_policy":
        rollout_bundle, score_bundle = teacher, student
    else:
        raise ValueError(f"unknown policy {cfg.policy!r}")

    rollout_feats = extract_features(rollout_bundle, batch)
    seqs = sample_rollout(
        rollout_bundle,
        rollout_feats,
        language=cfg.language,
        task=cfg.task,
        max_new_tokens=cfg.max_new_tokens,
        temperature=cfg.temperature,
        do_sample=True,
    )

    student_feats = extract_features(student, batch)
    teacher_feats = extract_features(teacher, batch)

    student_input_ids = retarget_prefix(seqs[:, :-1], prefix_len, student.prefix_ids)
    teacher_input_ids = retarget_prefix(seqs[:, :-1], prefix_len, teacher.prefix_ids)

    teacher_logits = score_sequences(teacher, teacher_feats, teacher_input_ids, grad=False)
    student_logits = score_sequences(student, student_feats, student_input_ids, grad=True)

    mask = build_loss_mask(seqs, prefix_len, student.eos_id)
    loss, _ = reverse_kl_loss(student_logits, teacher_logits, vocab_map, mask)

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(
        (p for p in student.model.parameters() if p.requires_grad), max_norm=1.0
    )
    optimizer.step()

    return float(loss.detach().cpu())


# ---------------------------------------------------------------------------
# Full arm (train + before/after eval), with the in-run cost guardrail
# ---------------------------------------------------------------------------


@dataclass
class StepLog:
    step: int
    loss: float
    seconds: float


@dataclass
class ArmResult:
    config_name: str
    policy: str
    baseline_eval: dict
    final_eval: dict
    step_log: list[StepLog] = field(default_factory=list)
    steps_run: int = 0
    steps_planned: int = 0
    stopped_early: bool = False
    stop_reason: Optional[str] = None
    wall_seconds: float = 0.0
    est_cost_usd: float = 0.0


def train_arm(
    cfg: DistillConfig,
    student: ModelBundle,
    teacher: ModelBundle,
    vocab_map: torch.Tensor,
    train_clips: list[Clip],
    eval_clips: list[Clip],
    already_spent_usd: float,
    evaluate_fn: Callable,
    on_step: Optional[Callable[[StepLog], None]] = None,
) -> ArmResult:
    from .config import gpu_second_rate

    t_start = time.monotonic()

    print(f"[{cfg.name}] baseline eval ({len(eval_clips)} clips)...", flush=True)
    baseline_eval = evaluate_fn(student, teacher, vocab_map, eval_clips, cfg)
    print(f"[{cfg.name}] baseline: {baseline_eval['summary']}", flush=True)

    optimizer = torch.optim.AdamW(
        (p for p in student.model.parameters() if p.requires_grad),
        lr=cfg.learning_rate,
    )
    cycler = ClipCycler(train_clips, batch_size=cfg.batch_size, seed=cfg.seed)

    step_log: list[StepLog] = []
    stopped_early = False
    stop_reason = None
    planned_steps = cfg.num_steps
    warmup_steps = min(3, cfg.num_steps)

    step = 0
    while step < planned_steps:
        batch = cycler.next_batch()
        t0 = time.monotonic()
        loss_val = training_step(cfg, student, teacher, vocab_map, batch, optimizer)
        dt = time.monotonic() - t0
        step += 1
        log = StepLog(step=step, loss=loss_val, seconds=dt)
        step_log.append(log)
        if on_step is not None:
            on_step(log)
        if step % cfg.log_every == 0 or step == planned_steps:
            print(f"[{cfg.name}] step {step}/{planned_steps} loss(KL)={loss_val:.4f} ({dt:.2f}s)", flush=True)

        # In-run, *measured* cost guardrail: after a short warmup, re-project
        # total spend from actual step latency and shrink the remaining plan
        # (rather than crashing mid-run) if we'd otherwise exceed budget.
        if step == warmup_steps:
            measured_spend_so_far = already_spent_usd + sum(s.seconds for s in step_log) * gpu_second_rate(cfg.gpu_type)
            remaining_budget = cfg.max_cost_usd - measured_spend_so_far
            measured_rate = sum(s.seconds for s in step_log) / len(step_log)
            affordable = affordable_steps(cfg.gpu_type, measured_rate, remaining_budget)
            new_plan = min(planned_steps, step + affordable)
            if new_plan < planned_steps:
                print(
                    f"[{cfg.name}] cost-guardrail: measured {measured_rate:.2f}s/step -> "
                    f"capping plan at {new_plan} steps (was {planned_steps}) to stay within "
                    f"${cfg.max_cost_usd:.2f} budget.",
                    flush=True,
                )
                planned_steps = max(step, new_plan)
                stopped_early = new_plan < cfg.num_steps
                stop_reason = "cost_guardrail" if stopped_early else None

    print(f"[{cfg.name}] final eval ({len(eval_clips)} clips)...", flush=True)
    final_eval = evaluate_fn(student, teacher, vocab_map, eval_clips, cfg)
    print(f"[{cfg.name}] final: {final_eval['summary']}", flush=True)

    wall_seconds = time.monotonic() - t_start
    est_cost_usd = wall_seconds * gpu_second_rate(cfg.gpu_type)

    return ArmResult(
        config_name=cfg.name,
        policy=cfg.policy,
        baseline_eval=baseline_eval,
        final_eval=final_eval,
        step_log=step_log,
        steps_run=len(step_log),
        steps_planned=cfg.num_steps,
        stopped_early=stopped_early,
        stop_reason=stop_reason,
        wall_seconds=wall_seconds,
        est_cost_usd=est_cost_usd,
    )
