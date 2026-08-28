"""Unit tests for the distillation math on synthetic tensors -- no model
download, no network, fast enough to run on every change."""

import torch

from onpolicy_distill.distill import build_loss_mask, retarget_prefix, reverse_kl_loss


def test_build_loss_mask_masks_prefix_and_post_eos_padding():
    eos = 99
    # [sot, lang, task, notime, content_a, content_b, eos, eos(pad)]
    seqs = torch.tensor([[0, 1, 2, 3, 10, 11, eos, eos]])
    mask = build_loss_mask(seqs, prefix_len=4, eos_id=eos)
    # targets = seqs[:, 1:] = [lang, task, notime, content_a, content_b, eos, eos]
    # prefix occupies target positions 0,1,2 (predicting lang/task/notime) -> masked out
    # content_a, content_b, first eos -> kept; second eos (padding) -> masked out
    expected = torch.tensor([[0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0]])
    assert torch.equal(mask, expected)


def test_build_loss_mask_no_eos_scores_everything_after_prefix():
    seqs = torch.tensor([[0, 1, 2, 3, 10, 11, 12]])
    mask = build_loss_mask(seqs, prefix_len=4, eos_id=999)
    expected = torch.tensor([[0.0, 0.0, 0.0, 1.0, 1.0, 1.0]])
    assert torch.equal(mask, expected)


def test_retarget_prefix_only_touches_prefix_positions():
    seqs = torch.tensor([[1, 2, 3, 4, 5, 6, 7]])
    new_prefix = torch.tensor([10, 20, 30, 40])
    out = retarget_prefix(seqs, prefix_len=4, new_prefix_ids=new_prefix)
    assert torch.equal(out[:, :4], new_prefix.unsqueeze(0))
    assert torch.equal(out[:, 4:], seqs[:, 4:])


def test_reverse_kl_zero_when_student_equals_teacher():
    torch.manual_seed(0)
    logits = torch.randn(2, 5, 8)
    vocab_map = torch.arange(8)
    mask = torch.ones(2, 5)
    loss, per_token = reverse_kl_loss(logits, logits, vocab_map, mask)
    assert torch.allclose(loss, torch.zeros_like(loss), atol=1e-5)
    assert torch.allclose(per_token, torch.zeros_like(per_token), atol=1e-5)


def test_reverse_kl_positive_and_finite_when_distributions_differ():
    torch.manual_seed(0)
    student_logits = torch.randn(2, 5, 8)
    teacher_logits = torch.randn(2, 5, 8)
    vocab_map = torch.arange(8)
    mask = torch.ones(2, 5)
    loss, per_token = reverse_kl_loss(student_logits, teacher_logits, vocab_map, mask)
    assert loss.item() > 0
    assert torch.isfinite(loss)
    assert torch.isfinite(per_token).all()


def test_reverse_kl_respects_mask():
    torch.manual_seed(1)
    student_logits = torch.randn(1, 3, 6)
    teacher_logits = torch.randn(1, 3, 6)
    vocab_map = torch.arange(6)
    mask_all = torch.ones(1, 3)
    mask_first_only = torch.tensor([[1.0, 0.0, 0.0]])
    loss_all, per_token = reverse_kl_loss(student_logits, teacher_logits, vocab_map, mask_all)
    loss_first, _ = reverse_kl_loss(student_logits, teacher_logits, vocab_map, mask_first_only)
    assert torch.allclose(loss_first, per_token[:, 0].mean(), atol=1e-5)
    assert not torch.allclose(loss_all, loss_first)


def test_reverse_kl_gradient_flows_only_through_student():
    torch.manual_seed(2)
    student_logits = torch.randn(1, 2, 4, requires_grad=True)
    teacher_logits = torch.randn(1, 2, 4)  # simulates a no_grad teacher forward
    vocab_map = torch.arange(4)
    mask = torch.ones(1, 2)
    loss, _ = reverse_kl_loss(student_logits, teacher_logits, vocab_map, mask)
    loss.backward()
    assert student_logits.grad is not None
    assert torch.isfinite(student_logits.grad).all()


def test_vocab_map_reindexes_teacher_logits_to_student_space():
    # Teacher has 5 tokens; student's 4 tokens correspond to teacher ids [0,1,3,4]
    # (as if teacher token id 2 were the "extra" special token, like <|yue|>).
    vocab_map = torch.tensor([0, 1, 3, 4])
    teacher_logits = torch.tensor([[[10.0, 20.0, 999.0, 30.0, 40.0]]])
    realigned = teacher_logits.index_select(-1, vocab_map)
    assert torch.equal(realigned, torch.tensor([[[10.0, 20.0, 30.0, 40.0]]]))
