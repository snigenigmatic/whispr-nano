"""Offline invariants for the selection pilot (no model or audio downloads)."""

import copy
import math

import pytest

from research.pilot.training import (
    IndexCycler,
    TrainConfig,
    build_decoder_batch,
    canonical_hash,
    minimal_normalize,
    summarize_predictions,
    validate_rows,
    word_counts,
)


def rows():
    return [
        {"utt_id": "tr1", "speaker_id": "S1", "native_language": "Hindi",
         "prompt_id": "a0001", "reference_text": "This is speech.", "audio_path": "S1/a0001.wav",
         "duration_s": 2.25, "split": "train"},
        {"utt_id": "dv1", "speaker_id": "S2", "native_language": "Hindi",
         "prompt_id": "a0002", "reference_text": "This is different speech.", "audio_path": "S2/a0002.wav",
         "duration_s": 3.25, "split": "dev_unseen"},
    ]


def test_forced_prefix_stays_in_history_but_not_in_loss():
    batch = build_decoder_batch([[1, 3, 4, 5, 9, 8, 2]], [1, 3, 4, 5], eos_id=2, pad_id=2)
    assert batch["decoder_input_ids"] == [[1, 3, 4, 5, 9, 8]]
    assert batch["labels"] == [[-100, -100, -100, 9, 8, 2]]
    assert batch["decoder_attention_mask"] == [[1, 1, 1, 1, 1, 1]]


def test_padding_masks_positions_not_shared_eos_token_id():
    batch = build_decoder_batch([[1, 3, 4, 5, 9, 2], [1, 3, 4, 5, 9, 8, 7, 2]],
                                [1, 3, 4, 5], eos_id=2, pad_id=2)
    assert batch["labels"][0] == [-100, -100, -100, 9, 2, -100, -100]
    assert batch["labels"][1] == [-100, -100, -100, 9, 8, 7, 2]
    assert batch["decoder_attention_mask"][0] == [1, 1, 1, 1, 1, 0, 0]


@pytest.mark.parametrize("sequence", [[1, 3, 4, 9, 2], [1, 3, 4, 5, 9], [1, 3, 4, 5, 2, 9, 2]])
def test_invalid_special_token_sequences_fail(sequence):
    with pytest.raises(ValueError):
        build_decoder_batch([sequence], [1, 3, 4, 5], eos_id=2, pad_id=2)


def test_final_test_is_rejected_even_if_it_has_disjoint_speaker():
    manifest = rows()
    manifest[1]["split"] = "test"
    with pytest.raises(ValueError, match="Only train/dev"):
        validate_rows(manifest)


def test_speaker_leakage_is_rejected():
    manifest = rows()
    manifest[1]["speaker_id"] = "S1"
    with pytest.raises(ValueError, match="speakers overlap"):
        validate_rows(manifest)


def test_duplicate_id_is_rejected():
    manifest = rows()
    manifest[1]["utt_id"] = "tr1"
    with pytest.raises(ValueError, match="Duplicate"):
        validate_rows(manifest)


def test_unseen_prompt_leakage_is_rejected_but_shared_diagnostic_is_explicit():
    manifest = rows()
    manifest[1]["prompt_id"] = "a0001"
    with pytest.raises(ValueError, match="prompt IDs overlap"):
        validate_rows(manifest)
    manifest[1]["split"] = "dev_shared"
    train, dev = validate_rows(manifest)
    assert train[0]["prompt_id"] == dev[0]["prompt_id"]


@pytest.mark.parametrize("duration", [0, -1, 30.1, float("nan"), float("inf")])
def test_invalid_duration_cannot_silently_truncate(duration):
    manifest = rows()
    manifest[0]["duration_s"] = duration
    with pytest.raises(ValueError, match="duration_s"):
        validate_rows(manifest)


def test_manifest_hash_is_order_invariant_after_canonical_sort_but_sensitive_to_labels():
    manifest = rows()
    first_train, first_dev = validate_rows(manifest)
    other_train, other_dev = validate_rows(list(reversed(manifest)))
    assert canonical_hash(first_train + first_dev) == canonical_hash(other_train + other_dev)
    changed = copy.deepcopy(first_dev)
    changed[0]["reference_text"] += " changed"
    assert canonical_hash(first_dev) != canonical_hash(changed)


def test_cycler_preserves_every_item_once_per_epoch_even_across_batches():
    cycle = IndexCycler(5, 7)
    stream = cycle.take(3) + cycle.take(3) + cycle.take(9)
    assert len(stream) == 15
    assert all(sorted(stream[offset:offset + 5]) == list(range(5)) for offset in (0, 5, 10))
    assert stream == IndexCycler(5, 7).take(15)
    assert stream != IndexCycler(5, 8).take(15)


@pytest.mark.parametrize("overrides", [{"num_updates": 0}, {"batch_size": 0}, {"eval_every": -1},
                                        {"learning_rate": math.nan}, {"max_new_tokens": 64},
                                        {"model_revision": "main"}])
def test_invalid_or_unregistered_configuration_fails(overrides):
    with pytest.raises(ValueError):
        TrainConfig(**overrides).validate()


def test_word_counts_include_insertions_and_empty_hypotheses():
    assert word_counts("one two three", "one four three extra") == {
        "errors": 2, "substitutions": 1, "deletions": 0, "insertions": 1, "reference_words": 3}
    assert word_counts("one two", "")["deletions"] == 2
    assert word_counts("", "extra")["insertions"] == 1


def test_macro_wer_does_not_accidentally_weight_by_group_size():
    predictions = []
    for uid, language, reference, hypothesis in [("a", "Hindi", "one two three four", "one two three four"),
                                                ("b", "Arabic", "one", "wrong")]:
        predictions.append({"utt_id": uid, "native_language": language, "speaker_id": uid, "split": "dev",
                            "minimal": word_counts(reference, hypothesis), "english": word_counts(reference, hypothesis)})
    summary = summarize_predictions(predictions)["english"]
    assert summary["micro_wer"] == 0.2
    assert summary["macro_language_wer"] == 0.5
    assert summary["worst_language_wer"] == 1.0


def test_minimal_normalizer_keeps_digits_and_removes_unicode_punctuation():
    assert minimal_normalize("“Hello”—2026!") == "hello 2026"


def test_real_tiny_architecture_forwards_unmasked_prefix_and_backprops_encoder_decoder():
    """Optional torch check; builds a tiny random architecture, never downloads.

    This verifies label handling against the actual Transformers loss path,
    not merely an implementation-shaped mock. It runs in the GPU runtime's CPU
    environment too; local environments without torch explicitly skip it.
    """
    torch = pytest.importorskip("torch")
    from transformers import WhisperConfig, WhisperForConditionalGeneration

    torch.manual_seed(17)
    config = WhisperConfig(vocab_size=20, num_mel_bins=4, d_model=8, encoder_layers=1,
                           decoder_layers=1, encoder_attention_heads=2, decoder_attention_heads=2,
                           encoder_ffn_dim=16, decoder_ffn_dim=16, max_source_positions=4,
                           max_target_positions=16, pad_token_id=2, bos_token_id=2,
                           eos_token_id=2, decoder_start_token_id=1, suppress_tokens=[],
                           begin_suppress_tokens=[])
    config._attn_implementation = "eager"
    model = WhisperForConditionalGeneration(config)
    batch = build_decoder_batch([[1, 3, 4, 5, 9, 2], [1, 3, 4, 5, 8, 9, 2]], [1, 3, 4, 5], 2, 2)
    tensors = {key: torch.tensor(value) for key, value in batch.items()}
    outputs = model(input_features=torch.randn(2, 4, 8), **tensors, use_cache=False)
    expected = torch.nn.functional.cross_entropy(outputs.logits.reshape(-1, 20), tensors["labels"].reshape(-1), ignore_index=-100)
    torch.testing.assert_close(outputs.loss, expected)
    outputs.loss.backward()
    assert model.get_encoder().conv1.weight.grad.abs().sum() > 0
    assert model.model.decoder.layers[0].self_attn.q_proj.weight.grad.abs().sum() > 0
