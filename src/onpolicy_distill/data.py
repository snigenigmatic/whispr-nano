"""LibriSpeech data loading for the on-policy distillation PoC.

We use the small `hf-internal-testing/librispeech_asr_dummy` split -- 73 real
LibriSpeech `clean`/`validation` clips (Panayotov et al., 2015), ~9 MB total.
It is the same split used in Hugging Face's official Whisper
fine-tuning/distillation tutorials: no auth, no gated download, no streaming
flakiness, and more than enough audio to demonstrate the distillation
mechanics inside a sub-$1 GPU budget. See PROPOSAL.md for the plan to scale
this up to the full 960h LibriSpeech training set with real credits.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Clip:
    clip_id: str
    audio: np.ndarray  # mono float32 waveform at `sampling_rate` Hz
    sampling_rate: int
    reference_text: str


def _decode_audio(audio_value) -> tuple[np.ndarray, int]:
    """Handle both the modern torchcodec-backed `Audio` feature and the
    legacy dict-of-array format across `datasets` versions."""
    if hasattr(audio_value, "get_all_samples"):
        samples = audio_value.get_all_samples()
        array = samples.data.mean(dim=0).numpy().astype(np.float32)
        return array, int(samples.sample_rate)
    array = np.asarray(audio_value["array"], dtype=np.float32)
    return array, int(audio_value["sampling_rate"])


def load_clips(
    dataset_id: str,
    dataset_config: str,
    dataset_split: str,
    num_train: int,
    num_eval: int,
    seed: int = 0,
) -> tuple[list[Clip], list[Clip]]:
    """Load a small ASR dataset and deterministically split it into a
    training pool and a disjoint held-out eval pool."""
    from datasets import Audio, load_dataset

    ds = load_dataset(dataset_id, dataset_config, split=dataset_split)
    ds = ds.cast_column("audio", Audio(sampling_rate=16_000))

    total_needed = num_train + num_eval
    if total_needed > len(ds):
        raise ValueError(
            f"Requested {total_needed} clips (train={num_train} + eval={num_eval}) "
            f"but {dataset_id}/{dataset_config}/{dataset_split} only has {len(ds)} rows."
        )

    rng = np.random.default_rng(seed)
    order = rng.permutation(len(ds))

    def to_clip(i: int) -> Clip:
        row = ds[int(i)]
        array, sr = _decode_audio(row["audio"])
        text = (row.get("text") or row.get("sentence") or "").strip()
        return Clip(
            clip_id=str(row.get("id", i)),
            audio=array,
            sampling_rate=sr,
            reference_text=text,
        )

    train_idx = order[:num_train]
    eval_idx = order[num_train:total_needed]
    return [to_clip(i) for i in train_idx], [to_clip(i) for i in eval_idx]


class ClipCycler:
    """Cycles through a fixed clip pool to build training batches, reshuffling
    (without replacement, within an epoch) each time the pool is exhausted."""

    def __init__(self, clips: list[Clip], batch_size: int, seed: int = 0):
        if not clips:
            raise ValueError("clip pool is empty")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        self.clips = clips
        self.batch_size = batch_size
        self._rng = np.random.default_rng(seed)
        self._order: list[int] = []

    def _refill(self) -> None:
        self._order = list(self._rng.permutation(len(self.clips)))

    def next_batch(self) -> list[Clip]:
        batch = []
        while len(batch) < self.batch_size:
            if not self._order:
                self._refill()
            batch.append(self.clips[self._order.pop()])
        return batch
