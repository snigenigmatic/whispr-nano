"""Supervised selection pilot; no teacher targets and no final-test evaluation.

The public entry point is ``train(config, manifest_rows, audio_root, output_dir)``.
Rows use utt_id, speaker_id, native_language, prompt_id, reference_text,
audio_path, duration_s and split. Include the externally selected training rows
and the complete frozen development rows; final-test rows are rejected. Heavy
packages are imported only by train(), so the control logic is CPU-checkable.

Whisper's language/task/no-timestamp prefix is supplied as decoder history. Its
forced tokens are excluded from CE; every reference content token and the first
EOS is supervised. Padding is masked by position, never by token identity, since
Whisper's pad token is also EOS. The encoder and decoder are both trainable.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import random
import time
import unicodedata
from collections import Counter
from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence


MODEL_ID = "openai/whisper-tiny"
MODEL_REVISION = "169d4a4341b33bc18d8881c4b69c2e104e1cc0af"


@dataclass(frozen=True)
class TrainConfig:
    arm_name: str = "supervised_selection_pilot"
    seed: int = 0
    num_updates: int = 200
    batch_size: int = 4
    grad_accum_steps: int = 1
    learning_rate: float = 1e-5
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    # Zero means baseline and final only. No checkpoint selection uses test data.
    eval_every: int = 0
    eval_batch_size: int = 4
    max_new_tokens: int = 224
    device: str = "auto"
    feature_cache_dir: str | None = None
    normalizer_path: str | None = None
    expected_initial_state_sha256: str | None = None
    expected_dev_manifest_sha256: str | None = None
    model_id: str = MODEL_ID
    model_revision: str = MODEL_REVISION

    def validate(self) -> None:
        for name in ("num_updates", "batch_size", "grad_accum_steps", "eval_batch_size"):
            if not isinstance(getattr(self, name), int) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.eval_every < 0 or not isinstance(self.eval_every, int):
            raise ValueError("eval_every must be a non-negative integer")
        for name in ("learning_rate", "max_grad_norm"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive and finite")
        if not math.isfinite(self.weight_decay) or self.weight_decay < 0:
            raise ValueError("weight_decay must be non-negative and finite")
        if (self.model_id, self.model_revision) != (MODEL_ID, MODEL_REVISION):
            raise ValueError("This pilot fixes the model and processor to the registered tiny SHA")
        if self.max_new_tokens != 224:
            raise ValueError("The frozen greedy decoder uses max_new_tokens=224")
        if not isinstance(self.seed, int) or self.seed < 0:
            raise ValueError("seed must be a non-negative integer")


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _state_hash(state: Mapping[str, Any]) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        digest.update(name.encode() + b"\0")
        digest.update(str(tensor.dtype).encode() + b"\0")
        digest.update(json.dumps(list(tensor.shape)).encode() + b"\0")
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def validate_rows(rows: Sequence[Mapping[str, Any]]) -> tuple[list[dict], list[dict]]:
    """Fail closed on test access, duplicate IDs, and speaker leakage."""
    required = ("utt_id", "speaker_id", "native_language", "prompt_id", "reference_text", "audio_path", "split")
    train_rows, dev_rows, seen = [], [], set()
    for source in rows:
        row = dict(source)
        for field in required:
            if not isinstance(row.get(field), str) or not row[field].strip():
                raise ValueError(f"Missing or empty string {field}: {row.get('utt_id')}")
        uid, split = row["utt_id"], row["split"]
        if uid in seen:
            raise ValueError(f"Duplicate utterance ID: {uid}")
        seen.add(uid)
        if split != "train" and split != "dev" and not split.startswith("dev_"):
            raise ValueError(f"Only train/dev rows may reach this pilot: {uid} split={split}")
        duration = float(row.get("duration_s", 0))
        if not math.isfinite(duration) or duration <= 0 or duration > 30.0:
            raise ValueError(f"duration_s must be in (0, 30], without audio truncation: {uid}")
        row["duration_s"] = duration
        (train_rows if split == "train" else dev_rows).append(row)
    if not train_rows or not dev_rows:
        raise ValueError("Both selected training rows and frozen development rows are required")
    overlap = {r["speaker_id"] for r in train_rows} & {r["speaker_id"] for r in dev_rows}
    if overlap:
        raise ValueError(f"Train/dev speakers overlap: {sorted(overlap)}")
    train_prompts = {r["prompt_id"] for r in train_rows}
    if any(r["split"] == "dev_unseen" and r["prompt_id"] in train_prompts for r in dev_rows):
        raise ValueError("dev_unseen prompt IDs overlap training")
    return sorted(train_rows, key=lambda r: r["utt_id"]), sorted(dev_rows, key=lambda r: r["utt_id"])


def build_decoder_batch(
    token_sequences: Sequence[Sequence[int]], prefix_ids: Sequence[int], eos_id: int, pad_id: int,
) -> dict[str, list[list[int]]]:
    """Shift the unmasked sequence; retain prefix history and supervise true EOS."""
    if not token_sequences or not prefix_ids:
        raise ValueError("Nonempty sequences and a forced decoder prefix are required")
    prefix = list(prefix_ids)
    seqs = [list(seq) for seq in token_sequences]
    for seq in seqs:
        if seq[:len(prefix)] != prefix or len(seq) <= len(prefix):
            raise ValueError("Token sequence does not start with the complete registered prefix")
        if seq[-1] != eos_id or eos_id in seq[len(prefix):-1]:
            raise ValueError("Reference must have exactly one final EOS after its prefix")
    width = max(len(seq) - 1 for seq in seqs)
    result = {"decoder_input_ids": [], "decoder_attention_mask": [], "labels": []}
    for seq in seqs:
        length = len(seq) - 1
        labels = [-100] * (len(prefix) - 1) + seq[len(prefix):]
        result["decoder_input_ids"].append(seq[:-1] + [pad_id] * (width - length))
        result["decoder_attention_mask"].append([1] * length + [0] * (width - length))
        result["labels"].append(labels + [-100] * (width - length))
    return result


class IndexCycler:
    """Exactly fixed exposures, without replacement within each shuffled epoch."""

    def __init__(self, size: int, seed: int):
        if size <= 0:
            raise ValueError("Training pool must be nonempty")
        self.size = size
        self.rng = random.Random(seed)
        self.pending: list[int] = []

    def take(self, count: int) -> list[int]:
        if count <= 0:
            raise ValueError("Batch size must be positive")
        result = []
        while len(result) < count:
            if not self.pending:
                self.pending = list(range(self.size))
                self.rng.shuffle(self.pending)
            result.append(self.pending.pop())
        return result


def minimal_normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).lower()
    return " ".join("".join(" " if unicodedata.category(c).startswith("P") else c for c in text).split())


def word_counts(reference: str, hypothesis: str) -> dict[str, int]:
    """Levenshtein S/D/I counts; deterministic substitution/deletion/insertion ties."""
    ref, hyp = reference.split(), hypothesis.split()
    previous = [(j, 0, 0, j) for j in range(len(hyp) + 1)]
    for i, ref_word in enumerate(ref, 1):
        current = [(i, 0, i, 0)]
        for j, hyp_word in enumerate(hyp, 1):
            if ref_word == hyp_word:
                current.append(previous[j - 1])
            else:
                sub, delete, insert = previous[j - 1], previous[j], current[j - 1]
                current.append(min((
                    (sub[0] + 1, sub[1] + 1, sub[2], sub[3]),
                    (delete[0] + 1, delete[1], delete[2] + 1, delete[3]),
                    (insert[0] + 1, insert[1], insert[2], insert[3] + 1),
                ), key=lambda x: x[0]))
        previous = current
    errors, substitutions, deletions, insertions = previous[-1]
    return dict(errors=errors, substitutions=substitutions, deletions=deletions,
                insertions=insertions, reference_words=len(ref))


def summarize_predictions(predictions: list[dict]) -> dict:
    output = {}
    for normalization in ("minimal", "english"):
        buckets: dict[str, list[dict]] = {"overall": predictions}
        for row in predictions:
            buckets.setdefault("language:" + row["native_language"], []).append(row)
            buckets.setdefault("split:" + row["split"], []).append(row)
        summaries = {}
        for key, group in sorted(buckets.items()):
            errors = sum(row[normalization]["errors"] for row in group)
            words = sum(row[normalization]["reference_words"] for row in group)
            summaries[key] = {"wer": errors / words if words else None, "errors": errors,
                              "reference_words": words, "utterances": len(group),
                              "speakers": len({r["speaker_id"] for r in group})}
        group_wers = [v["wer"] for k, v in summaries.items() if k.startswith("language:") and v["wer"] is not None]
        output[normalization] = {"micro_wer": summaries["overall"]["wer"],
                                 "macro_language_wer": sum(group_wers) / len(group_wers),
                                 "worst_language_wer": max(group_wers), "groups": summaries}
    return output


def _write_json(path: Path, data: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n")
    temporary.replace(path)


def train(config: TrainConfig | Mapping[str, Any], manifest_rows: Sequence[Mapping[str, Any]],
          audio_root: str | Path, output_dir: str | Path,
          verified_feature_receipts: Mapping[str, Mapping[str, Any]] | None = None) -> dict:
    """Train one fixed-selection supervised arm and return a JSON-safe summary.

    Outputs include the frozen manifest/config, initial model state hash, all
    update exposures and losses, baseline/final dev predictions, dev metrics,
    a standard Hugging Face final checkpoint and optimizer state. No final-test
    row or teacher transcript enters the training objective.
    """
    cfg = config if isinstance(config, TrainConfig) else TrainConfig(**dict(config))
    cfg.validate()
    train_rows, dev_rows = validate_rows(manifest_rows)
    dev_hash = canonical_hash(dev_rows)
    if cfg.expected_dev_manifest_sha256 and dev_hash != cfg.expected_dev_manifest_sha256:
        raise ValueError("Frozen development manifest hash mismatch")
    root, out = Path(audio_root).resolve(), Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()):
        raise ValueError(f"Output directory must be empty to prevent accidental result replacement: {out}")

    # Set before importing torch or initializing CUDA. Exact repeatability is
    # only claimed for identical software/hardware, recorded in the output.
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    import numpy as np
    import scipy.signal
    import soundfile as sf
    import torch
    import transformers
    from huggingface_hub import hf_hub_download
    from transformers import WhisperForConditionalGeneration, WhisperProcessor
    from research.pilot.frozen_english_normalizer import EnglishTextNormalizer
    import inspect
    transformers.utils.logging.set_verbosity_error()

    start = time.monotonic()
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device("cuda" if cfg.device == "auto" and torch.cuda.is_available() else
                          "cpu" if cfg.device == "auto" else cfg.device)
    model = WhisperForConditionalGeneration.from_pretrained(
        cfg.model_id, revision=cfg.model_revision, torch_dtype=torch.float32,
        attn_implementation="eager",
    )
    # Do not clear suppress_tokens/begin_suppress_tokens or other model defaults.
    processor = WhisperProcessor.from_pretrained(cfg.model_id, revision=cfg.model_revision,
                                                 language="english", task="transcribe")
    processor.tokenizer.set_prefix_tokens(language="english", task="transcribe", predict_timestamps=False)
    prefix = list(processor.tokenizer.prefix_tokens)
    if prefix[0] != model.config.decoder_start_token_id:
        raise ValueError("Tokenizer/model decoder-start mismatch")
    if model.generation_config.suppress_tokens is None or model.generation_config.begin_suppress_tokens is None:
        raise ValueError("Registered Whisper suppression defaults unexpectedly absent")
    norm_path = Path(cfg.normalizer_path) if cfg.normalizer_path else Path(
        hf_hub_download(cfg.model_id, "normalizer.json", revision=cfg.model_revision))
    english_normalize = EnglishTextNormalizer(json.loads(norm_path.read_text()))
    for row in train_rows + dev_rows:
        if not minimal_normalize(row["reference_text"]) or not english_normalize(row["reference_text"]):
            raise ValueError(f"Empty normalized reference: {row['utt_id']}")

    initial_hash = _state_hash(model.state_dict())
    initial_encoder = {name: tensor.detach().cpu().clone() for name, tensor in model.get_encoder().state_dict().items()}
    initial_encoder_hash = _state_hash(initial_encoder)
    if cfg.expected_initial_state_sha256 and cfg.expected_initial_state_sha256 != initial_hash:
        raise ValueError("Initial model weights differ from the registered comparator")
    for parameter in model.parameters():
        parameter.requires_grad_(True)
    model.to(device)
    parameters = list(model.parameters())
    optimizer = torch.optim.AdamW(parameters, lr=cfg.learning_rate, weight_decay=cfg.weight_decay,
                                  betas=(0.9, 0.999), eps=1e-8, foreach=False)
    token_sequences = {r["utt_id"]: processor.tokenizer(r["reference_text"], add_special_tokens=True).input_ids
                       for r in train_rows}
    for uid, sequence in token_sequences.items():
        if len(sequence) - 1 > model.config.max_target_positions:
            raise ValueError(f"Reference exceeds decoder context; truncation prohibited: {uid}")
        build_decoder_batch([sequence], prefix, processor.tokenizer.eos_token_id, processor.tokenizer.pad_token_id)

    cache = Path(cfg.feature_cache_dir) if cfg.feature_cache_dir else out / "feature_cache"
    cache.mkdir(parents=True, exist_ok=True)
    receipts = dict(verified_feature_receipts or {})
    if receipts and set(receipts) != {row["utt_id"] for row in train_rows + dev_rows}:
        raise ValueError("Verified feature receipts must cover the exact train/dev manifest")
    audio_info: dict[str, dict] = {}
    feature_paths: dict[str, Path] = {}
    feature_signature = canonical_hash({"revision": cfg.model_revision, "transformers": transformers.__version__,
                                        "numpy": np.__version__, "scipy": scipy.__version__, "soundfile": sf.__version__,
                                        "downmix": "arithmetic mean", "resampling": "scipy.signal.resample_poly to 16000",
                                        "feature_extractor": processor.feature_extractor.to_dict()})

    def features(row: dict):
        uid = row["utt_id"]
        if uid not in feature_paths:
            if receipts:
                receipt = dict(receipts[uid])
                required_receipt = {
                    "audio_sha256", "duration_s", "original_sample_rate", "channels",
                    "feature_file", "feature_sha256",
                }
                if not required_receipt.issubset(receipt):
                    raise ValueError(f"Incomplete verified feature receipt: {uid}")
                expected = row.get("audio_sha256")
                if expected and expected != receipt["audio_sha256"]:
                    raise ValueError(f"Verified audio checksum mismatch: {uid}")
                if abs(float(receipt["duration_s"]) - row["duration_s"]) > 0.02:
                    raise ValueError(f"Verified audio duration mismatch: {uid}")
                feature_path = cache / str(receipt["feature_file"])
                if not feature_path.is_file() or sha256_file(feature_path) != receipt["feature_sha256"]:
                    raise ValueError(f"Verified feature checksum mismatch: {uid}")
                feature = np.load(feature_path, mmap_mode="r", allow_pickle=False)
                if feature.shape != (80, 3000) or feature.dtype != np.float32 or not np.isfinite(feature).all():
                    raise ValueError(f"Invalid verified feature tensor: {uid}")
                audio_info[uid] = {
                    "audio_sha256": receipt["audio_sha256"],
                    "duration_s": float(receipt["duration_s"]),
                    "original_sample_rate": int(receipt["original_sample_rate"]),
                    "channels": int(receipt["channels"]),
                }
                feature_paths[uid] = feature_path
                return feature
            path = (root / row["audio_path"]).resolve()
            if not path.is_relative_to(root):
                raise ValueError(f"Audio path escapes audio_root: {uid}")
            checksum = sha256_file(path)
            expected = row.get("audio_sha256")
            if expected and checksum != expected:
                raise ValueError(f"Audio checksum mismatch: {uid}")
            audio, sample_rate = sf.read(path, dtype="float32", always_2d=True)
            duration = len(audio) / sample_rate
            if not len(audio) or duration > 30.0 or not np.isfinite(audio).all():
                raise ValueError(f"Invalid or >30s audio (no truncation allowed): {uid}")
            if abs(duration - row["duration_s"]) > max(0.01, 1 / sample_rate):
                raise ValueError(f"Manifest/audio duration mismatch: {uid}")
            audio_info[uid] = {"audio_sha256": checksum, "duration_s": duration,
                               "original_sample_rate": sample_rate, "channels": audio.shape[1]}
            feature_path = cache / (canonical_hash([checksum, feature_signature]) + ".npy")
            if not feature_path.exists():
                audio = audio.mean(axis=1)
                if sample_rate != 16000:
                    divisor = math.gcd(sample_rate, 16000)
                    audio = scipy.signal.resample_poly(audio, 16000 // divisor, sample_rate // divisor).astype(np.float32)
                extracted = processor.feature_extractor(audio, sampling_rate=16000, return_tensors="np").input_features[0]
                temporary = feature_path.with_name(feature_path.name + f".{os.getpid()}.tmp")
                with temporary.open("wb") as stream:
                    np.save(stream, extracted.astype(np.float32), allow_pickle=False)
                temporary.replace(feature_path)
            feature_paths[uid] = feature_path
        return np.load(feature_paths[uid], mmap_mode="r", allow_pickle=False)

    generation_kwargs = {"language": "en", "task": "transcribe", "return_timestamps": False,
                         "max_new_tokens": cfg.max_new_tokens, "do_sample": False,
                         "num_beams": 1, "use_cache": True}
    provenance = {"config": asdict(cfg), "train_manifest_sha256": canonical_hash(train_rows),
                  "dev_manifest_sha256": dev_hash, "initial_state_sha256": initial_hash,
                  "initial_encoder_sha256": initial_encoder_hash,
                  "normalizer_sha256": sha256_file(norm_path), "feature_signature": feature_signature,
                  "normalizer_implementation_sha256": sha256_file(Path(inspect.getfile(EnglishTextNormalizer))),
                  "training_code_sha256": sha256_file(Path(__file__)),
                  "normalizer_source": cfg.normalizer_path or f"{cfg.model_id}@{cfg.model_revision}/normalizer.json",
                  "verified_feature_receipts_sha256": canonical_hash(receipts) if receipts else None,
                  "decoder_prefix_ids": prefix, "generation_kwargs": generation_kwargs,
                  "generation_config": model.generation_config.to_dict(),
                  "trainable_parameters": sum(p.numel() for p in parameters if p.requires_grad),
                  "encoder_trainable_parameters": sum(p.numel() for p in model.get_encoder().parameters() if p.requires_grad),
                  "runtime": {"torch": torch.__version__, "transformers": transformers.__version__,
                              "numpy": np.__version__, "scipy": scipy.__version__, "soundfile": sf.__version__,
                              "cuda": torch.version.cuda, "device": str(device), "dtype": "float32",
                              "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu"},
                  "optimizer": "AdamW (all encoder and decoder parameters); constant learning rate",
                  "loss": "reference content plus EOS CE; forced prefix and padding excluded",
                  "claim_scope": "development-only controlled selection pilot; no final-test results"}
    _write_json(out / "provenance.json", provenance)
    _write_json(out / "manifest.json", train_rows + dev_rows)

    history: list[dict] = []

    def record(event: dict):
        history.append(event)
        with (out / "history.jsonl").open("a") as stream:
            stream.write(json.dumps(event, sort_keys=True, allow_nan=False) + "\n")

    def evaluate(update: int, label: str):
        was_training = model.training
        model.eval()
        predictions = []
        try:
            with torch.inference_mode():
                for offset in range(0, len(dev_rows), cfg.eval_batch_size):
                    chunk = dev_rows[offset:offset + cfg.eval_batch_size]
                    inputs = torch.from_numpy(np.stack([features(row) for row in chunk])).to(device)
                    generated = model.generate(input_features=inputs, **generation_kwargs)
                    hypotheses = processor.batch_decode(generated, skip_special_tokens=True,
                                                        clean_up_tokenization_spaces=False)
                    for row, hypothesis, sequence in zip(chunk, hypotheses, generated.cpu().tolist()):
                        ref = row["reference_text"]
                        hypothesis = hypothesis.strip()
                        predictions.append({**{key: row[key] for key in ("utt_id", "speaker_id", "native_language", "prompt_id", "split")},
                            "reference_text": ref, "prediction": hypothesis, "update": update,
                            "generated_token_ids": sequence,
                            "generation_reached_token_limit": processor.tokenizer.eos_token_id not in sequence[len(prefix):],
                            "minimal": word_counts(minimal_normalize(ref), minimal_normalize(hypothesis)),
                            "english": word_counts(english_normalize(ref), english_normalize(hypothesis))})
        finally:
            model.train(was_training)
        path = out / f"{label}_dev_predictions.jsonl"
        with path.open("w") as stream:
            for prediction in predictions:
                stream.write(json.dumps(prediction, ensure_ascii=False, sort_keys=True) + "\n")
        metrics = summarize_predictions(predictions)
        record({"event": "dev_eval", "update": update, "label": label, "metrics": metrics,
                "elapsed_s": time.monotonic() - start, "prediction_file": path.name})
        _write_json(out / f"{label}_dev_metrics.json", metrics)
        return metrics

    baseline_metrics = evaluate(0, "baseline")
    # Evaluation must not perturb the training RNG stream, even if an upstream
    # generation implementation starts consuming randomness in a future version.
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)
    cycler = IndexCycler(len(train_rows), cfg.seed)
    exposures: Counter[str] = Counter()
    total_tokens, total_audio_s = 0, 0.0
    model.train()
    for update in range(1, cfg.num_updates + 1):
        update_start = time.monotonic()
        optimizer.zero_grad(set_to_none=True)
        batch_rows = [[train_rows[i] for i in cycler.take(cfg.batch_size)] for _ in range(cfg.grad_accum_steps)]
        # Weight each microbatch by supervised token count, so accumulation
        # computes the same token mean as one concatenated effective batch.
        batches = [build_decoder_batch([token_sequences[r["utt_id"]] for r in rows], prefix,
                                       processor.tokenizer.eos_token_id, processor.tokenizer.pad_token_id)
                   for rows in batch_rows]
        counts = [sum(token != -100 for labels in batch["labels"] for token in labels) for batch in batches]
        update_tokens = sum(counts)
        update_loss = 0.0
        update_ids, update_audio_s = [], 0.0
        for rows, batch, token_count in zip(batch_rows, batches, counts):
            inputs = torch.from_numpy(np.stack([features(row) for row in rows])).to(device)
            tensors = {key: torch.tensor(value, dtype=torch.long, device=device) for key, value in batch.items()}
            result = model(input_features=inputs, **tensors, use_cache=False)
            if not torch.isfinite(result.loss):
                raise FloatingPointError(f"Non-finite CE at update {update}")
            weight = token_count / update_tokens
            (result.loss * weight).backward()
            update_loss += float(result.loss.detach()) * weight
            for row in rows:
                uid = row["utt_id"]
                exposures[uid] += 1
                update_ids.append(uid)
                update_audio_s += audio_info[uid]["duration_s"]
        grad_norm = torch.nn.utils.clip_grad_norm_(parameters, cfg.max_grad_norm, error_if_nonfinite=True, foreach=False)
        optimizer.step()
        total_tokens += update_tokens
        total_audio_s += update_audio_s
        record({"event": "train_update", "update": update, "loss": update_loss,
                "pre_clip_grad_norm": float(grad_norm), "learning_rate": cfg.learning_rate,
                "supervised_tokens": update_tokens, "audio_seconds": update_audio_s,
                "utterance_ids": update_ids, "unique_seen": len(exposures),
                "update_seconds": time.monotonic() - update_start, "elapsed_s": time.monotonic() - start})
        if cfg.eval_every and update % cfg.eval_every == 0 and update != cfg.num_updates:
            # Save and restore RNG state as well as train/eval mode.
            rng = (random.getstate(), np.random.get_state(), torch.get_rng_state(),
                   torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None)
            evaluate(update, f"update_{update:06d}")
            random.setstate(rng[0])
            np.random.set_state(rng[1])
            torch.set_rng_state(rng[2])
            if rng[3] is not None:
                torch.cuda.set_rng_state_all(rng[3])

    final_metrics = evaluate(cfg.num_updates, "final")
    checkpoint = out / "checkpoint_last"
    model.save_pretrained(checkpoint, safe_serialization=True)
    processor.save_pretrained(checkpoint)
    torch.save({"optimizer": optimizer.state_dict(), "update": cfg.num_updates,
                "torch_rng_state": torch.get_rng_state(), "config": asdict(cfg)}, checkpoint / "training_state.pt")
    final_encoder = {name: tensor.detach().cpu() for name, tensor in model.get_encoder().state_dict().items()}
    final_encoder_hash = _state_hash(final_encoder)
    encoder_delta_sq, encoder_initial_sq = 0.0, 0.0
    for name, tensor in final_encoder.items():
        delta = tensor.double() - initial_encoder[name].double()
        encoder_delta_sq += float(delta.square().sum())
        encoder_initial_sq += float(initial_encoder[name].double().square().sum())
    if final_encoder_hash == initial_encoder_hash or encoder_delta_sq <= 0:
        raise RuntimeError("Encoder weights did not change; pilot cannot establish acoustic adaptation")
    summary = {**provenance, "baseline_dev_metrics": baseline_metrics, "final_dev_metrics": final_metrics,
               "final_encoder_sha256": final_encoder_hash,
               "encoder_parameter_l2_change": math.sqrt(encoder_delta_sq),
               "encoder_relative_l2_change": math.sqrt(encoder_delta_sq / encoder_initial_sq),
               "completed_updates": cfg.num_updates, "train_pool_utterances": len(train_rows),
               "dev_utterances": len(dev_rows), "unique_training_utterances_seen": len(exposures),
               "total_training_exposures": sum(exposures.values()), "total_supervised_tokens": total_tokens,
               "total_training_audio_seconds": total_audio_s,
               "exposures_by_utterance": dict(sorted(exposures.items())),
               "actual_audio": dict(sorted(audio_info.items())), "checkpoint": str(checkpoint),
               "prediction_files": {"baseline": "baseline_dev_predictions.jsonl", "final": "final_dev_predictions.jsonl"},
               "history": history, "elapsed_s": time.monotonic() - start, "status": "completed"}
    _write_json(out / "summary.json", summary)
    return summary
