"""Modal workers for the correction-transfer study.

The local entrypoint is intentionally absent. ``experiments/publication_run.py``
is the single orx-supervised launcher. Workers mount the already verified
L2-ARCTIC volume and write to new publication-only data/results volumes.
"""

from __future__ import annotations

from pathlib import Path

import modal


ROOT = Path(__file__).resolve().parents[2] if modal.is_local() else Path("/root")
app = modal.App("whispr-publication-study")

source_data = modal.Volume.from_name("accent-selection-pilot-data")
target_data = modal.Volume.from_name("publication-study-data", create_if_missing=True)
results = modal.Volume.from_name("publication-study-results", create_if_missing=True)
hf_cache = modal.Volume.from_name("ast-asr-cache", create_if_missing=True)

data_image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("ffmpeg")
    .pip_install(
        "datasets==4.4.1",
        "huggingface_hub>=0.35,<2",
        "soundfile==0.14.0",
        "torchcodec==0.16.0",
    )
    .env({"HF_HOME": "/cache/huggingface", "HF_HUB_ENABLE_HF_TRANSFER": "0"})
)

H100_RATE_PER_SECOND = 0.001097 + 8 * 0.0000131 + 32 * 0.00000222
training_image = (
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
        "numpy==2.5.2",
        "jiwer==4.0.0",
    )
    .env({"HF_HOME": "/cache/huggingface", "HF_HUB_ENABLE_HF_TRANSFER": "0"})
    .add_local_file(ROOT / "research/pilot/training.py", "/root/research/pilot/training.py")
    .add_local_file(
        ROOT / "research/pilot/frozen_english_normalizer.py",
        "/root/research/pilot/frozen_english_normalizer.py",
    )
    .add_local_file(ROOT / "paper/data/whisper_normalizer.json", "/root/normalizer.json")
)


@app.function(
    image=data_image,
    cpu=2,
    memory=8192,
    timeout=900,
    retries=0,
    secrets=[modal.Secret.from_name("huggingface")],
    volumes={"/cache/huggingface": hf_cache},
)
def inspect_spire_train() -> dict:
    """Inspect only schema and train metadata; never read validation examples."""
    import os
    import time
    from collections import Counter

    from datasets import get_dataset_split_names, load_dataset, load_dataset_builder

    os.environ["HF_HOME"] = "/cache/huggingface"
    started = time.monotonic()
    repo_id = "VectorSigma389/spire-sies"
    splits = get_dataset_split_names(repo_id)
    builder = load_dataset_builder(repo_id)
    if "train" not in splits:
        raise RuntimeError(f"SPIRE dataset has no train split: {splits}")
    dataset = load_dataset(repo_id, split="train")
    feature_names = sorted(dataset.features) if dataset.features else []
    # Reuse the verified August cache. Remove audio and transcript columns from
    # the Arrow view before iteration, so neither bytes nor labels are copied
    # into Python rows during this metadata-only pass.
    remove = []
    for name, feature in (dataset.features or {}).items():
        if feature.__class__.__name__ == "Audio" or name.lower() in {
            "audio", "audio_bytes", "reference", "reference_text", "sentence", "text", "transcript"
        }:
            remove.append(name)
    stream = dataset.remove_columns(remove)
    # Metadata keys vary across mirrors. We inspect values only from identity
    # and demographic fields. Text/reference fields are never dereferenced.
    safe_candidates = (
        "speaker_id", "speaker", "client_id", "accent", "native_language",
        "language", "gender", "split", "subset",
    )
    counts: dict[str, Counter] = {name: Counter() for name in safe_candidates}
    observed = 0
    for row in stream:
        observed += 1
        for name in safe_candidates:
            if name in row and row[name] is not None:
                counts[name][str(row[name])] += 1
        if observed % 5000 == 0:
            print(f"inspected {observed} train metadata rows", flush=True)
    safe_counts = {
        name: dict(counter.most_common()) for name, counter in counts.items() if counter
    }
    return {
        "repo_id": repo_id,
        "splits": list(splits),
        "builder_features": str(builder.info.features),
        "train_feature_names": feature_names,
        "removed_before_iteration": sorted(remove),
        "train_rows": observed,
        "safe_metadata_counts": safe_counts,
        "validation_examples_read": 0,
        "wall_seconds": time.monotonic() - started,
    }


TARGET_ACCENTS = ("Bengali", "Hindi", "Kannada", "Marathi", "Tamil", "Telugu")


@app.function(
    image=data_image,
    cpu=8,
    memory=16384,
    timeout=900,
    retries=0,
    secrets=[modal.Secret.from_name("huggingface")],
    volumes={"/target": target_data, "/cache/huggingface": hf_cache},
)
def prepare_target_manifest(plan: dict) -> dict:
    """Materialize a balanced, transcript-free SPIRE train subset."""
    import hashlib
    import io
    import json
    import math
    import time
    from collections import Counter, defaultdict

    import soundfile as sf
    from datasets import load_dataset

    started = time.monotonic()
    if plan["seed"] != 12 or plan["per_accent"] != 800:
        raise ValueError("target manifest is frozen to seed 12 and 800 rows per accent")
    dataset = load_dataset("VectorSigma389/spire-sies", split="train")
    required = {
        "uid", "speaker_id", "accent", "split", "duration", "src_sr",
        "audio_bytes", "reference",
    }
    if not required.issubset(dataset.features):
        raise ValueError(f"SPIRE schema changed: {sorted(dataset.features)}")
    metadata = dataset.remove_columns(["audio_bytes", "reference"])
    train_speakers, val_speakers = set(), set()
    eligible: dict[str, list[tuple[str, int]]] = defaultdict(list)
    seen_uids: set[str] = set()
    for index, row in enumerate(metadata):
        uid = str(row["uid"])
        speaker = str(row["speaker_id"])
        split = str(row["split"])
        if uid in seen_uids:
            raise ValueError(f"duplicate SPIRE uid: {uid}")
        seen_uids.add(uid)
        if split == "val":
            val_speakers.add(speaker)
        elif split == "train":
            train_speakers.add(speaker)
            duration = float(row["duration"])
            if row["accent"] in TARGET_ACCENTS and math.isfinite(duration) and 1.0 <= duration <= 30.0:
                rank = hashlib.sha256(f"{plan['seed']}\0{uid}".encode()).hexdigest()
                eligible[str(row["accent"])].append((rank, index))
    overlap = sorted(train_speakers & val_speakers)
    if overlap:
        raise ValueError(f"SPIRE train/val speaker overlap: {overlap[:10]}")
    if len(val_speakers) != 198:
        raise ValueError(f"sealed validation speaker count changed: {len(val_speakers)}")
    selected_indices = []
    for accent in TARGET_ACCENTS:
        candidates = sorted(eligible[accent])
        if len(candidates) < plan["per_accent"]:
            raise ValueError(f"insufficient eligible {accent} rows: {len(candidates)}")
        selected_indices.extend(index for _, index in candidates[: plan["per_accent"]])
    selected_view = dataset.remove_columns(["reference"]).select(sorted(selected_indices))
    output = Path("/target/v1")
    if output.exists():
        raise ValueError("target/v1 already exists; immutable data may not be overwritten")
    audio_dir = output / "wav"
    audio_dir.mkdir(parents=True)
    manifest = []
    total_bytes = 0
    for offset, row in enumerate(selected_view):
        if "reference" in row:
            raise ValueError("reference leaked into target materialization view")
        payload = row["audio_bytes"]
        if not isinstance(payload, bytes) or not payload:
            raise ValueError(f"invalid audio bytes: {row['uid']}")
        info = sf.info(io.BytesIO(payload))
        duration = float(info.frames / info.samplerate)
        if abs(duration - float(row["duration"])) > max(0.02, 1 / info.samplerate):
            raise ValueError(f"duration mismatch: {row['uid']}")
        if info.samplerate != int(row["src_sr"]):
            raise ValueError(f"sample-rate mismatch: {row['uid']}")
        uid = str(row["uid"])
        relative = f"wav/{uid}.wav"
        path = output / relative
        path.write_bytes(payload)
        checksum = hashlib.sha256(payload).hexdigest()
        total_bytes += len(payload)
        manifest.append(
            {
                "utt_id": uid,
                "speaker_id": f"SPIRE_{row['speaker_id']}",
                "native_language": str(row["accent"]),
                "accent": str(row["accent"]),
                "gender": str(row["gender"]),
                "prompt_id": uid,
                "audio_path": relative,
                "audio_sha256": checksum,
                "duration_s": duration,
                "source_sample_rate": info.samplerate,
                "split": "train",
                "source_dataset_split": "train",
                "label_source": "untranscribed_target",
            }
        )
        if (offset + 1) % 500 == 0:
            print(f"materialized {offset + 1}/4800 target clips", flush=True)
    manifest.sort(key=lambda row: row["utt_id"])
    manifest_text = "".join(json.dumps(row, sort_keys=True) + "\n" for row in manifest)
    (output / "manifest.jsonl").write_text(manifest_text)
    selected_speakers = {row["speaker_id"] for row in manifest}
    report = {
        "run_id": plan["run_id"], "status": "completed",
        "repo_id": "VectorSigma389/spire-sies", "seed": 12,
        "selected_utterances": len(manifest),
        "selected_speakers": len(selected_speakers),
        "selected_by_accent": dict(sorted(Counter(row["accent"] for row in manifest).items())),
        "eligible_by_accent": {accent: len(eligible[accent]) for accent in TARGET_ACCENTS},
        "train_speakers": len(train_speakers), "sealed_val_speakers": len(val_speakers),
        "speaker_overlap": 0, "reference_fields_in_manifest": 0,
        "validation_audio_or_reference_examples_read": 0,
        "manifest_sha256": hashlib.sha256(manifest_text.encode()).hexdigest(),
        "selected_ids_sha256": hashlib.sha256(
            json.dumps(sorted(row["utt_id"] for row in manifest), separators=(",", ":")).encode()
        ).hexdigest(),
        "audio_bytes": total_bytes,
        "audio_hours": sum(row["duration_s"] for row in manifest) / 3600,
        "wall_seconds": time.monotonic() - started,
    }
    (output / "preparation_report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    target_data.commit()
    return {**report, "manifest_text": manifest_text}


@app.function(
    image=training_image,
    gpu="H100",
    cpu=8,
    memory=32768,
    timeout=360,
    startup_timeout=180,
    retries=0,
    max_containers=1,
    scaledown_window=2,
    volumes={
        "/data": source_data,
        "/outputs": results,
        "/cache/huggingface": hf_cache,
    },
)
def train_source_smoke(plan: dict) -> dict:
    """Run 100 updates with pseudo source labels and return a hashed receipt."""
    import hashlib
    import importlib.util
    import json
    import sys
    import time
    from concurrent.futures import ThreadPoolExecutor

    started = time.monotonic()
    spec = importlib.util.spec_from_file_location(
        "publication_frozen_training", "/root/research/pilot/training.py"
    )
    core = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = core
    spec.loader.exec_module(core)
    if core.sha256_file(Path(core.__file__)) != plan["training_code_sha256"]:
        raise ValueError("frozen training source changed")
    if plan["config"]["num_updates"] != 100 or plan["config"]["seed"] != 12:
        raise ValueError("smoke must be exactly 100 updates at seed 12")
    source_data.reload()
    rows = plan["rows"]
    train_rows, dev_rows = core.validate_rows(rows)
    cycler = core.IndexCycler(len(train_rows), 12)
    indices = cycler.take(plan["config"]["num_updates"] * plan["config"]["batch_size"])
    needed = {row["utt_id"]: row for row in dev_rows}
    needed.update({train_rows[index]["utt_id"]: train_rows[index] for index in indices})
    local_audio = Path("/tmp/publication-audio")
    local_features = Path("/tmp/publication-features")
    local_features.mkdir(parents=True, exist_ok=True)

    def stage(row: dict) -> str:
        receipt = plan["cache_receipts"][row["utt_id"]]
        paths = (
            (
                Path("/data/l2arctic") / row["audio_path"],
                local_audio / row["audio_path"],
                receipt["audio_sha256"],
            ),
            (
                Path("/data/feature_cache/frozen-v1") / receipt["feature_file"],
                local_features / receipt["feature_file"],
                receipt["feature_sha256"],
            ),
        )
        for source, target, expected in paths:
            payload = source.read_bytes()
            if hashlib.sha256(payload).hexdigest() != expected:
                raise ValueError(f"input checksum mismatch: {row['utt_id']}")
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(payload)
        return row["utt_id"]

    with ThreadPoolExecutor(max_workers=32) as pool:
        staged = list(pool.map(stage, needed.values()))
    output = Path("/outputs") / plan["output_path"]
    if output.exists():
        raise ValueError(f"output path already exists: {output}")
    config = {
        **plan["config"],
        "feature_cache_dir": str(local_features),
        "normalizer_path": "/root/normalizer.json",
    }
    summary = core.train(config, rows, local_audio, output)
    required = (
        "summary.json",
        "provenance.json",
        "manifest.json",
        "history.jsonl",
        "baseline_dev_predictions.jsonl",
        "final_dev_predictions.jsonl",
        "baseline_dev_metrics.json",
        "final_dev_metrics.json",
        "checkpoint_last/model.safetensors",
        "checkpoint_last/training_state.pt",
    )
    hashes = {}
    for relative in required:
        path = output / relative
        if not path.is_file() or path.stat().st_size == 0:
            raise ValueError(f"missing required artifact: {relative}")
        hashes[relative] = core.sha256_file(path)
    if summary["completed_updates"] != 100 or summary["total_training_exposures"] != 800:
        raise ValueError("smoke exposure receipt mismatch")
    if "H100" not in summary["runtime"]["device_name"]:
        raise ValueError("smoke did not use an H100")
    wall_seconds = time.monotonic() - started
    receipt = {
        "run_id": plan["run_id"],
        "status": "completed",
        "gpu": summary["runtime"]["device_name"],
        "completed_updates": summary["completed_updates"],
        "total_training_exposures": summary["total_training_exposures"],
        "train_pool_utterances": summary["train_pool_utterances"],
        "dev_utterances": summary["dev_utterances"],
        "label_source": "source_pseudo",
        "validation_examples_read": 0,
        "staged_utterances": len(staged),
        "staged_ids_sha256": core.canonical_hash(sorted(staged)),
        "artifacts": hashes,
        "wall_seconds": wall_seconds,
        "estimated_resource_cost_usd": wall_seconds * H100_RATE_PER_SECOND,
        "baseline_dev_metrics": summary["baseline_dev_metrics"],
        "final_dev_metrics": summary["final_dev_metrics"],
    }
    (output / "receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    results.commit()
    return json.loads(json.dumps(receipt))


@app.function(
    image=training_image,
    gpu="H100",
    cpu=8,
    memory=32768,
    timeout=480,
    startup_timeout=180,
    retries=0,
    max_containers=2,
    scaledown_window=2,
    volumes={
        "/data": source_data,
        "/outputs": results,
        "/cache/huggingface": hf_cache,
    },
)
def train_source_endpoint(plan: dict) -> dict:
    """Train one 600-update source endpoint from verified cached features."""
    import hashlib
    import importlib.util
    import json
    import sys
    import time
    from concurrent.futures import ThreadPoolExecutor

    started = time.monotonic()
    spec = importlib.util.spec_from_file_location(
        "publication_endpoint_training", "/root/research/pilot/training.py"
    )
    core = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = core
    spec.loader.exec_module(core)
    if core.sha256_file(Path(core.__file__)) != plan["training_code_sha256"]:
        raise ValueError("frozen training source changed")
    cfg = plan["config"]
    if cfg["num_updates"] != 600 or cfg["seed"] != 12 or cfg["batch_size"] != 8:
        raise ValueError("source endpoint must be 600 x 8 at seed 12")
    if plan["label_source"] not in {"source_real", "source_pseudo"}:
        raise ValueError("invalid source endpoint")
    source_data.reload()
    train_rows, dev_rows = core.validate_rows(plan["rows"])
    cycler = core.IndexCycler(len(train_rows), 12)
    indices = cycler.take(cfg["num_updates"] * cfg["batch_size"])
    needed = {row["utt_id"]: row for row in dev_rows}
    needed.update({train_rows[index]["utt_id"]: train_rows[index] for index in indices})
    local_audio = Path("/tmp/publication-audio")
    local_features = Path("/tmp/publication-features")
    local_features.mkdir(parents=True, exist_ok=True)

    def stage(row: dict) -> str:
        receipt = plan["cache_receipts"][row["utt_id"]]
        for source, target, expected in (
            (
                Path("/data/l2arctic") / row["audio_path"],
                local_audio / row["audio_path"],
                receipt["audio_sha256"],
            ),
            (
                Path("/data/feature_cache/frozen-v1") / receipt["feature_file"],
                local_features / receipt["feature_file"],
                receipt["feature_sha256"],
            ),
        ):
            payload = source.read_bytes()
            if hashlib.sha256(payload).hexdigest() != expected:
                raise ValueError(f"input checksum mismatch: {row['utt_id']}")
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(payload)
        return row["utt_id"]

    with ThreadPoolExecutor(max_workers=32) as pool:
        staged = list(pool.map(stage, needed.values()))
    output = Path("/outputs") / plan["output_path"]
    if output.exists():
        raise ValueError(f"output path already exists: {output}")
    summary = core.train(
        {**cfg, "feature_cache_dir": str(local_features), "normalizer_path": "/root/normalizer.json"},
        plan["rows"],
        local_audio,
        output,
    )
    required = (
        "summary.json", "provenance.json", "manifest.json", "history.jsonl",
        "baseline_dev_predictions.jsonl", "final_dev_predictions.jsonl",
        "baseline_dev_metrics.json", "final_dev_metrics.json",
        "checkpoint_last/model.safetensors", "checkpoint_last/training_state.pt",
    )
    hashes = {relative: core.sha256_file(output / relative) for relative in required}
    if any(not (output / relative).is_file() for relative in required):
        raise ValueError("source endpoint artifact set is incomplete")
    if summary["completed_updates"] != 600 or summary["total_training_exposures"] != 4800:
        raise ValueError("source endpoint exposure mismatch")
    if "H100" not in summary["runtime"]["device_name"]:
        raise ValueError("source endpoint did not use an H100")
    wall_seconds = time.monotonic() - started
    receipt = {
        "run_id": plan["run_id"], "status": "completed",
        "label_source": plan["label_source"], "gpu": summary["runtime"]["device_name"],
        "completed_updates": 600, "total_training_exposures": 4800,
        "train_pool_utterances": summary["train_pool_utterances"],
        "dev_utterances": summary["dev_utterances"],
        "validation_examples_read": 0, "staged_utterances": len(staged),
        "staged_ids_sha256": core.canonical_hash(sorted(staged)), "artifacts": hashes,
        "initial_state_sha256": summary["initial_state_sha256"],
        "checkpoint_sha256": hashes["checkpoint_last/model.safetensors"],
        "wall_seconds": wall_seconds,
        "estimated_resource_cost_usd": wall_seconds * H100_RATE_PER_SECOND,
        "baseline_dev_metrics": summary["baseline_dev_metrics"],
        "final_dev_metrics": summary["final_dev_metrics"],
    }
    (output / "receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    results.commit()
    return json.loads(json.dumps(receipt))


@app.function(
    image=training_image,
    gpu="H100",
    cpu=8,
    memory=32768,
    timeout=900,
    startup_timeout=180,
    retries=0,
    max_containers=1,
    scaledown_window=2,
    secrets=[modal.Secret.from_name("huggingface")],
    volumes={"/target": target_data, "/cache/huggingface": hf_cache},
)
def label_target_with_teacher(plan: dict) -> dict:
    """Generate deterministic large-v3 target labels and confidence scores."""
    import hashlib
    import importlib.metadata
    import json
    import math
    import time
    from concurrent.futures import ThreadPoolExecutor

    import librosa
    import numpy as np
    import soundfile as sf
    import torch
    from transformers import WhisperForConditionalGeneration, WhisperProcessor
    from transformers.utils import logging as hf_logging

    hf_logging.set_verbosity_error()
    started = time.monotonic()
    if plan["teacher_model"] != "openai/whisper-large-v3" or plan["batch_size"] != 16:
        raise ValueError("target teacher identity or batch size changed")
    target_data.reload()
    root = Path("/target/v1")
    manifest_path = root / "manifest.jsonl"
    if hashlib.sha256(manifest_path.read_bytes()).hexdigest() != plan["manifest_sha256"]:
        raise ValueError("target manifest hash mismatch")
    rows = [json.loads(line) for line in manifest_path.read_text().splitlines()]
    if len(rows) != 4800 or any("reference_text" in row for row in rows):
        raise ValueError("target teacher received labeled or incomplete manifest")
    local_root = Path("/tmp/target-audio")

    def stage(row: dict) -> str:
        source = root / row["audio_path"]
        payload = source.read_bytes()
        if hashlib.sha256(payload).hexdigest() != row["audio_sha256"]:
            raise ValueError(f"target audio hash mismatch: {row['utt_id']}")
        target = local_root / row["audio_path"]
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(payload)
        return row["utt_id"]

    with ThreadPoolExecutor(max_workers=32) as pool:
        staged = list(pool.map(stage, rows))
    model_id, revision = plan["teacher_model"], plan["teacher_revision"]
    processor = WhisperProcessor.from_pretrained(model_id, revision=revision)
    model = WhisperForConditionalGeneration.from_pretrained(
        model_id, revision=revision, dtype=torch.float32, attn_implementation="eager"
    ).to("cuda").eval()
    torch.manual_seed(0)
    torch.set_num_threads(8)
    predictions = []
    for begin in range(0, len(rows), plan["batch_size"]):
        if time.monotonic() - started > 840:
            raise TimeoutError("target teacher stopped before the hard timeout")
        batch = rows[begin : begin + plan["batch_size"]]
        arrays = []
        for row in batch:
            audio, sr = sf.read(local_root / row["audio_path"], dtype="float32")
            if audio.ndim != 1 or abs(len(audio) / sr - row["duration_s"]) > 0.02:
                raise ValueError(f"target audio structure mismatch: {row['utt_id']}")
            if sr != 16000:
                audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
            arrays.append(audio)
        features = processor(
            arrays, sampling_rate=16000, return_tensors="pt", padding="max_length",
            truncation=True, return_attention_mask=True,
        )
        with torch.inference_mode():
            generated = model.generate(
                input_features=features.input_features.to("cuda"),
                attention_mask=features.attention_mask.to("cuda"),
                language="en", task="transcribe", return_timestamps=False,
                do_sample=False, num_beams=1, max_new_tokens=224,
                return_dict_in_generate=True, output_scores=True,
            )
            transition = model.compute_transition_scores(
                generated.sequences, generated.scores, normalize_logits=True
            ).detach().cpu()
        hypotheses = processor.batch_decode(
            generated.sequences, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        for row, hypothesis, token_scores, sequence in zip(
            batch, hypotheses, transition, generated.sequences.detach().cpu().tolist()
        ):
            finite = token_scores[torch.isfinite(token_scores) & (token_scores < 0)]
            if finite.numel() == 0:
                raise ValueError(f"teacher returned no finite token scores: {row['utt_id']}")
            text = hypothesis.strip()
            if not text:
                raise ValueError(f"teacher returned empty target label: {row['utt_id']}")
            predictions.append(
                {
                    **row,
                    "pseudo_text": text,
                    "teacher_mean_logprob": float(finite.mean()),
                    "teacher_min_logprob": float(finite.min()),
                    "teacher_token_count": int(finite.numel()),
                    "teacher_generated_token_ids": sequence,
                    "label_source": "whisper_large_v3_pseudo",
                }
            )
        if len(predictions) % 200 == 0:
            print(f"teacher labeled {len(predictions)}/4800 clips", flush=True)
    predictions.sort(key=lambda row: row["utt_id"])
    text = "".join(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n" for row in predictions)
    pseudo_path = root / "pseudo_large_v3_seed12.jsonl"
    if pseudo_path.exists():
        raise ValueError("target pseudo-label artifact already exists")
    pseudo_path.write_text(text)
    audit = []
    for accent in TARGET_ACCENTS:
        group = sorted(
            (row for row in predictions if row["accent"] == accent),
            key=lambda row: (row["teacher_mean_logprob"], row["utt_id"]),
        )
        audit.extend({**row, "confidence_stratum": "low"} for row in group[:10])
        audit.extend({**row, "confidence_stratum": "high"} for row in group[-10:])
    # The blind audit manifest contains no corpus reference and no model text.
    blind_fields = (
        "utt_id", "speaker_id", "native_language", "accent", "gender",
        "audio_path", "audio_sha256", "duration_s", "confidence_stratum",
        "teacher_mean_logprob",
    )
    blind = [{key: row[key] for key in blind_fields} for row in audit]
    blind.sort(key=lambda row: (row["accent"], row["confidence_stratum"], row["utt_id"]))
    blind_text = "".join(json.dumps(row, sort_keys=True) + "\n" for row in blind)
    (root / "audit_120_blind.jsonl").write_text(blind_text)
    wall_seconds = time.monotonic() - started
    receipt = {
        "run_id": plan["run_id"], "status": "completed", "model_id": model_id,
        "model_revision": revision, "dtype": "float32", "batch_size": 16,
        "generation": {"language": "en", "task": "transcribe", "return_timestamps": False,
                       "do_sample": False, "num_beams": 1, "max_new_tokens": 224},
        "rows": len(predictions), "audit_rows": len(blind),
        "manifest_sha256": plan["manifest_sha256"],
        "pseudo_manifest_sha256": hashlib.sha256(text.encode()).hexdigest(),
        "audit_manifest_sha256": hashlib.sha256(blind_text.encode()).hexdigest(),
        "staged_ids_sha256": hashlib.sha256(
            json.dumps(sorted(staged), separators=(",", ":")).encode()
        ).hexdigest(),
        "confidence": {
            "minimum": min(row["teacher_mean_logprob"] for row in predictions),
            "maximum": max(row["teacher_mean_logprob"] for row in predictions),
            "mean": sum(row["teacher_mean_logprob"] for row in predictions) / len(predictions),
        },
        "validation_audio_or_reference_examples_read": 0,
        "gpu": torch.cuda.get_device_name(),
        "versions": {name: importlib.metadata.version(name) for name in (
            "torch", "transformers", "librosa", "soundfile"
        )},
        "wall_seconds": wall_seconds,
        "estimated_resource_cost_usd": wall_seconds * H100_RATE_PER_SECOND,
    }
    (root / "pseudo_large_v3_seed12.receipt.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n"
    )
    target_data.commit()
    return {**receipt, "pseudo_manifest_text": text, "audit_manifest_text": blind_text}
