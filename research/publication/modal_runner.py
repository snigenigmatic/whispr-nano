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
TARGET_H100_RATE_PER_SECOND = 0.001097 + 16 * 0.0000131 + 64 * 0.00000222
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
    .add_local_file(
        ROOT / "src/publication_study/arithmetic.py",
        "/root/src/publication_study/arithmetic.py",
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
    cpu=16,
    memory=65536,
    timeout=420,
    startup_timeout=180,
    retries=0,
    max_containers=1,
    scaledown_window=2,
    volumes={
        "/source": source_data,
        "/target": target_data,
        "/outputs": results,
        "/cache/huggingface": hf_cache,
    },
)
def train_target_pseudo(plan: dict) -> dict:
    """Cache exact tiny features, then train a frozen target-pseudo arm."""
    import hashlib
    import importlib.util
    import json
    import math
    import os
    import sys
    import time
    from concurrent.futures import ThreadPoolExecutor

    import numpy as np
    import scipy
    import scipy.signal
    import soundfile as sf
    import transformers
    from transformers import WhisperProcessor

    started = time.monotonic()
    spec = importlib.util.spec_from_file_location(
        "publication_target_training", "/root/research/pilot/training.py"
    )
    core = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = core
    spec.loader.exec_module(core)
    if core.sha256_file(Path(core.__file__)) != plan["training_code_sha256"]:
        raise ValueError("frozen target training source changed")
    cfg = plan["config"]
    if cfg["num_updates"] != 600 or cfg["seed"] != 12 or cfg["batch_size"] != 8:
        raise ValueError("target endpoint must be 600 x 8 at seed 12")
    source_data.reload()
    target_data.reload()
    target_root = Path("/target/v1")
    pseudo_path = target_root / "pseudo_large_v3_seed12.jsonl"
    if core.sha256_file(pseudo_path) != plan["pseudo_manifest_sha256"]:
        raise ValueError("target pseudo-label manifest hash mismatch")
    pseudo_rows = [json.loads(line) for line in pseudo_path.read_text().splitlines() if line]
    if len(pseudo_rows) != 4800:
        raise ValueError("target pseudo-label row count changed")
    forbidden = {"reference", "reference_text", "transcript", "sentence", "text"}
    if any(forbidden & set(row) for row in pseudo_rows):
        raise ValueError("target pseudo-label manifest contains a corpus reference field")
    if plan["selection"] == "all":
        selected_pseudo = pseudo_rows
    elif plan["selection"] == "top50_per_accent":
        selected_pseudo = []
        for accent in TARGET_ACCENTS:
            group = sorted(
                (row for row in pseudo_rows if row["accent"] == accent),
                key=lambda row: (-row["teacher_mean_logprob"], row["utt_id"]),
            )
            if len(group) != 800:
                raise ValueError(f"target confidence group changed: {accent}")
            selected_pseudo.extend(group[:400])
        selected_pseudo.sort(key=lambda row: row["utt_id"])
    else:
        raise ValueError("unknown target pseudo-label selection")
    if len(selected_pseudo) != plan["expected_train_pool"]:
        raise ValueError("target confidence selection count changed")
    target_rows = []
    for row in selected_pseudo:
        transformed = dict(row)
        transformed["reference_text"] = transformed.pop("pseudo_text")
        transformed["split"] = "train"
        transformed["label_source"] = "whisper_large_v3_pseudo"
        target_rows.append(transformed)
    dev_rows = [dict(row) for row in plan["source_dev_rows"]]
    rows = target_rows + dev_rows
    core.validate_rows(rows)

    processor = WhisperProcessor.from_pretrained(
        cfg["model_id"], revision=cfg["model_revision"], language="english", task="transcribe"
    )
    feature_signature = core.canonical_hash({
        "revision": cfg["model_revision"], "transformers": transformers.__version__,
        "numpy": np.__version__, "scipy": scipy.__version__, "soundfile": sf.__version__,
        "downmix": "arithmetic mean", "resampling": "scipy.signal.resample_poly to 16000",
        "feature_extractor": processor.feature_extractor.to_dict(),
    })
    local_features = Path("/tmp/publication-target-features")
    local_features.mkdir(parents=True, exist_ok=True)
    volume_features = target_root / "tiny_features_frozen_v1"
    volume_features.mkdir(parents=True, exist_ok=True)
    receipt_path = target_root / "tiny_features_frozen_v1.receipt.json"
    existing: dict[str, dict] = {}
    if receipt_path.exists():
        cached = json.loads(receipt_path.read_text())
        if cached.get("feature_signature") != feature_signature:
            raise ValueError("persisted target feature signature changed")
        existing = {row["utt_id"]: row for row in cached["files"]}
    if plan["requires_persisted_feature_cache"] and not {
        row["utt_id"] for row in target_rows
    }.issubset(existing):
        raise ValueError("required persisted target feature cache is incomplete")

    target_receipts: dict[str, dict] = {}
    created = reused = 0

    def persist_feature_progress(status: str) -> dict:
        persisted = {**existing, **target_receipts}
        report = {
            "status": status, "manifest_sha256": plan["target_manifest_sha256"],
            "feature_signature": feature_signature, "rows": len(persisted),
            "created_features": created, "reused_features": reused,
            "files": sorted(persisted.values(), key=lambda row: row["utt_id"]),
        }
        receipt_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        target_data.commit()
        return report

    def materialize(row: dict) -> tuple[dict, bytes | None]:
        uid = row["utt_id"]
        cached = existing.get(uid)
        if cached:
            source = volume_features / cached["feature_file"]
            payload = source.read_bytes()
            if hashlib.sha256(payload).hexdigest() != cached["feature_sha256"]:
                raise ValueError(f"persisted target feature checksum mismatch: {uid}")
            (local_features / cached["feature_file"]).write_bytes(payload)
            return cached, None
        source = target_root / row["audio_path"]
        payload = source.read_bytes()
        checksum = hashlib.sha256(payload).hexdigest()
        if checksum != row["audio_sha256"]:
            raise ValueError(f"target audio checksum mismatch: {uid}")
        audio, sample_rate = sf.read(source, dtype="float32", always_2d=True)
        duration = len(audio) / sample_rate
        if not len(audio) or audio.shape[1] < 1 or not np.isfinite(audio).all():
            raise ValueError(f"invalid target audio: {uid}")
        if abs(duration - row["duration_s"]) > max(0.02, 1 / sample_rate):
            raise ValueError(f"target audio duration mismatch: {uid}")
        channels = audio.shape[1]
        audio = audio.mean(axis=1)
        if sample_rate != 16000:
            divisor = math.gcd(sample_rate, 16000)
            audio = scipy.signal.resample_poly(
                audio, 16000 // divisor, sample_rate // divisor
            ).astype(np.float32)
        extracted = processor.feature_extractor(
            audio, sampling_rate=16000, return_tensors="np"
        ).input_features[0].astype(np.float32)
        if extracted.shape != (80, 3000) or not np.isfinite(extracted).all():
            raise ValueError(f"invalid target feature tensor: {uid}")
        feature_file = core.canonical_hash([checksum, feature_signature]) + ".npy"
        local = local_features / feature_file
        temporary = local.with_name(local.name + f".{uid}.{os.getpid()}.tmp")
        with temporary.open("wb") as stream:
            np.save(stream, extracted, allow_pickle=False)
        temporary.replace(local)
        feature_payload = local.read_bytes()
        receipt = {
            "utt_id": uid, "audio_sha256": checksum, "duration_s": duration,
            "original_sample_rate": sample_rate, "channels": int(channels),
            "feature_file": feature_file,
            "feature_sha256": hashlib.sha256(feature_payload).hexdigest(),
            "feature_bytes": len(feature_payload),
        }
        return receipt, feature_payload

    # Bounded chunks avoid retaining the full 13-hour waveform set in RAM.
    for begin in range(0, len(target_rows), 64):
        chunk = target_rows[begin : begin + 64]
        with ThreadPoolExecutor(max_workers=16) as pool:
            prepared = list(pool.map(materialize, chunk))
        for receipt, payload in prepared:
            target_receipts[receipt["utt_id"]] = receipt
            if payload is None:
                reused += 1
            else:
                destination = volume_features / receipt["feature_file"]
                if destination.exists():
                    if hashlib.sha256(destination.read_bytes()).hexdigest() != receipt["feature_sha256"]:
                        raise ValueError(f"target feature collision: {receipt['utt_id']}")
                else:
                    destination.write_bytes(payload)
                created += 1
        if len(target_receipts) % 400 == 0:
            print(f"tiny features prepared {len(target_receipts)}/4800", flush=True)
        if len(target_receipts) % 800 == 0:
            persist_feature_progress("caching")

    source_receipts = {
        row["utt_id"]: dict(plan["source_dev_receipts"][row["utt_id"]]) for row in dev_rows
    }

    def stage_source(row: dict) -> dict:
        receipt = dict(source_receipts[row["utt_id"]])
        source = Path("/source/feature_cache/frozen-v1") / receipt["feature_file"]
        payload = source.read_bytes()
        if hashlib.sha256(payload).hexdigest() != receipt["feature_sha256"]:
            raise ValueError("source development feature checksum mismatch")
        (local_features / receipt["feature_file"]).write_bytes(payload)
        audio_path = Path("/source/l2arctic") / row["audio_path"]
        audio_payload = audio_path.read_bytes()
        if hashlib.sha256(audio_payload).hexdigest() != receipt["audio_sha256"]:
            raise ValueError("source development audio checksum mismatch")
        info = sf.info(audio_path)
        duration = info.frames / info.samplerate
        if abs(duration - float(receipt["duration_s"])) > max(0.01, 1 / info.samplerate):
            raise ValueError("source development audio duration mismatch")
        if int(receipt["original_sample_rate"]) != info.samplerate:
            raise ValueError("source development sample-rate mismatch")
        receipt["channels"] = info.channels
        return receipt

    with ThreadPoolExecutor(max_workers=32) as pool:
        staged_source_receipts = list(pool.map(stage_source, dev_rows))
    source_receipts = {row["utt_id"]: row for row in staged_source_receipts}
    verified_receipts = {**target_receipts, **source_receipts}
    target_feature_report = persist_feature_progress("completed")

    output = Path("/outputs") / plan["output_path"]
    if output.exists():
        raise ValueError(f"output path already exists: {output}")
    summary = core.train(
        {**cfg, "feature_cache_dir": str(local_features), "normalizer_path": "/root/normalizer.json"},
        rows,
        "/tmp/unused-verified-audio",
        output,
        verified_feature_receipts=verified_receipts,
    )
    required = (
        "summary.json", "provenance.json", "manifest.json", "history.jsonl",
        "baseline_dev_predictions.jsonl", "final_dev_predictions.jsonl",
        "baseline_dev_metrics.json", "final_dev_metrics.json",
        "checkpoint_last/model.safetensors", "checkpoint_last/training_state.pt",
    )
    if any(not (output / relative).is_file() for relative in required):
        raise ValueError("target endpoint artifact set is incomplete")
    hashes = {relative: core.sha256_file(output / relative) for relative in required}
    if summary["completed_updates"] != 600 or summary["total_training_exposures"] != 4800:
        raise ValueError("target endpoint exposure mismatch")
    if summary["unique_training_utterances_seen"] != len(target_rows):
        raise ValueError("target endpoint unique exposure count changed")
    if "H100" not in summary["runtime"]["device_name"]:
        raise ValueError("target endpoint did not use an H100")
    wall_seconds = time.monotonic() - started
    receipt = {
        "run_id": plan["run_id"], "status": "completed", "label_source": plan["label_source"],
        "selection": plan["selection"],
        "selected_ids_sha256": core.canonical_hash(sorted(row["utt_id"] for row in target_rows)),
        "selected_by_accent": {
            accent: sum(row["accent"] == accent for row in target_rows) for accent in TARGET_ACCENTS
        },
        "minimum_selected_mean_logprob_by_accent": {
            accent: min(
                row["teacher_mean_logprob"] for row in selected_pseudo if row["accent"] == accent
            ) for accent in TARGET_ACCENTS
        },
        "gpu": summary["runtime"]["device_name"], "completed_updates": 600,
        "total_training_exposures": 4800,
        "unique_training_utterances_seen": summary["unique_training_utterances_seen"],
        "train_pool_utterances": len(target_rows), "dev_utterances": len(dev_rows),
        "validation_examples_read": 0, "target_features_created": created,
        "target_features_reused": reused,
        "source_dev_features_staged": len(staged_source_receipts),
        "target_feature_receipt_sha256": core.sha256_file(receipt_path),
        "artifacts": hashes, "initial_state_sha256": summary["initial_state_sha256"],
        "checkpoint_sha256": hashes["checkpoint_last/model.safetensors"],
        "wall_seconds": wall_seconds,
        "estimated_resource_cost_usd": wall_seconds * TARGET_H100_RATE_PER_SECOND,
        "baseline_dev_metrics": summary["baseline_dev_metrics"],
        "final_dev_metrics": summary["final_dev_metrics"],
    }
    (output / "receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    results.commit()
    return json.loads(json.dumps(receipt))


@app.function(
    image=data_image,
    cpu=4,
    memory=8192,
    timeout=300,
    retries=0,
    volumes={"/target": target_data, "/outputs": results},
)
def recover_target_pseudo_receipt(plan: dict) -> dict:
    """Validate a persisted complete endpoint and reconstruct its failed receipt."""
    import hashlib
    import json
    import time

    started = time.monotonic()
    target_data.reload()
    results.reload()
    output = Path("/outputs/seed12/target_pseudo")
    receipt_path = output / "receipt.json"
    if receipt_path.exists():
        raise ValueError("target endpoint receipt already exists")
    required = (
        "summary.json", "provenance.json", "manifest.json", "history.jsonl",
        "baseline_dev_predictions.jsonl", "final_dev_predictions.jsonl",
        "baseline_dev_metrics.json", "final_dev_metrics.json",
        "checkpoint_last/model.safetensors", "checkpoint_last/training_state.pt",
    )
    if any(not (output / relative).is_file() or (output / relative).stat().st_size == 0 for relative in required):
        raise ValueError("persisted target endpoint artifact set is incomplete")
    sha256 = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    artifacts = {relative: sha256(output / relative) for relative in required}
    summary = json.loads((output / "summary.json").read_text())
    if (
        summary.get("status") != "completed"
        or summary.get("completed_updates") != 600
        or summary.get("total_training_exposures") != 4800
        or summary.get("unique_training_utterances_seen") != 4800
        or summary.get("train_pool_utterances") != 4800
        or summary.get("dev_utterances") != 974
        or "H100" not in summary.get("runtime", {}).get("device_name", "")
    ):
        raise ValueError("persisted target endpoint summary invariants failed")
    pseudo_path = Path("/target/v1/pseudo_large_v3_seed12.jsonl")
    if sha256(pseudo_path) != plan["pseudo_manifest_sha256"]:
        raise ValueError("persisted target pseudo-label hash changed")
    pseudo = {
        row["utt_id"]: row
        for row in (json.loads(line) for line in pseudo_path.read_text().splitlines() if line)
    }
    manifest = json.loads((output / "manifest.json").read_text())
    train_rows = [row for row in manifest if row["split"] == "train"]
    dev_rows = [row for row in manifest if row["split"] != "train"]
    if {row["utt_id"] for row in train_rows} != set(pseudo):
        raise ValueError("persisted endpoint train IDs differ from target pseudo-labels")
    if any(
        row.get("reference_text") != pseudo[row["utt_id"]]["pseudo_text"]
        or row.get("label_source") != "whisper_large_v3_pseudo"
        for row in train_rows
    ):
        raise ValueError("persisted endpoint objective differs from target pseudo-labels")
    dev_ids_hash = hashlib.sha256(json.dumps(
        sorted(row["utt_id"] for row in dev_rows), separators=(",", ":")
    ).encode()).hexdigest()
    if dev_ids_hash != plan["source_dev_ids_sha256"]:
        raise ValueError("persisted endpoint source dev cohort changed")
    feature_receipt = Path("/target/v1/tiny_features_frozen_v1.receipt.json")
    feature_report = json.loads(feature_receipt.read_text())
    if feature_report.get("status") != "completed" or feature_report.get("rows") != 4800:
        raise ValueError("persisted target feature cache is incomplete")
    wall_seconds = float(plan["original_app_wall_seconds"])
    receipt = {
        "run_id": plan["recovered_run_id"], "status": "completed",
        "recovered_after_receipt_exception": True,
        "original_modal_app_id": plan["original_modal_app_id"],
        "original_app_wall_seconds": wall_seconds,
        "label_source": "target_pseudo", "selection": "all",
        "gpu": summary["runtime"]["device_name"], "completed_updates": 600,
        "total_training_exposures": 4800, "unique_training_utterances_seen": 4800,
        "train_pool_utterances": 4800, "dev_utterances": 974,
        "validation_examples_read": 0, "target_features_created": 4800,
        "target_feature_receipt_sha256": sha256(feature_receipt),
        "artifacts": artifacts, "initial_state_sha256": summary["initial_state_sha256"],
        "checkpoint_sha256": artifacts["checkpoint_last/model.safetensors"],
        "estimated_resource_cost_usd": wall_seconds * TARGET_H100_RATE_PER_SECOND,
        "baseline_dev_metrics": summary["baseline_dev_metrics"],
        "final_dev_metrics": summary["final_dev_metrics"],
        "recovery_validation_wall_seconds": time.monotonic() - started,
    }
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    results.commit()
    return receipt


@app.function(
    image=training_image,
    gpu="H100",
    cpu=8,
    memory=32768,
    timeout=420,
    startup_timeout=180,
    retries=0,
    max_containers=1,
    scaledown_window=2,
    volumes={
        "/source": source_data,
        "/outputs": results,
        "/cache/huggingface": hf_cache,
    },
)
def build_and_evaluate_global(plan: dict) -> dict:
    """Apply the exact global Pseudo2Real vector and evaluate on source dev."""
    import hashlib
    import importlib.util
    import json
    import shutil
    import sys
    import time
    from concurrent.futures import ThreadPoolExecutor

    import numpy as np
    import torch
    from safetensors.torch import load_file, save_file
    from transformers import WhisperForConditionalGeneration, WhisperProcessor

    started = time.monotonic()
    train_spec = importlib.util.spec_from_file_location(
        "publication_global_eval", "/root/research/pilot/training.py"
    )
    core = importlib.util.module_from_spec(train_spec)
    sys.modules[train_spec.name] = core
    train_spec.loader.exec_module(core)
    arithmetic_spec = importlib.util.spec_from_file_location(
        "publication_global_arithmetic", "/root/src/publication_study/arithmetic.py"
    )
    arithmetic = importlib.util.module_from_spec(arithmetic_spec)
    arithmetic_spec.loader.exec_module(arithmetic)
    if core.sha256_file(Path(core.__file__)) != plan["training_code_sha256"]:
        raise ValueError("global evaluator training source changed")
    if core.sha256_file(Path(arithmetic.__file__)) != plan["arithmetic_code_sha256"]:
        raise ValueError("global correction arithmetic source changed")
    if plan["scale"] != 1.0 or len(plan["source_dev_rows"]) != 974:
        raise ValueError("global correction scale or source dev cohort changed")
    checkpoints = {
        "source_real": Path("/outputs/seed12/source_real/checkpoint_last"),
        "source_pseudo": Path("/outputs/seed12/source_pseudo/checkpoint_last"),
        "target_pseudo": Path("/outputs/seed12/target_pseudo/checkpoint_last"),
    }
    model_paths = {name: path / "model.safetensors" for name, path in checkpoints.items()}
    for name, path in model_paths.items():
        if core.sha256_file(path) != plan["checkpoint_sha256"][name]:
            raise ValueError(f"{name} checkpoint hash mismatch")
    states = {name: load_file(str(path), device="cpu") for name, path in model_paths.items()}
    corrected, correction = arithmetic.apply_global_correction(
        states["source_real"], states["source_pseudo"], states["target_pseudo"], plan["scale"]
    )
    fingerprints = {name: arithmetic.state_fingerprint(state) for name, state in states.items()}
    fingerprints["global_pseudo2real"] = arithmetic.state_fingerprint(corrected)
    if fingerprints["source_real"] == fingerprints["source_pseudo"]:
        raise ValueError("source correction vector is identically zero")
    output = Path("/outputs") / plan["output_path"]
    if output.exists():
        raise ValueError(f"output path already exists: {output}")
    checkpoint = output / "checkpoint_last"
    checkpoint.mkdir(parents=True)
    for source in checkpoints["target_pseudo"].iterdir():
        if source.is_file() and source.name not in {"model.safetensors", "training_state.pt"}:
            shutil.copy2(source, checkpoint / source.name)
    save_file(corrected, str(checkpoint / "model.safetensors"), metadata={
        "format": "pt", "formula": correction["formula"], "scale": str(plan["scale"])
    })
    corrected_sha = core.sha256_file(checkpoint / "model.safetensors")

    local_features = Path("/tmp/publication-global-features")
    local_features.mkdir(parents=True)
    receipts = plan["source_dev_receipts"]

    def stage(row: dict) -> str:
        receipt = receipts[row["utt_id"]]
        payload = (Path("/source/feature_cache/frozen-v1") / receipt["feature_file"]).read_bytes()
        if hashlib.sha256(payload).hexdigest() != receipt["feature_sha256"]:
            raise ValueError(f"source dev feature checksum mismatch: {row['utt_id']}")
        (local_features / receipt["feature_file"]).write_bytes(payload)
        return row["utt_id"]

    with ThreadPoolExecutor(max_workers=32) as pool:
        staged = list(pool.map(stage, plan["source_dev_rows"]))
    processor = WhisperProcessor.from_pretrained(checkpoint, local_files_only=True)
    model = WhisperForConditionalGeneration.from_pretrained(
        checkpoint, local_files_only=True, torch_dtype=torch.float32, attn_implementation="eager"
    ).to("cuda").eval()
    from research.pilot.frozen_english_normalizer import EnglishTextNormalizer
    english_normalize = EnglishTextNormalizer(json.loads(Path("/root/normalizer.json").read_text()))
    predictions = []
    generation = {
        "language": "en", "task": "transcribe", "return_timestamps": False,
        "max_new_tokens": 224, "do_sample": False, "num_beams": 1, "use_cache": True,
    }
    rows = sorted(plan["source_dev_rows"], key=lambda row: row["utt_id"])
    with torch.inference_mode():
        for begin in range(0, len(rows), 8):
            chunk = rows[begin : begin + 8]
            features = []
            for row in chunk:
                receipt = receipts[row["utt_id"]]
                value = np.load(local_features / receipt["feature_file"], allow_pickle=False)
                if value.shape != (80, 3000) or value.dtype != np.float32 or not np.isfinite(value).all():
                    raise ValueError(f"invalid source dev feature: {row['utt_id']}")
                features.append(value)
            generated = model.generate(input_features=torch.from_numpy(np.stack(features)).to("cuda"), **generation)
            hypotheses = processor.batch_decode(
                generated, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
            for row, hypothesis, sequence in zip(chunk, hypotheses, generated.cpu().tolist()):
                reference = row["reference_text"]
                hypothesis = hypothesis.strip()
                predictions.append({
                    **{key: row[key] for key in (
                        "utt_id", "speaker_id", "native_language", "prompt_id", "split"
                    )},
                    "reference_text": reference, "prediction": hypothesis,
                    "generated_token_ids": sequence,
                    "minimal": core.word_counts(
                        core.minimal_normalize(reference), core.minimal_normalize(hypothesis)
                    ),
                    "english": core.word_counts(
                        english_normalize(reference), english_normalize(hypothesis)
                    ),
                })
    prediction_path = output / "source_dev_predictions.jsonl"
    prediction_path.write_text("".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in predictions
    ))
    metrics = core.summarize_predictions(predictions)
    (output / "source_dev_metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n"
    )
    correction_report = {
        **correction, "state_fingerprints": fingerprints,
        "input_checkpoint_sha256": plan["checkpoint_sha256"],
        "corrected_checkpoint_sha256": corrected_sha,
        "arithmetic_code_sha256": plan["arithmetic_code_sha256"],
        "evaluation_cohort": "L2-ARCTIC frozen development only",
        "target_validation_examples_read": 0,
    }
    report_path = output / "correction_report.json"
    report_path.write_text(json.dumps(correction_report, indent=2, sort_keys=True) + "\n")
    artifact_paths = {
        "checkpoint_last/model.safetensors": checkpoint / "model.safetensors",
        "correction_report.json": report_path,
        "source_dev_predictions.jsonl": prediction_path,
        "source_dev_metrics.json": output / "source_dev_metrics.json",
    }
    artifact_hashes = {name: core.sha256_file(path) for name, path in artifact_paths.items()}
    wall_seconds = time.monotonic() - started
    receipt = {
        "run_id": plan["run_id"], "status": "completed", "method": "global_pseudo2real",
        "formula": correction["formula"], "scale": plan["scale"],
        "gpu": torch.cuda.get_device_name(), "source_dev_utterances": len(predictions),
        "target_validation_examples_read": 0, "staged_source_dev_features": len(staged),
        "corrected_checkpoint_sha256": corrected_sha, "artifacts": artifact_hashes,
        "correction": correction, "source_dev_metrics": metrics,
        "wall_seconds": wall_seconds,
        "estimated_resource_cost_usd": wall_seconds * H100_RATE_PER_SECOND,
    }
    (output / "receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    results.commit()
    return json.loads(json.dumps(receipt))


@app.function(
    image=training_image,
    gpu="H100",
    cpu=8,
    memory=32768,
    timeout=420,
    startup_timeout=180,
    retries=0,
    max_containers=1,
    scaledown_window=2,
    volumes={
        "/source": source_data,
        "/outputs": results,
        "/cache/huggingface": hf_cache,
    },
)
def tune_global_pseudo2real(plan: dict) -> dict:
    """Tune the published global correction scale on source dev only."""
    import hashlib
    import importlib.util
    import json
    import shutil
    import sys
    import time
    from concurrent.futures import ThreadPoolExecutor

    import numpy as np
    import torch
    from safetensors.torch import load_file, save_file
    from transformers import WhisperForConditionalGeneration, WhisperProcessor

    started = time.monotonic()
    train_spec = importlib.util.spec_from_file_location(
        "publication_scale_eval", "/root/research/pilot/training.py"
    )
    core = importlib.util.module_from_spec(train_spec)
    sys.modules[train_spec.name] = core
    train_spec.loader.exec_module(core)
    arithmetic_spec = importlib.util.spec_from_file_location(
        "publication_scale_arithmetic", "/root/src/publication_study/arithmetic.py"
    )
    arithmetic = importlib.util.module_from_spec(arithmetic_spec)
    arithmetic_spec.loader.exec_module(arithmetic)
    if core.sha256_file(Path(core.__file__)) != plan["training_code_sha256"]:
        raise ValueError("scale tuner training source changed")
    if core.sha256_file(Path(arithmetic.__file__)) != plan["arithmetic_code_sha256"]:
        raise ValueError("scale tuner arithmetic source changed")
    expected_scales = [round(index / 10, 1) for index in range(1, 11)]
    if plan["scales"] != expected_scales or len(plan["source_dev_rows"]) != 974:
        raise ValueError("published source-dev scale grid changed")
    checkpoints = {
        "source_real": Path("/outputs/seed12/source_real/checkpoint_last"),
        "source_pseudo": Path("/outputs/seed12/source_pseudo/checkpoint_last"),
        "target_pseudo": Path("/outputs/seed12/target_pseudo/checkpoint_last"),
    }
    model_paths = {name: path / "model.safetensors" for name, path in checkpoints.items()}
    for name, path in model_paths.items():
        if core.sha256_file(path) != plan["checkpoint_sha256"][name]:
            raise ValueError(f"scale tuner {name} checkpoint hash mismatch")
    states = {name: load_file(str(path), device="cpu") for name, path in model_paths.items()}
    local_features = Path("/tmp/publication-scale-features")
    local_features.mkdir(parents=True)
    receipts = plan["source_dev_receipts"]

    def stage(row: dict) -> str:
        receipt = receipts[row["utt_id"]]
        payload = (Path("/source/feature_cache/frozen-v1") / receipt["feature_file"]).read_bytes()
        if hashlib.sha256(payload).hexdigest() != receipt["feature_sha256"]:
            raise ValueError(f"scale tuner source feature hash mismatch: {row['utt_id']}")
        (local_features / receipt["feature_file"]).write_bytes(payload)
        return row["utt_id"]

    with ThreadPoolExecutor(max_workers=32) as pool:
        staged = list(pool.map(stage, plan["source_dev_rows"]))
    processor = WhisperProcessor.from_pretrained(checkpoints["target_pseudo"], local_files_only=True)
    model = WhisperForConditionalGeneration.from_pretrained(
        checkpoints["target_pseudo"], local_files_only=True,
        torch_dtype=torch.float32, attn_implementation="eager",
    ).to("cuda").eval()
    from research.pilot.frozen_english_normalizer import EnglishTextNormalizer
    english_normalize = EnglishTextNormalizer(json.loads(Path("/root/normalizer.json").read_text()))
    rows = sorted(plan["source_dev_rows"], key=lambda row: row["utt_id"])
    feature_batches = []
    for begin in range(0, len(rows), 8):
        chunk = rows[begin : begin + 8]
        values = []
        for row in chunk:
            value = np.load(
                local_features / receipts[row["utt_id"]]["feature_file"], allow_pickle=False
            )
            if value.shape != (80, 3000) or value.dtype != np.float32 or not np.isfinite(value).all():
                raise ValueError(f"invalid scale tuner source feature: {row['utt_id']}")
            values.append(value)
        feature_batches.append((chunk, np.stack(values)))
    generation = {
        "language": "en", "task": "transcribe", "return_timestamps": False,
        "max_new_tokens": 224, "do_sample": False, "num_beams": 1, "use_cache": True,
    }
    output = Path("/outputs") / plan["output_path"]
    if output.exists():
        raise ValueError(f"output path already exists: {output}")
    candidates = output / "candidates"
    candidates.mkdir(parents=True)
    grid = []
    selected_state = None
    for scale in plan["scales"]:
        corrected, correction = arithmetic.apply_global_correction(
            states["source_real"], states["source_pseudo"], states["target_pseudo"], scale
        )
        model.load_state_dict(corrected, strict=True)
        model.eval()
        predictions = []
        with torch.inference_mode():
            for chunk, values in feature_batches:
                generated = model.generate(
                    input_features=torch.from_numpy(values).to("cuda"), **generation
                )
                hypotheses = processor.batch_decode(
                    generated, skip_special_tokens=True, clean_up_tokenization_spaces=False
                )
                for row, hypothesis, sequence in zip(chunk, hypotheses, generated.cpu().tolist()):
                    reference, hypothesis = row["reference_text"], hypothesis.strip()
                    predictions.append({
                        **{key: row[key] for key in (
                            "utt_id", "speaker_id", "native_language", "prompt_id", "split"
                        )},
                        "reference_text": reference, "prediction": hypothesis,
                        "generated_token_ids": sequence, "scale": scale,
                        "minimal": core.word_counts(
                            core.minimal_normalize(reference), core.minimal_normalize(hypothesis)
                        ),
                        "english": core.word_counts(
                            english_normalize(reference), english_normalize(hypothesis)
                        ),
                    })
        metrics = core.summarize_predictions(predictions)
        label = f"lambda_{scale:.1f}".replace(".", "p")
        prediction_path = candidates / f"{label}_predictions.jsonl"
        prediction_path.write_text("".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in predictions
        ))
        metrics_path = candidates / f"{label}_metrics.json"
        metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")
        grid.append({
            "scale": scale, "correction": correction,
            "english_macro_wer": metrics["english"]["macro_language_wer"],
            "english_micro_wer": metrics["english"]["micro_wer"],
            "prediction_sha256": core.sha256_file(prediction_path),
            "metrics_sha256": core.sha256_file(metrics_path),
        })
        print(f"scale {scale:.1f} macro WER {grid[-1]['english_macro_wer']:.6f}", flush=True)
    selected = min(grid, key=lambda row: (row["english_macro_wer"], row["scale"]))
    selected_state, selected_correction = arithmetic.apply_global_correction(
        states["source_real"], states["source_pseudo"],
        states["target_pseudo"], selected["scale"],
    )
    checkpoint = output / "checkpoint_last"
    checkpoint.mkdir()
    for source in checkpoints["target_pseudo"].iterdir():
        if source.is_file() and source.name not in {"model.safetensors", "training_state.pt"}:
            shutil.copy2(source, checkpoint / source.name)
    save_file(selected_state, str(checkpoint / "model.safetensors"), metadata={
        "format": "pt", "formula": selected_correction["formula"],
        "scale": str(selected["scale"]), "selection": "source_dev_macro_wer",
    })
    checkpoint_sha = core.sha256_file(checkpoint / "model.safetensors")
    report = {
        "status": "completed", "selection_metric": "source_dev_english_macro_wer",
        "target_validation_examples_read": 0, "scales": plan["scales"],
        "uncorrected_lambda_0_source_dev": plan["uncorrected_lambda_0_source_dev"],
        "grid": grid, "selected_scale": selected["scale"],
        "selected_source_dev_english_macro_wer": selected["english_macro_wer"],
        "selected_source_dev_english_micro_wer": selected["english_micro_wer"],
        "selected_correction": selected_correction,
        "input_checkpoint_sha256": plan["checkpoint_sha256"],
        "selected_checkpoint_sha256": checkpoint_sha,
    }
    report_path = output / "scale_selection.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    wall_seconds = time.monotonic() - started
    receipt = {
        "run_id": plan["run_id"], "status": "completed",
        "method": "global_pseudo2real_source_dev_tuned",
        "selection_metric": report["selection_metric"], "selected_scale": selected["scale"],
        "source_dev_utterances_per_scale": len(rows), "scales_evaluated": len(grid),
        "target_validation_examples_read": 0, "staged_source_dev_features": len(staged),
        "selected_checkpoint_sha256": checkpoint_sha,
        "scale_selection_sha256": core.sha256_file(report_path),
        "grid": grid, "gpu": torch.cuda.get_device_name(),
        "wall_seconds": wall_seconds,
        "estimated_resource_cost_usd": wall_seconds * H100_RATE_PER_SECOND,
    }
    (output / "receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    results.commit()
    return json.loads(json.dumps(receipt))


@app.function(
    image=training_image,
    gpu="H100",
    cpu=8,
    memory=32768,
    timeout=600,
    startup_timeout=180,
    retries=0,
    max_containers=2,
    scaledown_window=2,
    secrets=[modal.Secret.from_name("huggingface")],
    volumes={"/target": target_data, "/cache/huggingface": hf_cache},
)
def label_target_with_teacher(plan: dict) -> dict:
    """Generate deterministic large-v3 target labels and confidence scores."""
    import hashlib
    import importlib.metadata
    import io
    import json
    import math
    import time
    from concurrent.futures import ThreadPoolExecutor

    import numpy as np
    import scipy.signal
    import soundfile as sf
    import torch
    from transformers import WhisperForConditionalGeneration, WhisperProcessor
    from transformers.utils import logging as hf_logging

    hf_logging.set_verbosity_error()
    started = time.monotonic()
    if (
        plan["teacher_model"] != "openai/whisper-large-v3"
        or plan["batch_size"] != 32
        or plan["dtype"] != "float16"
        or plan["shards"] != 2
        or plan["shard_index"] not in (0, 1)
    ):
        raise ValueError("target teacher identity, dtype, or batch size changed")
    target_data.reload()
    root = Path("/target/v1")
    manifest_path = root / "manifest.jsonl"
    if hashlib.sha256(manifest_path.read_bytes()).hexdigest() != plan["manifest_sha256"]:
        raise ValueError("target manifest hash mismatch")
    rows = [json.loads(line) for line in manifest_path.read_text().splitlines()]
    if len(rows) != 4800 or any("reference_text" in row for row in rows):
        raise ValueError("target teacher received labeled or incomplete manifest")
    # Striding a duration sort gives each shard the same duration profile.
    decode_rows = sorted(rows, key=lambda row: (row["duration_s"], row["utt_id"]))[
        plan["shard_index"] :: plan["shards"]
    ]
    if len(decode_rows) != 2400:
        raise ValueError("target teacher shard is not exactly half the cohort")
    local_root = Path("/tmp/target-audio-16k")

    def stage(row: dict) -> str:
        source = root / row["audio_path"]
        payload = source.read_bytes()
        if hashlib.sha256(payload).hexdigest() != row["audio_sha256"]:
            raise ValueError(f"target audio hash mismatch: {row['utt_id']}")
        audio, sr = sf.read(io.BytesIO(payload), dtype="float32")
        if audio.ndim != 1 or abs(len(audio) / sr - row["duration_s"]) > 0.02:
            raise ValueError(f"target audio structure mismatch: {row['utt_id']}")
        if sr != 16000:
            divisor = math.gcd(sr, 16000)
            audio = scipy.signal.resample_poly(
                audio, 16000 // divisor, sr // divisor
            ).astype(np.float32)
        target = local_root / f"{row['utt_id']}.npy"
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("wb") as stream:
            np.save(stream, audio, allow_pickle=False)
        return row["utt_id"]

    with ThreadPoolExecutor(max_workers=16) as pool:
        staged = list(pool.map(stage, decode_rows))
    model_id, revision = plan["teacher_model"], plan["teacher_revision"]
    processor = WhisperProcessor.from_pretrained(model_id, revision=revision)
    model = WhisperForConditionalGeneration.from_pretrained(
        model_id, revision=revision, dtype=torch.float16, attn_implementation="sdpa"
    ).to("cuda").eval()
    torch.manual_seed(0)
    torch.set_num_threads(8)
    predictions = []
    for begin in range(0, len(decode_rows), plan["batch_size"]):
        if time.monotonic() - started > 540:
            raise TimeoutError("target teacher stopped before the hard timeout")
        batch = decode_rows[begin : begin + plan["batch_size"]]
        arrays = []
        for row in batch:
            audio = np.load(local_root / f"{row['utt_id']}.npy", allow_pickle=False)
            arrays.append(audio)
        features = processor(
            arrays, sampling_rate=16000, return_tensors="pt", padding="max_length",
            truncation=True, return_attention_mask=True,
        )
        with torch.inference_mode():
            generated = model.generate(
                input_features=features.input_features.to("cuda", dtype=torch.float16),
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
        if len(predictions) % 384 == 0 or len(predictions) == len(decode_rows):
            print(
                f"teacher shard {plan['shard_index']} labeled {len(predictions)}/2400 clips",
                flush=True,
            )
    predictions.sort(key=lambda row: row["utt_id"])
    text = "".join(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n" for row in predictions)
    wall_seconds = time.monotonic() - started
    receipt = {
        "run_id": plan["run_id"], "status": "completed", "model_id": model_id,
        "shard_index": plan["shard_index"], "shards": plan["shards"],
        "model_revision": revision, "dtype": "float16", "batch_size": 32,
        "attention_implementation": "sdpa",
        "batch_order": "duration_then_utt_id_strided_across_two_shards",
        "generation": {"language": "en", "task": "transcribe", "return_timestamps": False,
                       "do_sample": False, "num_beams": 1, "max_new_tokens": 224},
        "rows": len(predictions),
        "manifest_sha256": plan["manifest_sha256"],
        "shard_manifest_sha256": hashlib.sha256(text.encode()).hexdigest(),
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
    return {**receipt, "pseudo_manifest_text": text}


@app.function(
    image=data_image,
    cpu=2,
    memory=8192,
    timeout=120,
    retries=0,
    volumes={"/target": target_data},
)
def persist_target_teacher_artifacts(plan: dict) -> dict:
    """Persist only the already combined, locally validated teacher artifacts."""
    import hashlib
    import json

    pseudo_text = plan["pseudo_manifest_text"]
    audit_text = plan["audit_manifest_text"]
    if hashlib.sha256(pseudo_text.encode()).hexdigest() != plan["pseudo_manifest_sha256"]:
        raise ValueError("combined pseudo-label hash changed during persistence")
    if hashlib.sha256(audit_text.encode()).hexdigest() != plan["audit_manifest_sha256"]:
        raise ValueError("combined audit hash changed during persistence")
    if len(pseudo_text.splitlines()) != 4800 or len(audit_text.splitlines()) != 120:
        raise ValueError("combined target teacher artifact count changed")
    root = Path("/target/v1")
    paths = {
        "pseudo": root / "pseudo_large_v3_seed12.jsonl",
        "audit": root / "audit_120_blind.jsonl",
        "receipt": root / "pseudo_large_v3_seed12.receipt.json",
    }
    if any(path.exists() for path in paths.values()):
        raise ValueError("immutable combined target teacher artifact already exists")
    paths["pseudo"].write_text(pseudo_text)
    paths["audit"].write_text(audit_text)
    paths["receipt"].write_text(json.dumps(plan["receipt"], indent=2, sort_keys=True) + "\n")
    target_data.commit()
    return {
        "pseudo_manifest_sha256": plan["pseudo_manifest_sha256"],
        "audit_manifest_sha256": plan["audit_manifest_sha256"],
        "receipt_sha256": hashlib.sha256(paths["receipt"].read_bytes()).hexdigest(),
        "rows": 4800, "audit_rows": 120,
    }
