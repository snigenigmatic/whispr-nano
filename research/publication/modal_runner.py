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
