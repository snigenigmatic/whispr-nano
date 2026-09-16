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

    from datasets import Audio, get_dataset_split_names, load_dataset, load_dataset_builder

    os.environ["HF_HOME"] = "/cache/huggingface"
    started = time.monotonic()
    repo_id = "VectorSigma389/spire-sies"
    splits = get_dataset_split_names(repo_id)
    builder = load_dataset_builder(repo_id)
    if "train" not in splits:
        raise RuntimeError(f"SPIRE dataset has no train split: {splits}")
    stream = load_dataset(repo_id, split="train", streaming=True)
    feature_names = sorted(stream.features) if stream.features else []
    # Constructing a streaming row decodes Audio columns even if user code does
    # not index that field. Explicitly disable decode before metadata iteration.
    for name, feature in (stream.features or {}).items():
        if feature.__class__.__name__ == "Audio":
            stream = stream.cast_column(name, Audio(decode=False))
    # Metadata keys vary across mirrors. We inspect values only from identity
    # and demographic fields. Text/reference fields are never dereferenced.
    safe_candidates = (
        "speaker_id", "speaker", "client_id", "accent", "native_language",
        "language", "gender", "uid", "id", "audio_id",
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
        "train_rows": observed,
        "safe_metadata_counts": safe_counts,
        "validation_examples_read": 0,
        "wall_seconds": time.monotonic() - started,
    }
