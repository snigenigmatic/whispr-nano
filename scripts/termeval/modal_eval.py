"""Bounded Whisper large-v3 inference on the two audited public corpora.

Run with ``.venv/bin/modal run scripts/termeval/modal_eval.py``. Predictions are
append-only in a Modal Volume and copied back by the local entrypoint.
"""

from __future__ import annotations

from pathlib import Path

import modal


ROOT = Path(__file__).resolve().parents[2]
MODEL_ID = "openai/whisper-large-v3"
MODEL_REVISION = "06f233fe06e710322aca913c1bc4249a0d71fce1"
SEED = 20261011
RATE_USD_PER_SECOND = 0.001097 + 4 * 0.0000131 + 16 * 0.00000222
MAX_GPU_SECONDS_PER_DATASET = 4200
MAX_ESTIMATED_TOTAL_GPU_USD = 2 * MAX_GPU_SECONDS_PER_DATASET * RATE_USD_PER_SECOND
ARCHIVES = {
    "hiacc": {
        "url": "https://zenodo.org/api/records/15551669/files/Corpus.zip/content",
        "sha256": "fd83414b3aba28a4ae86c110f4aab0981621f9f3a1b105c161bf2dfd14737ccf",
        "file": "Corpus.zip",
    },
    "mucs_hien": {
        "url": "https://openslr.trmal.net/resources/104/Hindi-English_test.tar.gz",
        "sha256": "93e358b3bf8233a897fcd353c1f4f98fdda6b8c01b7eed17a70c7dd26e984b37",
        "file": "Hindi-English_test.tar.gz",
    },
}

app = modal.App("night1-term-error-pilot")
data_volume = modal.Volume.from_name("night1-term-data-v1", create_if_missing=True)
result_volume = modal.Volume.from_name("night1-term-results-v1", create_if_missing=True)
cache_volume = modal.Volume.from_name("ast-asr-cache", create_if_missing=True)

prep_image = modal.Image.debian_slim(python_version="3.12").pip_install(
    "requests==2.32.5", "numpy==2.2.6", "soundfile==0.13.1"
)
gpu_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch==2.8.0", "transformers==4.57.6", "numpy==2.2.6",
        "soundfile==0.13.1", "huggingface-hub==0.36.2", "accelerate==1.10.1",
    )
    .env({"HF_HOME": "/cache/huggingface", "TOKENIZERS_PARALLELISM": "false"})
    .add_local_file(ROOT / "data/termeval/manifest.jsonl", "/root/project/manifest.jsonl")
)


@app.function(image=prep_image, cpu=4, memory=8192, timeout=1800,
              volumes={"/data": data_volume})
def prepare(dataset: str, spec: dict) -> dict:
    import hashlib
    import json
    import os
    import tarfile
    import zipfile

    import requests
    import soundfile as sf

    root = Path("/data") / dataset
    root.mkdir(parents=True, exist_ok=True)
    archive = root / spec["file"]
    if not archive.exists():
        partial = root / (spec["file"] + ".partial")
        have = partial.stat().st_size if partial.exists() else 0
        headers = {"Range": f"bytes={have}-"} if have else {}
        with requests.get(spec["url"], headers=headers, stream=True, timeout=(30, 120)) as response:
            response.raise_for_status()
            if have and response.status_code != 206:
                raise RuntimeError("download resume refused by server")
            with partial.open("ab") as handle:
                for chunk in response.iter_content(1 << 20):
                    if chunk:
                        handle.write(chunk)
        os.replace(partial, archive)
    digest = hashlib.sha256()
    with archive.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    if digest.hexdigest() != spec["sha256"]:
        raise RuntimeError(f"{dataset} archive checksum mismatch")
    marker = root / ".prepared"
    if not marker.exists():
        if dataset == "hiacc":
            with zipfile.ZipFile(archive) as zf:
                for name in zf.namelist():
                    target = (root / name).resolve()
                    if not target.is_relative_to(root.resolve()):
                        raise ValueError("archive path escapes destination")
                zf.extractall(root)
        else:
            with tarfile.open(archive) as tf:
                tf.extractall(root, filter="data")
            tx = root / "test" / "transcripts"
            recordings = dict(line.strip().split(maxsplit=1) for line in (tx / "wav.scp").open())
            clips = root / "clips"
            clips.mkdir(exist_ok=True)
            for line in (tx / "segments").open():
                utt, recording, start_s, end_s = line.strip().split()
                out = clips / f"{utt}.wav"
                if out.exists():
                    continue
                source = root / "test" / recordings[recording]
                start, end = float(start_s), float(end_s)
                with sf.SoundFile(source) as inp:
                    if inp.samplerate != 16000:
                        raise ValueError("unexpected MUCS sample rate")
                    inp.seek(round(start * 16000))
                    audio = inp.read(round((end - start) * 16000), dtype="float32", always_2d=True)
                sf.write(out, audio.mean(axis=1), 16000, subtype="PCM_16")
        marker.write_text(json.dumps({"dataset": dataset, "archive_sha256": digest.hexdigest()}))
    data_volume.commit()
    result = {"dataset": dataset, "archive_sha256": digest.hexdigest(), "prepared": True}
    print(json.dumps({"event": "data_prepared", **result}), flush=True)
    return result


@app.function(image=gpu_image, gpu="H100", cpu=4, memory=16384,
              timeout=MAX_GPU_SECONDS_PER_DATASET + 120, max_containers=1, retries=0,
              volumes={"/data": data_volume, "/results": result_volume,
                       "/cache/huggingface": cache_volume})
def decode(dataset: str) -> dict:
    import hashlib
    import json
    import random
    import time

    import numpy as np
    import soundfile as sf
    import torch
    import transformers
    from transformers import AutoProcessor, WhisperForConditionalGeneration, pipeline

    started = time.monotonic()
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if not torch.cuda.is_available():
        raise RuntimeError("H100 CUDA unavailable")
    torch.set_num_threads(4)
    manifest_path = Path("/root/project/manifest.jsonl")
    manifest_hash = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    rows = []
    for line in manifest_path.open():
        row = json.loads(line)
        if row["dataset"] == dataset:
            rows.append(row)
    outdir = Path("/results") / dataset
    outdir.mkdir(parents=True, exist_ok=True)
    output = outdir / "predictions.jsonl"
    completed = set()
    if output.exists():
        for line in output.open():
            try:
                completed.add(json.loads(line)["utt_id"])
            except json.JSONDecodeError:
                raise RuntimeError("existing prediction cache has a truncated row")
    config = {"dataset": dataset, "model_id": MODEL_ID, "model_revision": MODEL_REVISION,
              "language": "hi", "task": "transcribe", "num_beams": 1,
              "dtype": "float16", "chunk_length_s": 30, "batch_size": 8,
              "seed": SEED, "transformers": transformers.__version__,
              "torch": torch.__version__, "manifest_sha256": manifest_hash,
              "maximum_gpu_seconds": MAX_GPU_SECONDS_PER_DATASET,
              "estimated_upper_bound_usd": MAX_GPU_SECONDS_PER_DATASET * RATE_USD_PER_SECOND}
    (outdir / "config.json").write_text(json.dumps(config, indent=2))
    print(json.dumps({"event": "decode_config", **config, "cached": len(completed)}), flush=True)
    processor = AutoProcessor.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
    model = WhisperForConditionalGeneration.from_pretrained(
        MODEL_ID, revision=MODEL_REVISION, torch_dtype=torch.float16,
        attn_implementation="sdpa"
    ).to("cuda").eval()
    asr = pipeline("automatic-speech-recognition", model=model, tokenizer=processor.tokenizer,
                   feature_extractor=processor.feature_extractor, device=0,
                   torch_dtype=torch.float16, chunk_length_s=30)
    pending = [r for r in rows if r["utt_id"] not in completed]
    n_chunks = sum(r["duration_s"] > 30 for r in rows)
    processed = 0
    with output.open("a") as handle:
        for start in range(0, len(pending), 8):
            if time.monotonic() - started > MAX_GPU_SECONDS_PER_DATASET - 120:
                break
            batch = pending[start:start + 8]
            audios = []
            for row in batch:
                marker = f"/data/{dataset}/"
                if marker not in row["audio_path"]:
                    raise ValueError(f"manifest audio path lacks {marker}")
                local_tail = Path(row["audio_path"].split(marker, 1)[1])
                remote_audio = Path("/data") / dataset / local_tail
                audio, sr = sf.read(remote_audio, dtype="float32")
                if sr != 16000 or audio.ndim != 1:
                    raise ValueError(f"invalid audio format: {remote_audio}")
                audios.append({"raw": audio, "sampling_rate": sr})
            outputs = asr(audios, batch_size=8, generate_kwargs={
                "language": "hi", "task": "transcribe", "num_beams": 1,
                "do_sample": False, "return_timestamps": False,
            })
            for row, pred in zip(batch, outputs):
                record = {"dataset": dataset, "utt_id": row["utt_id"],
                          "system": "whisper-large-v3", "config_hash": hashlib.sha256(
                              json.dumps(config, sort_keys=True).encode()).hexdigest(),
                          "model_revision": MODEL_REVISION, "text": pred["text"].strip()}
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()
            processed += len(batch)
            if processed % 80 == 0:
                result_volume.commit()
                print(json.dumps({"event": "decode_progress", "dataset": dataset,
                                  "processed_this_run": processed, "total_cached": len(completed) + processed,
                                  "target": len(rows), "elapsed_seconds": time.monotonic() - started}), flush=True)
    result_volume.commit()
    cache_volume.commit()
    elapsed = time.monotonic() - started
    summary = {"dataset": dataset, "processed_this_run": processed,
               "total_cached": len(completed) + processed, "target": len(rows),
               "longer_than_30s": n_chunks, "gpu_wall_seconds": elapsed,
               "estimated_function_resource_usd": elapsed * RATE_USD_PER_SECOND,
               "provider_bill_verified": False, "gpu": torch.cuda.get_device_name(0),
               "complete": len(completed) + processed == len(rows)}
    (outdir / "receipt.json").write_text(json.dumps(summary, indent=2))
    result_volume.commit()
    print(json.dumps({"event": "decode_summary", **summary}), flush=True)
    return summary


@app.local_entrypoint()
def main():
    import json
    from subprocess import run

    if MAX_ESTIMATED_TOTAL_GPU_USD > 10.0:
        raise RuntimeError("pre-flight maximum GPU cost exceeds the $10 pilot guardrail")
    print(json.dumps({"event": "preflight", "max_gpu_seconds_per_dataset": MAX_GPU_SECONDS_PER_DATASET,
                      "max_estimated_gpu_usd": MAX_ESTIMATED_TOTAL_GPU_USD,
                      "pricing_source": "https://modal.com/pricing"}), flush=True)
    for dataset in ("hiacc", "mucs_hien"):
        prepare.remote(dataset, ARCHIVES[dataset])
        summary = decode.remote(dataset)
        if not summary["complete"]:
            raise RuntimeError(f"{dataset} partial; cache preserved for a bounded resume")
        target = ROOT / "results" / "night1" / f"predictions_{dataset}.jsonl"
        target.parent.mkdir(parents=True, exist_ok=True)
        run([str(ROOT / ".venv/bin/modal"), "volume", "get", "night1-term-results-v1",
             f"/{dataset}/predictions.jsonl", str(target), "--force"], check=True)
        receipt = ROOT / "results" / "night1" / f"receipt_{dataset}.json"
        run([str(ROOT / ".venv/bin/modal"), "volume", "get", "night1-term-results-v1",
             f"/{dataset}/receipt.json", str(receipt), "--force"], check=True)
