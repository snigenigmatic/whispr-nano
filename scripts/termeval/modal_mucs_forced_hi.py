"""Bounded full MUCS forced-Hindi decode, after the matched language probe.

The full auto-language run continues independently. This post-QC decode setting
is reported alongside it; no output is silently replaced. Run with:
    .venv/bin/modal run scripts/termeval/modal_mucs_forced_hi.py
"""

from __future__ import annotations

from pathlib import Path

import modal


_module_file = Path(__file__).resolve()
ROOT = _module_file.parents[2] if len(_module_file.parents) > 2 else Path("/root/project")
SOURCE_MANIFEST = (ROOT / "data/termeval/manifest.jsonl"
                   if (ROOT / "data/termeval/manifest.jsonl").exists()
                   else Path("/root/project/manifest.jsonl"))
MODEL_ID = "openai/whisper-large-v3"
MODEL_REVISION = "06f233fe06e710322aca913c1bc4249a0d71fce1"
SEED = 20261011
MAX_GPU_SECONDS = 6000
RATE_USD_PER_SECOND = 0.001097 + 4 * 0.0000131 + 16 * 0.00000222
VOLUME = "night1-term-results-v1"
REMOTE_DIR = "/mucs_hien_forced_hi_greedy"

app = modal.App("night1-mucs-forced-hi")
data_volume = modal.Volume.from_name("night1-term-data-v1")
result_volume = modal.Volume.from_name(VOLUME)
cache_volume = modal.Volume.from_name("ast-asr-cache")
image = (modal.Image.debian_slim(python_version="3.12")
         .pip_install("torch==2.8.0", "transformers==4.57.6", "numpy==2.2.6",
                      "soundfile==0.13.1", "huggingface-hub==0.36.2", "accelerate==1.10.1")
         .env({"HF_HOME": "/cache/huggingface", "TOKENIZERS_PARALLELISM": "false"})
         .add_local_file(SOURCE_MANIFEST, "/root/project/manifest.jsonl"))


@app.function(image=image, gpu="H100", cpu=4, memory=16384, timeout=MAX_GPU_SECONDS + 120,
              max_containers=1, retries=0,
              volumes={"/data": data_volume, "/results": result_volume,
                       "/cache/huggingface": cache_volume})
def decode_full() -> dict:
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
    manifest_file = Path("/root/project/manifest.jsonl")
    manifest = {r["utt_id"]: r for r in map(json.loads, manifest_file.open())}
    ids = [utt for utt, row in manifest.items() if row["dataset"] == "mucs_hien"]
    if len(ids) != len(set(ids)) or len(ids) != 3136:
        raise ValueError("full MUCS manifest is not 3136 unique utterances")
    config = {"dataset": "mucs_hien", "model_id": MODEL_ID,
              "model_revision": MODEL_REVISION, "condition_id": "forced_hi_greedy",
              "language": "hi", "task": "transcribe", "num_beams": 1,
              "dtype": "float16", "chunk_length_s": 30, "batch_size": 8,
              "seed": SEED, "torch": torch.__version__,
              "transformers": transformers.__version__,
              "manifest_sha256": hashlib.sha256(manifest_file.read_bytes()).hexdigest(),
              "maximum_gpu_seconds": MAX_GPU_SECONDS}
    config_hash = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    outdir = Path("/results") / REMOTE_DIR.lstrip("/")
    outdir.mkdir(parents=True, exist_ok=True)
    config_path = outdir / "config.json"
    if config_path.exists() and json.loads(config_path.read_text()) != config:
        raise ValueError("existing full-decode cache has a different config")
    config_path.write_text(json.dumps(config, indent=2))
    output = outdir / "predictions.jsonl"
    cached = [json.loads(line) for line in output.open()] if output.exists() else []
    if len(cached) != len({p["utt_id"] for p in cached}) or any(p["config_hash"] != config_hash for p in cached):
        raise ValueError("duplicate or mixed forced-Hindi cache")
    completed = {p["utt_id"] for p in cached}
    print(json.dumps({"event": "forced_hi_config", **config, "cached": len(completed)}), flush=True)
    processor = AutoProcessor.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
    model = WhisperForConditionalGeneration.from_pretrained(
        MODEL_ID, revision=MODEL_REVISION, torch_dtype=torch.float16,
        attn_implementation="sdpa").to("cuda").eval()
    asr = pipeline("automatic-speech-recognition", model=model,
                   tokenizer=processor.tokenizer, feature_extractor=processor.feature_extractor,
                   device=0, torch_dtype=torch.float16, chunk_length_s=30)
    pending = [manifest[utt] for utt in ids if utt not in completed]
    processed = 0
    with output.open("a") as handle:
        for start in range(0, len(pending), 8):
            if time.monotonic() - started > MAX_GPU_SECONDS - 120:
                break
            batch = pending[start:start + 8]
            audios = []
            for row in batch:
                marker = "/data/mucs_hien/"
                if marker not in row["audio_path"]:
                    raise ValueError("MUCS audio path mismatch")
                tail = Path(row["audio_path"].split(marker, 1)[1])
                audio, sr = sf.read(Path("/data/mucs_hien") / tail, dtype="float32")
                if sr != 16000 or audio.ndim != 1:
                    raise ValueError("MUCS audio format mismatch")
                audios.append({"raw": audio, "sampling_rate": sr})
            outputs = asr(audios, batch_size=8, generate_kwargs={
                "language": "hi", "task": "transcribe", "num_beams": 1,
                "do_sample": False, "return_timestamps": False})
            for row, prediction in zip(batch, outputs):
                handle.write(json.dumps({"dataset": "mucs_hien", "utt_id": row["utt_id"],
                                         "system": "whisper-large-v3", "condition_id": "forced_hi_greedy",
                                         "config_hash": config_hash, "model_revision": MODEL_REVISION,
                                         "text": prediction["text"].strip()}, ensure_ascii=False) + "\n")
            handle.flush()
            processed += len(batch)
            if processed % 80 == 0:
                result_volume.commit()
                print(json.dumps({"event": "forced_hi_progress", "processed": processed,
                                  "target": len(ids), "elapsed_seconds": time.monotonic() - started}), flush=True)
    result_volume.commit()
    elapsed = time.monotonic() - started
    summary = {"dataset": "mucs_hien", "cached_before": len(completed), "processed_this_run": processed,
               "total_cached": len(completed) + processed, "target": len(ids),
               "gpu_wall_seconds": elapsed,
               "estimated_function_resource_usd": elapsed * RATE_USD_PER_SECOND,
               "provider_bill_verified": False, "gpu": torch.cuda.get_device_name(0),
               "complete": len(completed) + processed == len(ids)}
    (outdir / "receipt.json").write_text(json.dumps(summary, indent=2))
    result_volume.commit()
    print(json.dumps({"event": "forced_hi_summary", **summary}), flush=True)
    return summary


@app.local_entrypoint()
def main():
    from subprocess import run

    if MAX_GPU_SECONDS * RATE_USD_PER_SECOND > 7.20:
        raise RuntimeError("forced-Hindi preflight exceeds $7.20")
    summary = decode_full.remote()
    if not summary["complete"]:
        raise RuntimeError("forced-Hindi decode incomplete; cache preserved")
    out = ROOT / "results/night1"
    for remote, local in (("predictions.jsonl", "predictions_mucs_hien_forced_hi.jsonl"),
                          ("config.json", "config_mucs_hien_forced_hi.json"),
                          ("receipt.json", "receipt_mucs_hien_forced_hi.json")):
        run([str(ROOT / ".venv/bin/modal"), "volume", "get", VOLUME,
             f"{REMOTE_DIR}/{remote}", str(out / local), "--force"], check=True)
