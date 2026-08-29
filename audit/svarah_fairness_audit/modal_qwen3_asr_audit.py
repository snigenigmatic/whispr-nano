"""Qwen3-ASR audit on Svarah, matching the Whisper-family audit's scoring
convention (ast_asr.taxonomy / ast_asr.metrics) so numbers are directly
comparable across systems.

Qwen3-ASR has a different generation API than Whisper (chat-template-based,
via `processor.apply_transcription_request`), so this is a new script
rather than a parameterization of `modal_whisper_family_audit.py` -- but it
reuses that script's scoring/reporting functions unchanged.

Usage:
    modal run audit/modal_qwen3_asr_audit.py::smoke
    modal run audit/modal_qwen3_asr_audit.py::run
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import modal

sys.path.insert(0, str(Path(__file__).parent))

app = modal.App("svarah-qwen3-asr-audit")

image = (
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
    )
    .env({"HF_HOME": "/cache/huggingface", "HF_HUB_ENABLE_HF_TRANSFER": "0"})
    .add_local_python_source("ast_asr")
)

hf_cache_volume = modal.Volume.from_name("ast-asr-cache", create_if_missing=True)
HF_CACHE_PATH = "/cache/huggingface"
hf_secret = modal.Secret.from_name("huggingface")

GPU_TYPE = "T4"
GPU_HOURLY_USD = 0.59
HARD_TIMEOUT_S = 5400
LOCAL_RESULTS_DIR = Path(__file__).parent
DEFAULT_BATCH_SIZE = 4
DEFAULT_MAX_NEW_TOKENS = 256
KNOWN_SVARAH_TEST_SIZE = 6656
MODEL_ID = "Qwen/Qwen3-ASR-0.6B-hf"


def _cost_guard(seconds: float, max_cost_usd: float, label: str) -> None:
    cost = seconds * GPU_HOURLY_USD / 3600
    print(f"[cost-guardrail:{label}] {seconds:.0f}s -> ${cost:.4f} vs ${max_cost_usd:.2f} budget "
          f"[{'OK' if cost <= max_cost_usd else 'OVER BUDGET'}]", flush=True)
    if cost > max_cost_usd:
        raise RuntimeError(
            f"Projected cost ${cost:.4f} exceeds --max-cost-usd ${max_cost_usd:.2f}. "
            f"Reduce --num-utterances or raise the budget explicitly."
        )


def _decode_audio(value):
    import numpy as np

    if hasattr(value, "get_all_samples"):
        samples = value.get_all_samples()
        return samples.data.mean(dim=0).numpy().astype(np.float32), int(samples.sample_rate)
    arr = np.asarray(value["array"], dtype=np.float32)
    return arr, int(value["sampling_rate"])


@app.function(
    image=image,
    gpu=GPU_TYPE,
    timeout=HARD_TIMEOUT_S,
    scaledown_window=15,
    volumes={HF_CACHE_PATH: hf_cache_volume},
    secrets=[hf_secret],
)
def transcribe_svarah_qwen(
    num_utterances: int,
    seed: int,
    batch_size: int,
    max_new_tokens: int,
    max_cost_usd: float,
) -> dict:
    import tempfile

    import numpy as np
    import soundfile as sf
    import torch
    from datasets import Audio, load_dataset
    from transformers import AutoModelForMultimodalLM, AutoProcessor

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32  # stay consistent with the rest of this audit's fp32-only policy

    t0 = time.monotonic()
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = AutoModelForMultimodalLM.from_pretrained(MODEL_ID, dtype=dtype).to(device)
    model.eval()
    load_s = time.monotonic() - t0
    print(f"[audit:qwen3-asr] loaded in {load_s:.1f}s on {device}", flush=True)

    t1 = time.monotonic()
    ds = load_dataset("ai4bharat/svarah", split="test")
    ds = ds.cast_column("audio_filepath", Audio(sampling_rate=16_000))
    n_total = len(ds)
    rng = np.random.default_rng(seed)
    if num_utterances <= 0 or num_utterances >= n_total:
        idx = np.arange(n_total)
    else:
        idx = np.sort(rng.choice(n_total, size=num_utterances, replace=False))
    dataset_load_s = time.monotonic() - t1
    print(f"[audit:qwen3-asr] sampling {len(idx)}/{n_total} utterances (seed={seed})", flush=True)

    tmp_dir = Path(tempfile.mkdtemp())
    planned = list(idx)
    n_chunks_planned = -(-len(planned) // batch_size)
    warmup_chunks = min(3, n_chunks_planned)
    rows: list[dict] = []

    t2 = time.monotonic()
    chunk_i = 0
    while chunk_i < n_chunks_planned:
        start = chunk_i * batch_size
        chunk_idx = planned[start:start + batch_size]
        batch_rows = [ds[int(i)] for i in chunk_idx]

        wav_paths = []
        for j, r in enumerate(batch_rows):
            arr, sr = _decode_audio(r["audio_filepath"])
            path = tmp_dir / f"clip_{chunk_i}_{j}.wav"
            sf.write(str(path), arr, sr)
            wav_paths.append(str(path))

        inputs = processor.apply_transcription_request(
            audio=wav_paths, language=["English"] * len(wav_paths),
        ).to(device, dtype)
        with torch.no_grad():
            output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        generated_ids = output_ids[:, inputs["input_ids"].shape[1]:]
        hyps = processor.decode(generated_ids, return_format="transcription_only")

        for r, hyp, path in zip(batch_rows, hyps, wav_paths):
            rows.append({
                "utt_id": f"svarah_{chunk_idx[batch_rows.index(r)]:05d}",
                "primary_language": r["primary_language"],
                "gender": r.get("gender", "Unknown"),
                "duration_s": r.get("duration"),
                "reference_text": r["text"],
                "hypothesis_text": (hyp or "").strip(),
            })
            Path(path).unlink(missing_ok=True)
        chunk_i += 1

        if chunk_i % 25 == 0 or chunk_i == n_chunks_planned:
            elapsed = time.monotonic() - t2
            print(f"[audit:qwen3-asr] chunk {chunk_i}/{n_chunks_planned} ({elapsed:.0f}s elapsed, "
                  f"{len(rows)} utterances done)", flush=True)

        if chunk_i == warmup_chunks:
            measured_rate = (time.monotonic() - t2) / chunk_i
            spent_so_far = (time.monotonic() - t2) * (GPU_HOURLY_USD / 3600)
            remaining_budget = max_cost_usd - spent_so_far
            rate_per_chunk_usd = measured_rate * (GPU_HOURLY_USD / 3600)
            affordable_chunks = int(remaining_budget / rate_per_chunk_usd) if rate_per_chunk_usd > 0 else n_chunks_planned
            new_plan = min(n_chunks_planned, chunk_i + affordable_chunks)
            if new_plan < n_chunks_planned:
                print(f"[audit:qwen3-asr] cost-guardrail (measured): {measured_rate:.2f}s/chunk -> "
                      f"capping at {new_plan}/{n_chunks_planned} chunks to stay within ${max_cost_usd:.2f}", flush=True)
                n_chunks_planned = new_plan

    transcribe_s = time.monotonic() - t2
    total_wall_s = load_s + dataset_load_s + transcribe_s
    return {
        "model_id": MODEL_ID,
        "rows": rows,
        "load_s": load_s,
        "dataset_load_s": dataset_load_s,
        "transcribe_s": transcribe_s,
        "total_wall_s": total_wall_s,
        "n_utterances_requested": int(len(idx)),
        "n_utterances_completed": len(rows),
        "n_total_dataset": int(n_total),
        "gpu_type": GPU_TYPE,
    }


@app.local_entrypoint()
def smoke(num_utterances: int = 20, max_cost_usd: float = 0.30, seed: int = 0):
    from modal_whisper_family_audit import _print_report, _score_rows, _write_report

    n_chunks = -(-num_utterances // DEFAULT_BATCH_SIZE)
    _cost_guard(n_chunks * 20.0, max_cost_usd, "assumed")

    print(f"Launching smoke run for {MODEL_ID} on Modal ({GPU_TYPE}, {num_utterances} utterances)...")
    remote_result = transcribe_svarah_qwen.remote(
        num_utterances=num_utterances, seed=seed, batch_size=DEFAULT_BATCH_SIZE,
        max_new_tokens=DEFAULT_MAX_NEW_TOKENS, max_cost_usd=max_cost_usd,
    )
    scored = _score_rows(remote_result["rows"])
    result = _write_report(remote_result, scored)
    _print_report(result, remote_result)
    print(f"\nsmoke OK for {MODEL_ID}: pipeline ran end-to-end without error.")


@app.local_entrypoint()
def run(num_utterances: int = 800, max_cost_usd: float = 1.50, seed: int = 0):
    from modal_whisper_family_audit import _print_report, _score_rows, _write_report

    n_for_estimate = num_utterances if num_utterances > 0 else KNOWN_SVARAH_TEST_SIZE
    n_chunks = -(-n_for_estimate // DEFAULT_BATCH_SIZE)
    _cost_guard(n_chunks * 20.0, max_cost_usd, "assumed")

    print(f"Launching full run for {MODEL_ID} on Modal ({GPU_TYPE}, {num_utterances} utterances requested)...")
    remote_result = transcribe_svarah_qwen.remote(
        num_utterances=num_utterances, seed=seed, batch_size=DEFAULT_BATCH_SIZE,
        max_new_tokens=DEFAULT_MAX_NEW_TOKENS, max_cost_usd=max_cost_usd,
    )
    scored = _score_rows(remote_result["rows"])
    result = _write_report(remote_result, scored)
    _print_report(result, remote_result)
