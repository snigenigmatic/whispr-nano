"""Multi-system Whisper-family fairness audit on Svarah.

Reuses this repo's own authoritative language-family taxonomy
(`ast_asr.taxonomy.SVARAH_LANGUAGE_FAMILIES`) and WER-scoring convention
(`ast_asr.metrics.normalize_for_wer` / `word_edit_counts`) so numbers stay
comparable with every other Svarah number in this project. Deliberately does
NOT depend on this repo's LoRA/policy training code (`modeling.py`,
`whisper_policy.py`) -- this is a zero-shot, no-adapter audit of public
checkpoints, so model loading is a plain `WhisperForConditionalGeneration` /
`WhisperProcessor` pair, kept self-contained and easy to extend with a new
model id.

Usage:
    modal run audit/modal_whisper_family_audit.py::smoke --model-id openai/whisper-tiny
    modal run audit/modal_whisper_family_audit.py::run --model-id openai/whisper-tiny
    modal run audit/modal_whisper_family_audit.py::run --model-id openai/whisper-small
    modal run audit/modal_whisper_family_audit.py::run --model-id openai/whisper-large-v3
    modal run audit/modal_whisper_family_audit.py::run --model-id distil-whisper/distil-large-v3

Run `smoke` for a new model_id first, always. Only run `run` after `smoke`
succeeds cleanly for that model.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import modal

# ---------------------------------------------------------------------------
# App, image, volume, secret
# ---------------------------------------------------------------------------

app = modal.App("svarah-whisper-family-audit")

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
hf_secret = modal.Secret.from_name("huggingface")  # ai4bharat/svarah is gated

GPU_TYPE = "T4"
GPU_HOURLY_USD = 0.59  # Modal's published T4 rate, https://modal.com/pricing
HARD_TIMEOUT_S = 5400  # 90 min unconditional backstop per model
LOCAL_RESULTS_DIR = Path(__file__).parent
DEFAULT_BATCH_SIZE = 4
DEFAULT_MAX_NEW_TOKENS = 200  # Svarah utterances run up to ~25s
KNOWN_SVARAH_TEST_SIZE = 6656


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
    """Handle both the modern torchcodec-backed Audio feature and the
    legacy dict-of-array format across `datasets` versions."""
    import numpy as np

    if hasattr(value, "get_all_samples"):
        samples = value.get_all_samples()
        return samples.data.mean(dim=0).numpy().astype(np.float32), int(samples.sample_rate)
    arr = np.asarray(value["array"], dtype=np.float32)
    return arr, int(value["sampling_rate"])


# ---------------------------------------------------------------------------
# Remote GPU function: dataset -> greedy transcripts (the only GPU-bound step)
# ---------------------------------------------------------------------------


@app.function(
    image=image,
    gpu=GPU_TYPE,
    timeout=HARD_TIMEOUT_S,
    scaledown_window=15,
    volumes={HF_CACHE_PATH: hf_cache_volume},
    secrets=[hf_secret],
)
def transcribe_svarah(
    model_id: str,
    num_utterances: int,
    seed: int,
    batch_size: int,
    max_new_tokens: int,
    max_cost_usd: float,
) -> dict:
    import numpy as np
    import torch
    from datasets import Audio, load_dataset
    from transformers import WhisperForConditionalGeneration, WhisperProcessor

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32  # fp16 sampling/generate() has a known CUDA NaN failure mode on T4; stay fp32 throughout

    t0 = time.monotonic()
    processor = WhisperProcessor.from_pretrained(model_id)
    model = WhisperForConditionalGeneration.from_pretrained(model_id, dtype=dtype).to(device)
    model.eval()
    load_s = time.monotonic() - t0
    print(f"[audit:{model_id}] loaded in {load_s:.1f}s on {device} "
          f"({'fast -- cache reused' if load_s < 90 else 'slow -- likely first download for this model'})", flush=True)

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
    print(f"[audit:{model_id}] sampling {len(idx)}/{n_total} utterances (seed={seed})", flush=True)

    planned = list(idx)
    n_chunks_total = -(-len(planned) // batch_size)
    warmup_chunks = min(3, n_chunks_total)
    rows: list[dict] = []

    t2 = time.monotonic()
    chunk_i = 0
    n_chunks_planned = n_chunks_total
    while chunk_i < n_chunks_planned:
        start = chunk_i * batch_size
        chunk_idx = planned[start:start + batch_size]
        batch_rows = [ds[int(i)] for i in chunk_idx]
        audios = [_decode_audio(r["audio_filepath"]) for r in batch_rows]
        arrays = [a for a, _ in audios]

        inputs = processor(
            arrays, sampling_rate=16_000, return_tensors="pt",
            padding="max_length", truncation=True, return_attention_mask=True,
        )
        input_features = inputs.input_features.to(device=device, dtype=dtype)
        attention_mask = inputs.attention_mask.to(device)
        with torch.no_grad():
            generated = model.generate(
                input_features=input_features,
                attention_mask=attention_mask,
                do_sample=False,
                num_beams=1,
                language="en",
                task="transcribe",
                max_new_tokens=max_new_tokens,
            )
        hyps = processor.batch_decode(generated, skip_special_tokens=True)
        for row, hyp in zip(batch_rows, hyps):
            rows.append({
                "utt_id": f"svarah_{chunk_idx[batch_rows.index(row)]:05d}",
                "primary_language": row["primary_language"],
                "gender": row.get("gender", "Unknown"),
                "duration_s": row.get("duration"),
                "reference_text": row["text"],
                "hypothesis_text": hyp.strip(),
            })
        chunk_i += 1

        if chunk_i % 25 == 0 or chunk_i == n_chunks_planned:
            elapsed = time.monotonic() - t2
            print(f"[audit:{model_id}] chunk {chunk_i}/{n_chunks_planned} ({elapsed:.0f}s elapsed, "
                  f"{len(rows)} utterances done)", flush=True)

        if chunk_i == warmup_chunks:
            measured_rate = (time.monotonic() - t2) / chunk_i
            spent_so_far = (time.monotonic() - t2) * (GPU_HOURLY_USD / 3600)
            remaining_budget = max_cost_usd - spent_so_far
            rate_per_chunk_usd = measured_rate * (GPU_HOURLY_USD / 3600)
            affordable_chunks = int(remaining_budget / rate_per_chunk_usd) if rate_per_chunk_usd > 0 else n_chunks_planned
            new_plan = min(n_chunks_planned, chunk_i + affordable_chunks)
            if new_plan < n_chunks_planned:
                print(f"[audit:{model_id}] cost-guardrail (measured): {measured_rate:.2f}s/chunk -> "
                      f"capping at {new_plan}/{n_chunks_planned} chunks to stay within ${max_cost_usd:.2f}", flush=True)
                n_chunks_planned = new_plan

    transcribe_s = time.monotonic() - t2
    total_wall_s = load_s + dataset_load_s + transcribe_s
    return {
        "model_id": model_id,
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


# ---------------------------------------------------------------------------
# Local (CPU-side) scoring: this repo's own taxonomy + WER convention
# ---------------------------------------------------------------------------


def _score_rows(rows: list[dict]) -> list[dict]:
    """Score with this repo's own WER arithmetic and family taxonomy, NOT
    Whisper's EnglishTextNormalizer, so numbers stay comparable with every
    other number already reported in this project (see docs/
    HANDOFF_BRAINSTORM_2026-08-25.md section 10's normalization warning)."""
    from ast_asr.metrics import normalize_for_wer, word_edit_counts
    from ast_asr.taxonomy import SVARAH_LANGUAGE_FAMILIES

    scored = []
    for r in rows:
        language = str(r["primary_language"]).strip().title()
        family = SVARAH_LANGUAGE_FAMILIES.get(language)
        if family is None:
            raise ValueError(f"unknown primary_language {language!r}; check ast_asr.taxonomy")
        counts = word_edit_counts(r["reference_text"], r["hypothesis_text"])
        scored.append({
            **r,
            "family": family,
            "reference_normalized": normalize_for_wer(r["reference_text"]),
            "hypothesis_normalized": normalize_for_wer(r["hypothesis_text"]),
            "wer": counts.wer,
            "n_sub": counts.substitutions,
            "n_del": counts.deletions,
            "n_ins": counts.insertions,
            "ref_len_words": counts.reference_words,
        })
    return scored


def _per_family_wer(rows: list[dict]) -> dict[str, float]:
    totals: dict[str, list[int]] = {}
    for r in rows:
        n_err = r["n_sub"] + r["n_del"] + r["n_ins"]
        totals.setdefault(r["family"], [0, 0])
        totals[r["family"]][0] += n_err
        totals[r["family"]][1] += r["ref_len_words"]
    return {f: (err / ref if ref else float("nan")) for f, (err, ref) in totals.items()}


def _fit_poisson_drop_in_deviance(rows: list[dict]) -> dict:
    import numpy as np
    import pandas as pd
    import statsmodels.api as sm
    import statsmodels.formula.api as smf
    from scipy.stats import chi2

    df = pd.DataFrame(rows)
    df["n_errors"] = df["n_sub"] + df["n_del"] + df["n_ins"]
    offset = np.log(df["ref_len_words"].astype(float).clip(lower=1))

    null_model = smf.glm("n_errors ~ 1", data=df, family=sm.families.Poisson(), offset=offset).fit()
    alt_model = smf.glm("n_errors ~ C(family)", data=df, family=sm.families.Poisson(), offset=offset).fit()

    deviance_diff = float(null_model.deviance - alt_model.deviance)
    df_diff = int(null_model.df_resid - alt_model.df_resid)
    return {
        "deviance_diff": deviance_diff,
        "df_diff": df_diff,
        "p_value": float(chi2.sf(deviance_diff, df_diff)),
    }


def _write_report(remote_result: dict, scored_rows: list[dict]) -> dict:
    model_id = remote_result["model_id"]
    model_tag = model_id.split("/")[-1]
    total_err = sum(r["n_sub"] + r["n_del"] + r["n_ins"] for r in scored_rows)
    total_ref = sum(r["ref_len_words"] for r in scored_rows)
    overall_wer = total_err / total_ref if total_ref else float("nan")
    per_family = _per_family_wer(scored_rows)
    family_n = {f: sum(1 for r in scored_rows if r["family"] == f) for f in per_family}
    cost_usd = remote_result["total_wall_s"] * (GPU_HOURLY_USD / 3600)

    result = {
        "model_id": model_id,
        "overall_wer": overall_wer,
        "n_utterances": len(scored_rows),
        "per_family_wer": per_family,
        "family_n": family_n,
        "cost_usd": cost_usd,
        "wall_seconds": remote_result["total_wall_s"],
    }
    if len(per_family) >= 2 and len(scored_rows) >= 10:
        poisson = _fit_poisson_drop_in_deviance(scored_rows)
        result["delta_dp_pp"] = (max(per_family.values()) - min(per_family.values())) * 100
        result["poisson"] = poisson
        result["interpretation"] = "structural" if poisson["p_value"] < 0.05 else "not distinguishable from sampling noise"

    LOCAL_RESULTS_DIR.mkdir(exist_ok=True)
    csv_path = LOCAL_RESULTS_DIR / f"results_{model_tag}_svarah_clean.csv"
    _write_csv(csv_path, scored_rows)
    json_path = LOCAL_RESULTS_DIR / f"summary_{model_tag}_svarah.json"
    json_path.write_text(json.dumps(result, indent=2))
    print(f"\nWrote {csv_path} ({len(scored_rows)} rows)")
    print(f"Wrote {json_path}")
    return result


def _write_csv(path: Path, rows: list[dict]) -> None:
    import csv

    columns = [
        "utt_id", "primary_language", "family", "gender", "duration_s",
        "reference_text", "reference_normalized", "hypothesis_text", "hypothesis_normalized",
        "wer", "n_sub", "n_del", "n_ins", "ref_len_words",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for r in rows:
            writer.writerow({c: r.get(c) for c in columns})


def _print_report(result: dict, remote_result: dict) -> None:
    print(f"\n=== {result['model_id']} report ===")
    print(f"gpu={remote_result['gpu_type']} load={remote_result['load_s']:.1f}s "
          f"dataset_load={remote_result['dataset_load_s']:.1f}s "
          f"transcribe={remote_result['transcribe_s']:.1f}s "
          f"total={remote_result['total_wall_s']:.1f}s cost=${result['cost_usd']:.4f}")
    print(f"utterances: completed={result['n_utterances']} of {remote_result['n_total_dataset']} total in dataset")
    print(f"overall WER: {result['overall_wer'] * 100:.2f}%")
    print("per-family WER:")
    for family in sorted(result["per_family_wer"]):
        print(f"  {family:14s} WER={result['per_family_wer'][family] * 100:6.2f}%  "
              f"n_utterances={result['family_n'][family]}")
    if "poisson" in result:
        print(f"ΔDP: {result['delta_dp_pp']:.2f}pp  "
              f"Poisson p={result['poisson']['p_value']:.4g} ({result['interpretation']})")
    else:
        print("(too few utterances/families for a meaningful Poisson fit)")


# ---------------------------------------------------------------------------
# Local entrypoints
# ---------------------------------------------------------------------------


@app.local_entrypoint()
def smoke(model_id: str = "openai/whisper-tiny", num_utterances: int = 32, max_cost_usd: float = 0.20, seed: int = 0):
    """`modal run audit/modal_whisper_family_audit.py::smoke --model-id <id>`
    -- run this FIRST for any new model_id, before `run`."""
    n_chunks = -(-num_utterances // DEFAULT_BATCH_SIZE)
    _cost_guard(n_chunks * 10.0, max_cost_usd, "assumed")

    print(f"Launching smoke run for {model_id} on Modal ({GPU_TYPE}, {num_utterances} utterances)...")
    remote_result = transcribe_svarah.remote(
        model_id=model_id, num_utterances=num_utterances, seed=seed,
        batch_size=DEFAULT_BATCH_SIZE, max_new_tokens=DEFAULT_MAX_NEW_TOKENS, max_cost_usd=max_cost_usd,
    )
    scored = _score_rows(remote_result["rows"])
    result = _write_report(remote_result, scored)
    _print_report(result, remote_result)
    print(f"\nsmoke OK for {model_id}: pipeline ran end-to-end without error.")


@app.local_entrypoint()
def run(model_id: str = "openai/whisper-tiny", num_utterances: int = 2500, max_cost_usd: float = 2.00, seed: int = 0):
    """`modal run audit/modal_whisper_family_audit.py::run --model-id <id>`
    -- the real pass. Only run this after `smoke` succeeds for the same
    model_id. `num_utterances=2500` is a random subsample of Svarah's 6,656
    total (pass --num-utterances 0 for the full corpus, ~2-3x the cost)."""
    n_for_estimate = num_utterances if num_utterances > 0 else KNOWN_SVARAH_TEST_SIZE
    n_chunks = -(-n_for_estimate // DEFAULT_BATCH_SIZE)
    _cost_guard(n_chunks * 10.0, max_cost_usd, "assumed")

    print(f"Launching full run for {model_id} on Modal ({GPU_TYPE}, {num_utterances} utterances requested)...")
    remote_result = transcribe_svarah.remote(
        model_id=model_id, num_utterances=num_utterances, seed=seed,
        batch_size=DEFAULT_BATCH_SIZE, max_new_tokens=DEFAULT_MAX_NEW_TOKENS, max_cost_usd=max_cost_usd,
    )
    scored = _score_rows(remote_result["rows"])
    result = _write_report(remote_result, scored)
    _print_report(result, remote_result)
