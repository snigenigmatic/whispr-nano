"""Feasibility-check audit: is `openai/whisper-large-v3`'s WER on Svarah
(Javed et al., "Svarah: Evaluating English ASR Systems on Indian Accents,"
Interspeech 2023, arXiv:2305.15760; `ai4bharat/svarah` on Hugging Face)
different across Indo-Aryan / Dravidian / Sino-Tibetan accent families, and
is any gap real or just sampling noise?

This is a standalone add-on to the on-policy-distillation repo -- a
*different* Modal App from `modal_app.py`'s `onpolicy-whisper-distill`, but
deliberately reusing:
  - the same image-build pattern (ffmpeg + pinned pip versions) as
    `modal_app.py`,
  - the same `onpolicy-distill-hf-cache` Volume name/mount path, so the
    already-downloaded whisper-large-v3 fp32 checkpoint is reused instead
    of re-downloaded,
  - `load_model_bundle` from `onpolicy_distill.distill` (loads the model in
    fp32, exactly as this repo's own history requires -- fp16 sampling in
    HF's Whisper `generate()` hit a documented CUDA NaN bug here before),
  - `_chunks` / `transcribe_greedy` from `onpolicy_distill.evaluate` (small
    batches of ~4, to avoid the OOM this repo's git history shows from
    un-chunked eval batches on a T4),
  - `estimate_cost` / `enforce_budget` / `affordable_steps` from
    `onpolicy_distill.config` for the same two-layer cost guardrail
    (assumed pre-flight estimate, then a measured in-run re-projection)
    used by `modal_app.py`.

Design note on where work happens: the GPU-bound step is audio ->
transcript (this Modal function). WER scoring (jiwer), family lookup, and
the Poisson significance test are cheap CPU work and are done **locally**
in the `local_entrypoint`s below, using the same dev venv already used for
this repo's tests -- this keeps the remote GPU image lean and the stats
code trivially inspectable/re-runnable without spending anything.

Usage:
    modal run audit/opd_bias_extension/modal_svarah_audit.py::smoke   # ~20-40 utterances, validates the whole pipeline
    modal run audit/opd_bias_extension/modal_svarah_audit.py::run     # the real pass, budget-gated at --max-cost-usd (default $3.00)

`ai4bharat/svarah` is a *gated* dataset on the Hugging Face Hub -- the
remote function is given the existing `huggingface` Modal secret (already
present in this account, confirmed via `modal secret list`) so it can
authenticate.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import modal

from onpolicy_distill.config import (
    affordable_steps,
    enforce_budget,
    estimate_cost,
    gpu_second_rate,
)

# Make the sibling `family_mapping.py` importable regardless of cwd (Python
# normally puts the invoked script's own directory on sys.path, but we
# guard explicitly since `modal run` sometimes invokes files unusually).
sys.path.insert(0, str(Path(__file__).parent))

# ---------------------------------------------------------------------------
# App, image, volume, secret
# ---------------------------------------------------------------------------

app = modal.App("svarah-family-audit")  # separate from modal_app.py's "onpolicy-whisper-distill"

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
        "jiwer==4.0.0",
        "numpy==2.5.2",
    )
    .env({"HF_HOME": "/cache/huggingface", "HF_HUB_ENABLE_HF_TRANSFER": "0"})
    .add_local_python_source("onpolicy_distill")
)

hf_cache_volume = modal.Volume.from_name("onpolicy-distill-hf-cache", create_if_missing=True)
HF_CACHE_PATH = "/cache/huggingface"

# `ai4bharat/svarah` is gated; this secret (already configured in this
# Modal account -- see `modal secret list`) supplies HF_TOKEN.
hf_secret = modal.Secret.from_name("huggingface")

MODEL_NAME = "openai/whisper-large-v3"
GPU_TYPE = "T4"
# Unconditional backstop. Sized from the `smoke` run's *measured* throughput
# (~5.6-5.85s per 4-utterance chunk on a T4, fp32, max_new_tokens=200 --
# see the smoke run log) so it's a generous ceiling rather than the thing
# that actually stops `run`: ~2500 utterances / 4 = 625 chunks x ~6s/chunk
# is ~62 minutes; 5400s (90 min) leaves ~45% margin over that estimate.
HARD_TIMEOUT_S = 5400

# Assumed (pre-flight, before any GPU spend) seconds per chunk of
# `batch_size` clips -- conservative prior for greedy decoding of
# whisper-large-v3 in fp32 on a T4; the in-run guardrail below replaces
# this with the *measured* per-chunk rate after a short warmup, exactly as
# `train_arm` does in `onpolicy_distill/distill.py`.
ASSUMED_SECONDS_PER_CHUNK = 10.0

DEFAULT_BATCH_SIZE = 4
DEFAULT_MAX_NEW_TOKENS = 200  # Svarah has utterances up to ~25s; LibriSpeech's 64 (used in distill.py) is too short here
LOCAL_RESULTS_DIR = Path(__file__).parent

# ai4bharat/svarah's `test` split (its only split) size, confirmed by
# directly loading and calling `len(dataset)` on 2026-08-28 -- see
# family_mapping.py's docstring. Used only to size the *local* pre-flight
# cost estimate when `run`'s `--num-utterances` sentinel (<=0) requests the
# full dataset; the remote function always re-derives the true size itself
# via `len(ds)` rather than trusting this constant.
KNOWN_SVARAH_TEST_SIZE = 6656


# ---------------------------------------------------------------------------
# Remote GPU function: dataset -> greedy transcripts (the only part that
# needs a GPU; WER/family/stats happen locally, see bottom of file)
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
    num_utterances: int,
    seed: int,
    batch_size: int,
    max_new_tokens: int,
    max_cost_usd: float,
) -> dict:
    """Load whisper-large-v3 (fp32, trainable=False) and greedily transcribe
    a random sample of `num_utterances` Svarah utterances (or the full
    dataset if `num_utterances` >= its size), in `batch_size`-sized chunks.

    Includes the same *measured*, in-run cost guardrail as `train_arm`
    (distill.py): after a short warmup, re-project total spend from actual
    per-chunk latency and shrink the remaining chunk plan if needed, rather
    than silently overspending or crashing mid-run.
    """
    import numpy as np
    import torch
    from datasets import Audio, load_dataset

    from onpolicy_distill.data import Clip, _decode_audio
    from onpolicy_distill.distill import load_model_bundle
    from onpolicy_distill.evaluate import _chunks, transcribe_greedy

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # fp32 only -- see distill.py / README.md: fp16 sampling in HF's Whisper
    # generate() hit a documented CUDA NaN assertion on a T4 in this repo's
    # own history. Greedy decoding (do_sample=False) is less exposed to
    # that bug than temperature sampling, but we stay consistent with the
    # rest of the repo and don't re-introduce fp16 here.
    dtype = torch.float32

    t0 = time.monotonic()
    bundle = load_model_bundle(MODEL_NAME, "en", "transcribe", device, dtype, trainable=False)
    model_load_s = time.monotonic() - t0
    print(f"[audit] {MODEL_NAME} loaded in {model_load_s:.1f}s on {device} "
          f"({'fast -- cache reused' if model_load_s < 60 else 'SLOW -- likely re-downloaded, check the volume'})",
          flush=True)

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
    print(f"[audit] ai4bharat/svarah loaded ({n_total} total utterances) in {dataset_load_s:.1f}s; "
          f"sampling {len(idx)} (seed={seed})", flush=True)

    clips: list[Clip] = []
    meta: list[dict] = []
    for i in idx:
        row = ds[int(i)]
        array, sr = _decode_audio(row["audio_filepath"])
        clip_id = f"svarah_{int(i):05d}"
        clips.append(Clip(clip_id=clip_id, audio=array, sampling_rate=sr, reference_text=row["text"]))
        meta.append({
            "utt_id": clip_id,
            "primary_language": row["primary_language"],
            "native_place_state": row["native_place_state"],
            "gender": row["gender"],
            "duration_s": row["duration"],
        })

    chunks = _chunks(clips, batch_size)
    planned_chunks = len(chunks)
    warmup_chunks = min(3, planned_chunks)
    hypotheses: list[str] = []

    t2 = time.monotonic()
    chunk_i = 0
    while chunk_i < planned_chunks:
        chunk = chunks[chunk_i]
        t_chunk0 = time.monotonic()
        preds = transcribe_greedy(
            bundle, chunk, language="en", task="transcribe",
            max_new_tokens=max_new_tokens, batch_size=len(chunk),
        )
        dt = time.monotonic() - t_chunk0
        hypotheses.extend(preds)
        chunk_i += 1

        if chunk_i % 25 == 0 or chunk_i == planned_chunks:
            elapsed = time.monotonic() - t2
            print(f"[audit] chunk {chunk_i}/{planned_chunks} ({dt:.2f}s this chunk, "
                  f"{elapsed:.0f}s elapsed, {len(hypotheses)} utterances done)", flush=True)

        if chunk_i == warmup_chunks:
            measured_rate = (time.monotonic() - t2) / chunk_i
            spent_so_far = (time.monotonic() - t2) * gpu_second_rate(GPU_TYPE)
            remaining_budget = max_cost_usd - spent_so_far
            affordable = affordable_steps(GPU_TYPE, measured_rate, remaining_budget)
            new_plan = min(planned_chunks, chunk_i + affordable)
            print(f"[audit] cost-guardrail (measured): {measured_rate:.2f}s/chunk -> "
                  f"{new_plan}/{planned_chunks} chunks fit ${max_cost_usd:.2f} budget", flush=True)
            if new_plan < planned_chunks:
                planned_chunks = new_plan

    transcribe_s = time.monotonic() - t2

    clips = clips[: len(hypotheses)]
    meta = meta[: len(hypotheses)]
    rows = [
        {**m, "reference_text": c.reference_text, "hypothesis_text": hyp}
        for c, m, hyp in zip(clips, meta, hypotheses)
    ]

    total_wall_s = model_load_s + dataset_load_s + transcribe_s
    return {
        "rows": rows,
        "model_load_s": model_load_s,
        "dataset_load_s": dataset_load_s,
        "transcribe_s": transcribe_s,
        "total_wall_s": total_wall_s,
        "n_utterances_requested": int(len(idx)),
        "n_utterances_completed": len(rows),
        "n_total_dataset": int(n_total),
        "gpu_type": GPU_TYPE,
        "stopped_early_for_budget": len(rows) < len(idx),
    }


# ---------------------------------------------------------------------------
# Local (CPU-side) scoring: family lookup, per-utterance jiwer detail, and
# the Poisson drop-in-deviance significance test. No GPU/Modal needed here.
# ---------------------------------------------------------------------------


def _score_rows(rows: list[dict]) -> list[dict]:
    """Normalize text (Whisper's own EnglishTextNormalizer, matching
    evaluate.py's WER protocol) and compute per-utterance WER + detailed
    substitution/deletion/insertion/reference-length counts via
    `jiwer.process_words` (jiwer's aggregate `jiwer.wer` only returns a
    corpus-level scalar, not the per-utterance breakdown the Poisson model
    needs)."""
    import jiwer
    from transformers.models.whisper.english_normalizer import EnglishTextNormalizer

    from family_mapping import get_family

    norm = EnglishTextNormalizer({})
    scored = []
    for r in rows:
        ref_norm = norm(r["reference_text"]) or " "
        hyp_norm = norm(r["hypothesis_text"]) or " "
        wo = jiwer.process_words(ref_norm, hyp_norm)
        ref_len_words = max(wo.hits + wo.substitutions + wo.deletions, 1)
        scored.append({
            **r,
            "family": get_family(r["primary_language"]),
            "reference_normalized": ref_norm,
            "hypothesis_normalized": hyp_norm,
            "wer": wo.wer,
            "n_sub": wo.substitutions,
            "n_del": wo.deletions,
            "n_ins": wo.insertions,
            "ref_len_words": ref_len_words,
        })
    return scored


def _per_family_wer(df) -> dict[str, float]:
    """Corpus-level (micro-averaged) WER per family: total errors / total
    reference words -- the same aggregation convention `jiwer.wer` itself
    uses over multiple sentences, just grouped by family."""
    out = {}
    for family, sub in df.groupby("family"):
        n_err = (sub["n_sub"] + sub["n_del"] + sub["n_ins"]).sum()
        n_ref = sub["ref_len_words"].sum()
        out[family] = float(n_err / n_ref) if n_ref else float("nan")
    return out


def _fit_poisson_drop_in_deviance(df) -> dict:
    """Null model: n_errors ~ 1 + offset(log ref_len_words). Alternative:
    n_errors ~ family + offset(log ref_len_words). The deviance difference
    is chi-squared distributed (df = number of family levels - 1) under the
    null of no family effect; report the upper-tail probability as p."""
    import numpy as np
    import statsmodels.api as sm
    import statsmodels.formula.api as smf
    from scipy.stats import chi2

    work = df.copy()
    work["n_errors"] = work["n_sub"] + work["n_del"] + work["n_ins"]
    offset = np.log(work["ref_len_words"].astype(float))

    null_model = smf.glm("n_errors ~ 1", data=work, family=sm.families.Poisson(), offset=offset).fit()
    alt_model = smf.glm("n_errors ~ C(family)", data=work, family=sm.families.Poisson(), offset=offset).fit()

    deviance_diff = float(null_model.deviance - alt_model.deviance)
    df_diff = int(null_model.df_resid - alt_model.df_resid)
    p_value = float(chi2.sf(deviance_diff, df_diff))

    return {
        "deviance_null": float(null_model.deviance),
        "deviance_alt": float(alt_model.deviance),
        "deviance_diff": deviance_diff,
        "df_diff": df_diff,
        "p_value": p_value,
    }


def _write_summary_md(path: Path, remote_result: dict, scored_rows: list[dict]) -> dict:
    """Write the panel-ready markdown summary and return the key numbers
    (overall WER, per-family WER, ΔDP, Poisson p-value, cost) so the
    caller can also print/report them."""
    import pandas as pd

    df = pd.DataFrame(scored_rows)
    total_err = int((df["n_sub"] + df["n_del"] + df["n_ins"]).sum())
    total_ref = int(df["ref_len_words"].sum())
    overall_wer = total_err / total_ref if total_ref else float("nan")
    per_family = _per_family_wer(df)
    family_n = {family: int((df["family"] == family).sum()) for family in per_family}
    poisson = _fit_poisson_drop_in_deviance(df)
    delta_dp = (max(per_family.values()) - min(per_family.values())) * 100
    cost_usd = remote_result["total_wall_s"] * gpu_second_rate(remote_result["gpu_type"])
    interpretation = "structural" if poisson["p_value"] < 0.05 else "not distinguishable from sampling noise at this eval size"

    family_order = sorted(per_family, key=lambda f: -family_n[f])
    family_table_rows = "\n".join(
        f"| {family} | {family_n[family]} | {per_family[family] * 100:.2f}% |"
        for family in family_order
    )

    n_completed = remote_result["n_utterances_completed"]
    n_total_dataset = remote_result["n_total_dataset"]

    md = f"""# `whisper-large-v3` on Svarah: per-family WER audit

**Feasibility check, not a publication-grade result.** See
["How this differs from the source project's protocol"](#how-this-differs-from-the-source-projects-protocol)
below before citing these numbers anywhere.

- Model: `openai/whisper-large-v3` (fp32, zero-shot, greedy decoding, no fine-tuning on Svarah)
- Data: [`ai4bharat/svarah`](https://huggingface.co/datasets/ai4bharat/svarah)
  (Javed et al., "Svarah: Evaluating English ASR Systems on Indian Accents," Interspeech 2023,
  [arXiv:2305.15760](https://arxiv.org/abs/2305.15760)) -- {n_completed} of {n_total_dataset}
  total utterances (a random subsample; see below)
- Compute: a single Modal `T4`, fp32 -- **actual cost: ${cost_usd:.4f}**,
  wall time {remote_result['total_wall_s']:.1f}s
  (model load {remote_result['model_load_s']:.1f}s + dataset load
  {remote_result['dataset_load_s']:.1f}s + transcription
  {remote_result['transcribe_s']:.1f}s)

## Headline numbers

**Overall WER: {overall_wer * 100:.2f}%** ({total_err} word errors / {total_ref} reference words)

| Family | n utterances | WER |
|---|---|---|
{family_table_rows}

**ΔDP (max family WER − min family WER): {delta_dp:.2f} percentage points**

**Poisson drop-in-deviance test:** deviance difference = {poisson['deviance_diff']:.3f},
df = {poisson['df_diff']}, **p = {poisson['p_value']:.4g}**

> **Interpretation: this cross-family WER gap is {interpretation}.**
> (Convention: p < 0.05 = "structural" / a real gap; p ≥ 0.05 = not
> distinguishable from sampling noise at this eval size.)

Method: fit a Poisson GLM on per-utterance error counts (substitutions +
deletions + insertions) with `log(reference length in words)` as an offset;
the null model has no family term, the alternative adds a categorical
family fixed effect; the deviance difference between the two nested models
is chi-squared distributed under the null, and the reported p is the
upper-tail probability of that statistic.

## How this differs from the source project's protocol

This audit is a fast, cheap feasibility check run ahead of a review
deadline, reusing this repo's existing Modal/cost-guardrail infrastructure
-- it is **not** a re-run of the (separate, unseen) capstone project's
pipeline, and differs from it in at least three material ways:

1. **Full dataset vs. held-out eval split.** This run evaluates a random
   subsample of Svarah (see below), drawn uniformly from the *entire*
   6,656-utterance `test` split (Svarah's only split) rather than any
   particular held-out slice the source project may have reserved for
   evaluation. Since `whisper-large-v3` was never trained or tuned on
   Svarah, there is no train/eval leakage risk here either way, but the
   *sampling* itself (uniform random vs. a specific curated held-out set)
   is not guaranteed to match the source project's protocol.
2. **Independently-reconstructed family mapping, not the authoritative
   one.** `audit/opd_bias_extension/family_mapping.py`'s Svarah-accent -> family table was
   built from scratch from public linguistic classification and
   cross-checked against the Svarah paper's own language list, landing on
   14 Indo-Aryan / 4 Dravidian / 1 Sino-Tibetan (19 total) -- close to, but
   not exactly, the "15/4/1" figure this task's brief attributed to the
   source project (which does not itself sum to 19, suggesting an
   approximate recollection). **This mapping has not been verified against
   the source project's authoritative file and must be cross-checked before
   any of the per-family numbers above are used in a publication-track
   deliverable.** See the disclaimer and full reasoning at the top of
   `audit/opd_bias_extension/family_mapping.py`.
3. **A single feasibility-check run, not a publication-grade result.** One
   model, one random seed, one Modal T4, greedy decoding only -- no repeated
   seeds/variance estimate, no comparison across ASR systems, and (per
   point 1) `{n_completed}` of Svarah's `{n_total_dataset}` utterances
   rather than the full corpus, sized to fit comfortably under the
   `--max-cost-usd` budget and the Modal function's hard timeout backstop.

## Reproduction

```bash
uv sync --extra audit                                    # adds statsmodels/pandas for the Poisson fit
uv run modal run audit/opd_bias_extension/modal_svarah_audit.py::smoke      # ~20-40 utterances, validates the pipeline
uv run modal run audit/opd_bias_extension/modal_svarah_audit.py::run        # the real pass reported above
```
"""
    path.write_text(md)
    return {
        "overall_wer": overall_wer,
        "per_family_wer": per_family,
        "family_n": family_n,
        "delta_dp_pp": delta_dp,
        "poisson": poisson,
        "cost_usd": cost_usd,
        "interpretation": interpretation,
    }


def _print_report(label: str, remote_result: dict, scored_rows: list[dict]) -> None:
    import pandas as pd

    df = pd.DataFrame(scored_rows)
    total_err = (df["n_sub"] + df["n_del"] + df["n_ins"]).sum()
    total_ref = df["ref_len_words"].sum()
    overall_wer = total_err / total_ref if total_ref else float("nan")
    per_family = _per_family_wer(df)
    cost_usd = remote_result["total_wall_s"] * gpu_second_rate(remote_result["gpu_type"])

    print(f"\n=== {label} report ===")
    print(f"gpu={remote_result['gpu_type']} "
          f"model_load={remote_result['model_load_s']:.1f}s "
          f"dataset_load={remote_result['dataset_load_s']:.1f}s "
          f"transcribe={remote_result['transcribe_s']:.1f}s "
          f"total_wall={remote_result['total_wall_s']:.1f}s "
          f"est_cost=${cost_usd:.4f}")
    print(f"utterances: requested={remote_result['n_utterances_requested']} "
          f"completed={remote_result['n_utterances_completed']} "
          f"of {remote_result['n_total_dataset']} total in dataset "
          f"(stopped_early_for_budget={remote_result['stopped_early_for_budget']})")
    print(f"overall WER: {overall_wer * 100:.2f}%")
    print("per-family WER:")
    for family in sorted(per_family):
        n = (df["family"] == family).sum()
        print(f"  {family:14s} WER={per_family[family] * 100:6.2f}%  n_utterances={n}")

    if len(per_family) >= 2 and len(df) >= 10:
        poisson = _fit_poisson_drop_in_deviance(df)
        delta_dp = (max(per_family.values()) - min(per_family.values())) * 100
        print(f"ΔDP (max-min family WER): {delta_dp:.2f} pp")
        print(f"Poisson drop-in-deviance: deviance_diff={poisson['deviance_diff']:.3f} "
              f"df={poisson['df_diff']} p={poisson['p_value']:.4g} "
              f"({'structural' if poisson['p_value'] < 0.05 else 'not distinguishable from noise'})")
    else:
        print("(too few utterances/families for a meaningful Poisson fit -- expected for a tiny smoke sample)")


# ---------------------------------------------------------------------------
# Local entrypoints
# ---------------------------------------------------------------------------


@app.local_entrypoint()
def smoke(num_utterances: int = 32, max_cost_usd: float = 0.20, seed: int = 0):
    """`modal run audit/opd_bias_extension/modal_svarah_audit.py::smoke` -- a cheap end-to-end
    check (model load from cache, chunked greedy decode, family mapping,
    jiwer detail, Poisson fit) on ~20-40 utterances before spending
    anything on the full pass."""
    n_chunks = -(-num_utterances // DEFAULT_BATCH_SIZE)  # ceil div
    est = estimate_cost(GPU_TYPE, n_chunks, ASSUMED_SECONDS_PER_CHUNK, max_cost_usd)
    enforce_budget(est)

    print(f"Launching smoke run on Modal ({GPU_TYPE}, {num_utterances} utterances)...")
    result = transcribe_svarah.remote(
        num_utterances=num_utterances, seed=seed, batch_size=DEFAULT_BATCH_SIZE,
        max_new_tokens=DEFAULT_MAX_NEW_TOKENS, max_cost_usd=max_cost_usd,
    )
    scored = _score_rows(result["rows"])
    _print_report("smoke", result, scored)
    print("\nsmoke OK: pipeline ran end-to-end without error.")


@app.local_entrypoint()
def run(num_utterances: int = 2500, max_cost_usd: float = 3.00, seed: int = 0):
    """`modal run audit/opd_bias_extension/modal_svarah_audit.py::run` -- the real pass.

    `ai4bharat/svarah` actually contains 6,656 utterances total (discovered
    by inspecting the dataset directly -- see family_mapping.py's docstring
    for how this was confirmed), well more than the ~2000 originally
    assumed. `num_utterances=2500` here is a *budget-and-runtime-driven
    random subsample* (uniform over all utterances, so each family's
    utterances appear in roughly their natural dataset proportion, e.g.
    ~2500/6656 x 402 =~ 151 Sino-Tibetan/Bodo utterances -- still enough for
    a meaningful Poisson fit) rather than the full corpus: at the ~5.6-5.85
    s/chunk measured in the `smoke` run, the full 6,656 utterances would
    cost only ~$1.5-1.6 (comfortably under budget) but take ~2.5-3 hours of
    wall time, which trades off against this being a *fast* feasibility
    check. Pass `--num-utterances 0` (or any value >= 6656) to use the full
    corpus instead if that trade-off is preferred for a given run. The
    pre-flight estimate below still gates the request before any GPU is
    billed, and only proceeds if it clears `--max-cost-usd`; the remote
    function's in-run guardrail (see `transcribe_svarah`) additionally caps
    the number of utterances actually completed if the *measured* per-chunk
    rate would otherwise exceed budget."""
    n_chunks_for_estimate = num_utterances if num_utterances > 0 else KNOWN_SVARAH_TEST_SIZE
    n_chunks = -(-n_chunks_for_estimate // DEFAULT_BATCH_SIZE)
    est = estimate_cost(GPU_TYPE, n_chunks, ASSUMED_SECONDS_PER_CHUNK, max_cost_usd)
    enforce_budget(est)  # raises BudgetExceededError before any GPU is requested

    print(f"Launching full run on Modal ({GPU_TYPE}, {num_utterances} utterances requested)...")
    result = transcribe_svarah.remote(
        num_utterances=num_utterances, seed=seed, batch_size=DEFAULT_BATCH_SIZE,
        max_new_tokens=DEFAULT_MAX_NEW_TOKENS, max_cost_usd=max_cost_usd,
    )
    scored = _score_rows(result["rows"])
    _print_report("run", result, scored)

    import pandas as pd

    df = pd.DataFrame(scored)
    csv_path = LOCAL_RESULTS_DIR / "results_whisper-large-v3_svarah_clean.csv"
    columns = [
        "utt_id", "primary_language", "family", "native_place_state", "gender", "duration_s",
        "reference_text", "reference_normalized", "hypothesis_text", "hypothesis_normalized",
        "wer", "n_sub", "n_del", "n_ins", "ref_len_words",
    ]
    df[columns].to_csv(csv_path, index=False)
    print(f"\nWrote {csv_path} ({len(df)} rows)")

    md_path = LOCAL_RESULTS_DIR / "summary_whisper-large-v3_svarah.md"
    key_numbers = _write_summary_md(md_path, result, scored)
    print(f"Wrote {md_path}")
    print(f"\nFinal numbers: overall_wer={key_numbers['overall_wer'] * 100:.2f}% "
          f"delta_dp={key_numbers['delta_dp_pp']:.2f}pp "
          f"p={key_numbers['poisson']['p_value']:.4g} "
          f"({key_numbers['interpretation']}) "
          f"cost=${key_numbers['cost_usd']:.4f}")
