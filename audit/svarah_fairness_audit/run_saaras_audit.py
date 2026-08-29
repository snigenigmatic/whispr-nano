"""Run all fetched Svarah clips through Saaras V3, score with ast_asr's own
taxonomy/metrics convention (imported from the already-validated
modal_whisper_family_audit.py in the same style), with disk response caching
so re-runs after any interruption are free."""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import requests

# Assumes this file lives in <ast-asr repo root>/audit/, alongside
# modal_whisper_family_audit.py, with the repo's own src/ package installed
# (e.g. `uv sync` or `pip install -e .` from the repo root) or PYTHONPATH set
# to <repo root>/src.
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(_REPO_ROOT / "src"))

from modal_whisper_family_audit import _print_report, _score_rows, _write_csv  # noqa: E402
from ast_asr.taxonomy import SVARAH_LANGUAGE_FAMILIES  # noqa: E402

# Fetched clips + manifest.json (see fetch_stratified.py's local_entrypoint)
# are expected under this directory before running this script.
AUDIT_DIR = _REPO_ROOT / "audit" / "saaras_audit"
CACHE_DIR = AUDIT_DIR / "response_cache"
CACHE_DIR.mkdir(exist_ok=True)
GPU_HOURLY_USD = 0.0  # not GPU-billed; Saaras cost tracked separately below
SARVAM_RATE_INR_PER_SEC = 30.0 / 3600.0
INR_PER_USD = 87.0


def transcribe_cached(wav_path: Path, api_key: str, model: str = "saaras:v3") -> tuple[str, bool]:
    """Returns (transcript, was_cached)."""
    cache_file = CACHE_DIR / f"{wav_path.stem}.json"
    if cache_file.exists():
        return json.loads(cache_file.read_text())["transcript"], True

    for attempt in range(5):
        with open(wav_path, "rb") as f:
            resp = requests.post(
                "https://api.sarvam.ai/speech-to-text",
                headers={"api-subscription-key": api_key},
                files={"file": (wav_path.name, f, "audio/wav")},
                data={"model": model, "mode": "transcribe", "language_code": "en-IN"},
                timeout=60,
            )
        if resp.status_code == 200:
            data = resp.json()
            cache_file.write_text(json.dumps(data))
            return data["transcript"], False
        if resp.status_code in (429, 500, 502, 503):
            wait = 2 ** attempt
            print(f"  [retry] status {resp.status_code} for {wav_path.name}, waiting {wait}s...", flush=True)
            time.sleep(wait)
            continue
        raise RuntimeError(f"Saaras API failed ({resp.status_code}) for {wav_path.name}: {resp.text[:300]}")
    raise RuntimeError(f"Saaras API failed after 5 attempts for {wav_path.name}")


def main():
    api_key = os.environ["SARVAM_API_KEY"]
    manifest = json.loads((AUDIT_DIR / "manifest.json").read_text())
    print(f"Transcribing {len(manifest)} clips via Saaras V3...", flush=True)

    total_duration_s = sum(m.get("duration_s") or 0 for m in manifest)
    est_cost_inr = total_duration_s * SARVAM_RATE_INR_PER_SEC
    print(f"Estimated audio duration: {total_duration_s:.0f}s ({total_duration_s/3600:.2f}h) "
          f"-> est. cost INR {est_cost_inr:.2f} (~${est_cost_inr/INR_PER_USD:.3f})", flush=True)

    rows = []
    t0 = time.monotonic()
    n_cached = 0
    for i, m in enumerate(manifest):
        wav_path = Path(m["wav_path"])
        transcript, was_cached = transcribe_cached(wav_path, api_key)
        n_cached += int(was_cached)
        rows.append({
            "utt_id": f"svarah_{m['index']:05d}",
            "primary_language": m["primary_language"],
            "gender": m.get("gender", "Unknown"),
            "duration_s": m.get("duration_s"),
            "reference_text": m["text"],
            "hypothesis_text": transcript.strip(),
        })
        if (i + 1) % 50 == 0 or (i + 1) == len(manifest):
            elapsed = time.monotonic() - t0
            print(f"  {i + 1}/{len(manifest)} done ({elapsed:.0f}s elapsed, {n_cached} from cache)", flush=True)
        if not was_cached:
            time.sleep(0.15)  # be polite to the rate limit

    transcribe_s = time.monotonic() - t0
    scored = _score_rows(rows)

    remote_result = {
        "model_id": "sarvam/saaras-v3",
        "load_s": 0.0,
        "dataset_load_s": 0.0,
        "transcribe_s": transcribe_s,
        "total_wall_s": transcribe_s,
        "n_total_dataset": 6656,
        "gpu_type": "api",
    }

    total_err = sum(r["n_sub"] + r["n_del"] + r["n_ins"] for r in scored)
    total_ref = sum(r["ref_len_words"] for r in scored)
    overall_wer = total_err / total_ref if total_ref else float("nan")
    per_family: dict[str, list[int]] = {}
    for r in scored:
        per_family.setdefault(r["family"], [0, 0])
        per_family[r["family"]][0] += r["n_sub"] + r["n_del"] + r["n_ins"]
        per_family[r["family"]][1] += r["ref_len_words"]
    per_family_wer = {f: (e / rl if rl else float("nan")) for f, (e, rl) in per_family.items()}
    family_n = {f: sum(1 for r in scored if r["family"] == f) for f in per_family_wer}

    from modal_whisper_family_audit import _fit_poisson_drop_in_deviance
    poisson = _fit_poisson_drop_in_deviance(scored) if len(per_family_wer) >= 2 else None

    result = {
        "model_id": "sarvam/saaras-v3",
        "overall_wer": overall_wer,
        "n_utterances": len(scored),
        "per_family_wer": per_family_wer,
        "family_n": family_n,
        "cost_usd": est_cost_inr / INR_PER_USD,
        "cost_inr": est_cost_inr,
        "wall_seconds": transcribe_s,
    }
    if poisson:
        result["delta_dp_pp"] = (max(per_family_wer.values()) - min(per_family_wer.values())) * 100
        result["poisson"] = poisson
        result["interpretation"] = "structural" if poisson["p_value"] < 0.05 else "not distinguishable from sampling noise"

    _write_csv(AUDIT_DIR / "results_saarasv3_svarah_clean.csv", scored)
    (AUDIT_DIR / "summary_saarasv3_svarah.json").write_text(json.dumps(result, indent=2))
    print(f"\nWrote {AUDIT_DIR / 'results_saarasv3_svarah_clean.csv'} ({len(scored)} rows)")
    print(f"Wrote {AUDIT_DIR / 'summary_saarasv3_svarah.json'}")

    print(f"\n=== sarvam/saaras-v3 report ===")
    print(f"transcribe={transcribe_s:.1f}s cost=${result['cost_usd']:.4f} (INR {result['cost_inr']:.2f})")
    print(f"utterances: completed={result['n_utterances']}")
    print(f"overall WER: {result['overall_wer'] * 100:.2f}%")
    print("per-family WER:")
    for family in sorted(per_family_wer):
        print(f"  {family:14s} WER={per_family_wer[family] * 100:6.2f}%  n_utterances={family_n[family]}")
    if poisson:
        print(f"ΔDP: {result['delta_dp_pp']:.2f}pp  Poisson p={poisson['p_value']:.4g} ({result['interpretation']})")


if __name__ == "__main__":
    main()
