"""Acceptance runner for the gated IndicConformer ONNX wrapper.

The wrapper API is deliberately explicit: no model or package API is guessed.
An adapter module must export ``load(model_dir, providers)`` and return an
object with ``transcribe_batch(items, languages)``. ``items`` is a list of
``(float32_mono_16k_waveform, 16000)`` pairs; the method returns one text per
item in input order. It may expose ``evidence()`` for ONNX session metadata.
"""
from __future__ import annotations
import argparse, hashlib, importlib, importlib.metadata, json, math, sys, time
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[3]
MODEL_ID = "ai4bharat/indic-conformer-600m-multilingual"
MODEL_REVISION = "e9b71b369c048e2c6b634d4c131061c34e441179"
LANGUAGES = {"as", "bn", "hi"}

def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""): h.update(block)
    return h.hexdigest()

def now(): return datetime.now(timezone.utc).isoformat()

def load_rows(cfg):
    manifest = Path(cfg["manifest_path"])
    if sha(manifest) != cfg["manifest_sha256"]: raise ValueError("manifest hash mismatch")
    byid = {json.loads(x)["conversation_id"]: json.loads(x) for x in manifest.read_text().splitlines() if x.strip()}
    ids = cfg["conversation_ids"]
    if len(ids) != len(set(ids)) or set(ids) - set(byid): raise ValueError("invalid configured cohort")
    rows = [byid[x] for x in ids]
    if any(r["language"] not in LANGUAGES or r["reference_qc"] != "pass" for r in rows): raise ValueError("unsupported language or QC failure")
    return rows

def load_adapter(spec, model_dir, providers):
    module_name, sep, attr = spec.partition(":")
    if not sep: raise ValueError("adapter must be module:factory")
    factory = getattr(importlib.import_module(module_name), attr)
    adapter = factory(Path(model_dir), providers=list(providers))
    if not hasattr(adapter, "transcribe_batch"): raise TypeError("adapter lacks transcribe_batch")
    return adapter

def run(config_path, adapter_spec, model_dir):
    cfg = json.loads(Path(config_path).read_text()); rows = load_rows(cfg)
    out = Path(cfg["output_dir"]); out.mkdir(parents=True, exist_ok=False)
    (out / "config.json").write_text(json.dumps(cfg, indent=2, ensure_ascii=False) + "\n")
    files = {str(p.relative_to(model_dir)): sha(p) for p in Path(model_dir).rglob("*") if p.is_file()}
    started = time.monotonic(); adapter = load_adapter(adapter_spec, model_dir, cfg["onnx_providers"])
    evidence = {"started_at": now(), "model_id": cfg["model_id"], "model_revision": cfg["model_revision"],
        "model_file_sha256": files, "adapter": adapter_spec, "onnx_providers": cfg["onnx_providers"],
        "eval_state_observable": False, "packages": {n: importlib.metadata.version(n) for n in cfg["required_packages"]}}
    if hasattr(adapter, "evidence"): evidence["adapter_evidence"] = adapter.evidence()
    evidence["load_seconds"] = time.monotonic() - started
    evidence["config_sha256"] = sha(Path(config_path)); evidence["manifest_sha256"] = cfg["manifest_sha256"]
    (out / "environment.json").write_text(json.dumps(evidence, indent=2, ensure_ascii=False) + "\n")
    predictions = []
    for begin in range(0, len(rows), cfg["batch_size"]):
        group = rows[begin:begin + cfg["batch_size"]]; items=[]; audio_meta=[]
        for row in group:
            audio, sr = sf.read(row["audio_path"], dtype="float32")
            if sr != 16000 or audio.ndim != 1: raise ValueError("expected mono 16 kHz audio")
            peak = float(np.max(np.abs(audio))) if audio.size else 0.0
            if not np.isfinite(audio).all() or peak <= 0 or peak > 1: raise ValueError("invalid waveform")
            items.append((audio, sr)); audio_meta.append((row, peak, len(audio)/sr))
        texts = adapter.transcribe_batch(items, [r["language"] for r in group])
        if not isinstance(texts, list) or len(texts) != len(group): raise ValueError("adapter output cardinality mismatch")
        for (row, peak, duration), text in zip(audio_meta, texts):
            text = str(text); ref = json.loads(Path(row["reference_path"]).read_text())["reference_text"]
            predictions.append({"conversation_id": row["conversation_id"], "language": row["language"], "model_id": cfg["model_id"], "model_revision": cfg["model_revision"], "audio_sha256": row["audio_sha256"], "reference_sha256": row["reference_sha256"], "status": "success" if text.strip() else "empty_output", "text_raw": text, "text_for_scoring": normalize(text), "metrics": score(ref, text), "duration_seconds": duration, "audio_peak_abs": peak, "batch_index": begin // cfg["batch_size"]})
    (out / "predictions.jsonl").write_text("\n".join(json.dumps(x, ensure_ascii=False) for x in predictions) + "\n")
    summary={"status":"completed", "attempted":len(rows), "scored":len(predictions), "model_id":cfg["model_id"], "model_revision":cfg["model_revision"], "languages":sorted(LANGUAGES)}
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n"); return 0

def normalize(text):
    from research.bridge_open.scoring import normalize_v1
    return normalize_v1(text)
def score(ref, hyp):
    from research.bridge_open.scoring import score as scorer
    return scorer(ref, hyp)

if __name__ == "__main__":
    p=argparse.ArgumentParser(); p.add_argument("config"); p.add_argument("--adapter", required=True); p.add_argument("--model-dir", required=True)
    a=p.parse_args(); raise SystemExit(run(a.config, a.adapter, a.model_dir))
