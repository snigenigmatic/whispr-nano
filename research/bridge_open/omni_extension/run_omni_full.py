"""Pinned, reference-independent full-waveform OmniASR CTC FLEURS runner.

The model is CTC: a completed forward pass has no EOS token or generation
length.  The output deliberately records CTC token evidence and leaves
termination fields null; the validator must apply the CTC contract described
in OMNI_RUNTIME_CONTRACT.md.
"""
from __future__ import annotations

import hashlib
import importlib.metadata
import inspect
import json
import platform
import shutil
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from .scoring import normalize_v1, score

MODEL_ID = "facebook/omniASR-CTC-1B"
MODEL_REVISION = "8c22e3ffdaa4aab6431b128b84b991a7d9c2515c"
MODEL_CARD = "omniASR_CTC_1B"
MAX_ALLOWED_AUDIO_SEC = 40
EXPECTED_ASSETS = {
    "omniASR-CTC-1B.pt": "e8564fa59dab7caedbcdb54ab7fb9bd6c96989f4d19add2ad81ddd969716952c",
    "omniASR_tokenizer.model": "b954cc166b0c9e0271b953fa226fa27ca706a25b7029e84579fe2c60a2b451fe",
}
OFFICIAL_RUNTIME_COMMIT = "81f51e224ce9e74b02cc2a3eaf21b2d91d743455"


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def verify(path: Path, expected: str) -> None:
    if sha(path) != expected:
        raise ValueError(f"Content hash mismatch: {path}")


def _load_pinned_pipeline():
    """Download exact HF files, then feed them through official fairseq2 cards."""
    from huggingface_hub import hf_hub_download
    import omnilingual_asr  # registers official fairseq2 cards
    from fairseq2.assets import get_asset_store, AssetCard
    from fairseq2.data.tokenizers.hub import load_tokenizer
    from fairseq2.models.hub import load_model
    from omnilingual_asr.models.inference.pipeline import ASRInferencePipeline

    cache_dir = "/cache/huggingface"
    downloaded_model = Path(hf_hub_download(MODEL_ID, "omniASR-CTC-1B.pt",
                                             revision=MODEL_REVISION, cache_dir=cache_dir, token=False))
    downloaded_tokenizer = Path(hf_hub_download(MODEL_ID, "omniASR_tokenizer.model",
                                                 revision=MODEL_REVISION, cache_dir=cache_dir, token=False))
    for path, expected in [(downloaded_model, EXPECTED_ASSETS["omniASR-CTC-1B.pt"]),
                           (downloaded_tokenizer, EXPECTED_ASSETS["omniASR_tokenizer.model"] )]:
        if sha(path) != expected:
            raise RuntimeError(f"Pinned HF asset hash mismatch: {path}")
    # HF may return an extensionless content-addressed blob. fairseq2 selects
    # its checkpoint deserializer from the suffix, so preserve explicit names.
    verified_dir = Path("/tmp/bridge_omni_verified")
    verified_dir.mkdir(parents=True, exist_ok=True)
    model_path = verified_dir / "omniASR-CTC-1B.pt"
    tokenizer_path = verified_dir / "omniASR_tokenizer.model"
    shutil.copyfile(downloaded_model, model_path)
    shutil.copyfile(downloaded_tokenizer, tokenizer_path)
    for path, expected in [(model_path, EXPECTED_ASSETS["omniASR-CTC-1B.pt"]),
                           (tokenizer_path, EXPECTED_ASSETS["omniASR_tokenizer.model"] )]:
        if sha(path) != expected:
            raise RuntimeError(f"Verified local asset copy hash mismatch: {path}")

    store = get_asset_store()
    model_card = store.retrieve_card(MODEL_CARD)
    tokenizer_card = store.retrieve_card("omniASR_tokenizer_v1")
    # AssetCard metadata is the official fairseq2 override seam. Absolute file
    # paths are accepted by AssetCardField.as_uri() and avoid a second download.
    model_card = AssetCard(model_card.name, {**model_card.metadata, "checkpoint": model_path.resolve().as_uri()}, model_card.base)
    tokenizer_card = AssetCard(tokenizer_card.name, {**tokenizer_card.metadata, "tokenizer": tokenizer_path.resolve().as_uri()}, tokenizer_card.base)
    model = load_model(model_card, device=torch.device("cuda"), dtype=torch.bfloat16, progress=False).eval()
    parameter_dtypes = sorted({str(parameter.dtype) for parameter in model.parameters()})
    if parameter_dtypes != ["torch.bfloat16"]:
        raise RuntimeError(f"Expected every Omni model parameter to be BF16; found {parameter_dtypes}")
    if model.training:
        raise RuntimeError("Expected Omni model.eval() to disable training mode")
    tokenizer = load_tokenizer(tokenizer_card)
    pipeline = ASRInferencePipeline(model_card=None, model=model, tokenizer=tokenizer,
                                    device="cuda", dtype=torch.bfloat16)
    pipeline_file = Path(inspect.getsourcefile(ASRInferencePipeline) or "")
    package_file = Path(omnilingual_asr.__file__ or "")
    runtime_evidence = {
        "official_omnilingual_asr_commit": OFFICIAL_RUNTIME_COMMIT,
        "pipeline_source_path": str(pipeline_file),
        "pipeline_source_sha256": sha(pipeline_file),
        "package_init_path": str(package_file),
        "package_init_sha256": sha(package_file),
        "parameter_dtypes": parameter_dtypes,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "model_training": bool(model.training),
        "requested_pipeline_dtype": str(torch.bfloat16),
    }
    return pipeline, {"omniASR-CTC-1B.pt": sha(model_path),
                      "omniASR_tokenizer.model": sha(tokenizer_path)}, runtime_evidence


def _preflight_audio(row: dict) -> dict:
    """Reject malformed staged waveform inputs before loading the model."""
    path = Path(row["audio_path"])
    info = sf.info(path)
    audio, sample_rate = sf.read(path, dtype="float32", always_2d=False)
    if sample_rate != 16000 or info.samplerate != 16000 or info.channels != 1 or audio.ndim != 1:
        raise ValueError(f"Expected mono 16 kHz WAV for {row['conversation_id']}")
    if info.frames <= 0 or len(audio) != info.frames:
        raise ValueError(f"Empty or frame-count-mismatched WAV for {row['conversation_id']}")
    if not np.isfinite(audio).all():
        raise ValueError(f"Non-finite WAV samples for {row['conversation_id']}")
    peak = float(np.max(np.abs(audio)))
    if peak == 0.0:
        raise ValueError(f"Silent WAV samples for {row['conversation_id']}")
    if peak > 1.0:
        raise ValueError(f"Out-of-range WAV peak for {row['conversation_id']}: {peak}")
    declared_frames = float(row["duration_seconds"]) * sample_rate
    if abs(declared_frames - info.frames) > 1e-6:
        raise ValueError(
            f"Manifest duration is not the exact sample duration for {row['conversation_id']}: "
            f"declared={row['duration_seconds']} actual={info.frames / sample_rate}"
        )
    return {
        "sample_rate": sample_rate,
        "channels": info.channels,
        "frames": info.frames,
        "subtype": info.subtype,
        "duration_seconds": info.frames / sample_rate,
        "peak": peak,
        "finite": True,
    }


def _ctc_decode(pipeline, audio: np.ndarray, sample_rate: int) -> tuple[str, list[int], list[int]]:
    """Run the official pipeline preprocessing/model, retaining CTC IDs."""
    from fairseq2.nn.batch_layout import BatchLayout

    builder = pipeline._build_audio_wavform_pipeline([{
        "waveform": audio,
        "sample_rate": sample_rate,
    }])
    processed = list(builder.and_return())
    batch = pipeline._create_batch_simple([(processed[0], None)])
    layout = BatchLayout(batch.source_seqs.shape, seq_lens=batch.source_seq_lens,
                         device=batch.source_seqs.device)
    with torch.inference_mode():
        logits, model_out = pipeline.model(batch.source_seqs, layout)
    raw_ids = torch.argmax(logits, dim=-1)[0][: model_out.seq_lens[0]].detach().cpu().tolist()
    collapsed = []
    for token in raw_ids:
        if not collapsed or token != collapsed[-1]:
            collapsed.append(int(token))
    return pipeline.token_decoder(torch.tensor(collapsed)), [int(x) for x in raw_ids], collapsed


def _validate_config(cfg: dict) -> None:
    if cfg.get("model_id") != MODEL_ID or cfg.get("model_revision") != MODEL_REVISION:
        raise ValueError("Omni runner requires the frozen facebook/omniASR-CTC-1B revision")
    if cfg.get("inference_mode") not in {"acceptance", "full"}:
        raise ValueError("inference_mode must be acceptance or full")
    if cfg.get("full_waveform") is not True or "chunk_seconds" in cfg:
        raise ValueError("FLEURS Omni condition requires exactly one full waveform and no fixed chunk setting")
    if cfg.get("batch_size") != 1 or cfg.get("termination_semantics") != "ctc_no_eos":
        raise ValueError("FLEURS Omni condition requires batch size 1 and CTC no-EOS semantics")
    if cfg.get("max_audio_seconds") != MAX_ALLOWED_AUDIO_SEC:
        raise ValueError("FLEURS Omni condition must retain the official 40-second input cap")
    if cfg.get("decoding_mode") != "full_waveform_ctc":
        raise ValueError("FLEURS Omni condition must declare full_waveform_ctc decoding")


def run(config_path: str | Path) -> int:
    config_path = Path(config_path)
    cfg = json.loads(config_path.read_text())
    _validate_config(cfg)
    manifest_path = Path(cfg["manifest_path"])
    verify(manifest_path, cfg["manifest_sha256"])
    rows = [json.loads(line) for line in manifest_path.read_text().splitlines()]
    by_id = {r["conversation_id"]: r for r in rows}
    ids = cfg["conversation_ids"]
    if len(by_id) != len(rows) or len(ids) != len(set(ids)):
        raise ValueError("Duplicate conversation IDs")
    selected = [by_id[cid] for cid in ids]
    input_evidence = {}
    for row in selected:
        if row["reference_qc"] != "pass" or row["language"] not in cfg["supported_languages"]:
            raise ValueError("QC failure or unsupported language in configured cohort")
        verify(Path(row["audio_path"]), row["audio_sha256"])
        # Hash only: reference text stays unavailable until inference completes.
        verify(Path(row["reference_path"]), row["reference_sha256"])
        if float(row["duration_seconds"]) > MAX_ALLOWED_AUDIO_SEC:
            raise ValueError("Configured audio exceeds the official 40-second Omni cap")
        input_evidence[row["conversation_id"]] = _preflight_audio(row)

    out = Path(cfg["output_dir"])
    out.mkdir(parents=True, exist_ok=False)
    (out / "config.json").write_text(json.dumps(cfg, indent=2) + "\n")
    torch.manual_seed(cfg.get("seed", 12))
    np.random.seed(cfg.get("seed", 12))
    torch.set_num_threads(4)
    if cfg.get("device") != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("Configured CUDA device unavailable; refuse fallback")
    started = time.monotonic()

    pipeline, assets, runtime_evidence = _load_pinned_pipeline()
    packages = {}
    for name in ["torch", "torchaudio", "fairseq2", "omnilingual-asr", "numpy", "scipy", "soundfile", "jiwer", "huggingface-hub"]:
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    environment = dict(started_at=now(), config_sha256=sha(config_path),
                       manifest_sha256=sha(manifest_path), model_id=MODEL_ID,
                       model_revision=MODEL_REVISION, model_card=MODEL_CARD,
                       asset_sha256=assets,
                       code_sha256={p.name: sha(p) for p in [Path(__file__), Path(__file__).with_name("scoring.py")]},
                       runtime_evidence=runtime_evidence,
                       input_preflight=input_evidence,
                       packages=packages, platform=platform.platform(), python=platform.python_version(),
                       device="cuda", gpu=torch.cuda.get_device_name(0), cuda=torch.version.cuda,
                       dtype="bfloat16", model_load_seconds=time.monotonic() - started,
                       ctc_termination=True, full_waveform=True,
                       cloud_cost_usd=None if cfg.get("executor") in {"modal", "hf"} else 0)
    (out / "environment.json").write_text(json.dumps(environment, indent=2) + "\n")
    print(json.dumps({"event": "run_started", "config": cfg, "environment": environment}), flush=True)

    all_results = []
    with (out / "predictions.jsonl").open("x") as predictions:
        for row in selected:
            record = dict(record_type="conversation_prediction", schema_version="1.1-ctc",
                          conversation_id=row["conversation_id"], language=row["language"],
                          model_id=MODEL_ID, model_revision=MODEL_REVISION,
                          condition_id=cfg.get("condition_id", "audio_only_full_waveform_native_ctc"), attempt=1,
                          audio_sha256=row["audio_sha256"], reference_sha256=row["reference_sha256"],
                          started_at=now(), status="inference_error", chunks=[], failure=None,
                          termination_semantics="ctc_no_eos")
            conv_started = time.monotonic()
            try:
                audio, sr = sf.read(row["audio_path"], dtype="float32")
                if sr != 16000 or audio.ndim != 1:
                    raise ValueError("Expected prevalidated mono 16kHz WAV")
                if time.monotonic() - started > cfg.get("max_wall_seconds", 3600):
                    raise TimeoutError("Configured run wall-time limit reached before full waveform")
                if len(audio) / sr > MAX_ALLOWED_AUDIO_SEC:
                    raise ValueError("Omni official pipeline caps each input at 40 seconds")
                # Acceptance runs test the public API against the retained-token
                # path; full evaluation deliberately avoids duplicate forwards.
                public_text = None
                if cfg["inference_mode"] == "acceptance":
                    public_text = pipeline.transcribe([{"waveform": audio, "sample_rate": sr}], batch_size=1)[0]
                torch.cuda.reset_peak_memory_stats()
                text, raw_ids, collapsed_ids = _ctc_decode(pipeline, audio, sr)
                if public_text is not None and public_text != text:
                    raise RuntimeError("Official OmniASR transcribe output differs from retained-token decode")
                record["chunks"] = [dict(start_seconds=0.0, end_seconds=len(audio) / sr, text_raw=text,
                    ctc_raw_token_ids=raw_ids, ctc_collapsed_token_ids=collapsed_ids, eos_present=None,
                    ctc_collapse_semantics="consecutive identical argmax IDs collapsed; official tokenizer decoder applied",
                    peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated())]
                record["api_token_agreement"] = public_text is not None
                record["text_raw"] = text.strip()
                record["text_for_scoring"] = normalize_v1(record["text_raw"])
                record["status"] = "success" if record["text_for_scoring"] else "empty_output"
                record["duration_seconds"] = len(audio) / sr
                record["peak_cuda_allocated_bytes"] = record["chunks"][0]["peak_cuda_allocated_bytes"]
                verify(Path(row["reference_path"]), row["reference_sha256"])
                reference = json.loads(Path(row["reference_path"]).read_text())["reference_text"]
                record["metrics"] = score(reference, record["text_raw"])
            except Exception as exc:
                record["failure"] = {"type": type(exc).__name__, "message": str(exc)}
                traceback.print_exc()
            record["ended_at"] = now()
            record["wall_seconds"] = time.monotonic() - conv_started
            if "duration_seconds" in record:
                record["real_time_factor"] = record["wall_seconds"] / record["duration_seconds"]
            predictions.write(json.dumps(record, ensure_ascii=False) + "\n")
            predictions.flush()
            all_results.append(record)

    scored = [r for r in all_results if r["status"] in {"success", "empty_output"} and "metrics" in r]
    aggregates = {}
    for language in sorted({r["language"] for r in selected}):
        group = [r for r in scored if r["language"] == language]
        if not group:
            continue
        word_errors = sum(sum(r["metrics"][k] for k in ["substitutions", "deletions", "insertions"]) for r in group)
        ref_words = sum(r["metrics"]["ref_words"] for r in group)
        char_errors = sum(r["metrics"]["character_errors"] for r in group)
        ref_chars = sum(r["metrics"]["ref_chars"] for r in group)
        aggregates[language] = dict(estimand="all fully processed conversations", conversation_ids=[r["conversation_id"] for r in group],
            scored_count=len(group), attempted_count=sum(r["language"] == language for r in all_results),
            clean_completion_count=sum(r["status"] == "success" for r in group), generation_limit_count=0,
            word_errors=word_errors, ref_words=ref_words, character_errors=char_errors, ref_chars=ref_chars,
            micro_wer=word_errors / ref_words, micro_cer=char_errors / ref_chars,
            mean_conversation_wer=sum(r["metrics"]["wer"] for r in group) / len(group),
            mean_conversation_wil=sum(r["metrics"]["wil"] for r in group) / len(group))
    if len(scored) != len(selected):
        summary_status = "completed_with_runtime_failures"
    elif all(r["status"] == "success" for r in scored):
        summary_status = "completed"
    else:
        summary_status = "completed_with_quality_flags"
    summary = dict(status=summary_status,
                   scope="engineering/public-data pilot; CTC no-EOS contract", attempted=len(selected),
                   successful=sum(r["status"] == "success" for r in all_results), scored=len(scored),
                   empty_output_count=sum(r["status"] == "empty_output" for r in all_results),
                   runtime_failure_count=sum(r["status"] == "inference_error" for r in all_results),
                   conversation_ids=ids, per_language=aggregates,
                   results=[{k: r.get(k) for k in ["conversation_id", "language", "status", "metrics", "wall_seconds", "real_time_factor"]} for r in all_results],
                   total_wall_seconds=time.monotonic() - started,
                   cloud_cost_usd=None if cfg.get("executor") in {"modal", "hf"} else 0)
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({"event": "final_summary", **summary}), flush=True)
    return 0 if len(scored) == len(selected) else 1


if __name__ == "__main__":
    raise SystemExit(run(Path(__file__).with_name("run_config.json")))
