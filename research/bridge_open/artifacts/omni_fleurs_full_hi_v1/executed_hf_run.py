"""Stage a full FLEURS shard, run the frozen Omni runtime, and export evidence."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import time
import traceback
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def source_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def _check_config(cfg: dict, source_rows: list[dict]) -> list[dict]:
    if cfg.get("inference_mode") != "full" or cfg.get("decoding_mode") != "full_waveform_ctc":
        raise ValueError("Full worker refuses a non-full-waveform condition")
    if cfg.get("condition_id") == cfg.get("acceptance_condition_id") or "acceptance" in cfg.get("condition_id", ""):
        raise ValueError("Full and acceptance conditions must remain distinct")
    if cfg.get("full_waveform") is not True or "chunk_seconds" in cfg or cfg.get("batch_size") != 1:
        raise ValueError("Omni full FLEURS condition requires one unsplit waveform per forward")
    expected_counts = cfg.get("frozen_cohort_counts")
    observed_counts = {language: sum(row["language"] == language for row in source_rows)
                       for language in ["as", "bn", "hi"]}
    if expected_counts != observed_counts or sum(observed_counts.values()) != 2322:
        raise ValueError("Frozen source cohort counts differ from the declared 2,322-record contract")
    wanted = cfg.get("conversation_ids", [])
    if not wanted or len(wanted) != len(set(wanted)):
        raise ValueError("Configured shard must contain unique conversation IDs")
    by_id = {row["conversation_id"]: row for row in source_rows}
    selected = [by_id[cid] for cid in wanted if cid in by_id]
    if len(selected) != len(wanted):
        raise ValueError("Configured IDs are absent from the frozen source manifest")
    expected_language = cfg.get("expected_language")
    if expected_language is not None and any(row["language"] != expected_language for row in selected):
        raise ValueError("Language-shard config mixes languages")
    if set(row["language"] for row in selected) != set(cfg.get("supported_languages", [])):
        raise ValueError("Configured supported_languages differs from the selected shard")
    return selected


def main(argv: list[str] | None = None) -> int:
    import torch
    from research.bridge_open.omni_extension.artifact_transport import emit
    from research.bridge_open.omni_extension.run_omni_full import run
    from research.bridge_open.omni_extension.validate_run import validate
    from research.bridge_open.stage_fleurs import stage

    started = time.monotonic()
    config_path = ROOT / "research/bridge_open/run_config.json"
    cfg = json.loads(config_path.read_text())
    cfg["output_dir"] = str(ROOT / "artifacts" / cfg["cloud_job_id"])
    output = Path(cfg["output_dir"])
    code = 1
    validation = {"valid": False, "reason": "runner did not reach independent validation"}
    failure = None
    try:
        if torch.__version__.split("+")[0] != "2.8.0" or not torch.cuda.is_available():
            raise RuntimeError("Requires PyTorch 2.8.0 and an available CUDA GPU")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        source_manifest = source_path(cfg["source_manifest"])
        reference_pack = source_path(cfg["reference_pack"])
        if digest(source_manifest) != cfg["source_manifest_sha256"]:
            raise ValueError("Frozen FLEURS v2 manifest hash mismatch")
        if digest(reference_pack) != cfg["reference_pack_sha256"]:
            raise ValueError("Frozen reference pack hash mismatch")
        source_rows = [json.loads(line) for line in source_manifest.read_text().splitlines()]
        selected = _check_config(cfg, source_rows)
        # Stage checks each source archive and every output audio/reference hash.
        # It only receives selected rows, so references cannot affect inference until
        # the runtime reads them after each prediction has been produced.
        stage(selected, reference_pack)
        manifest = Path(cfg["manifest_path"])
        manifest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source_manifest, manifest)
        if digest(manifest) != cfg["manifest_sha256"]:
            raise ValueError("Staged v2 manifest no longer matches frozen bytes")
        os.chdir(ROOT)
        cfg["manifest_path"] = str(manifest)
        effective = ROOT / "effective_omni_full_config.json"
        effective.write_text(json.dumps(cfg, ensure_ascii=False, indent=2) + "\n")
        code = run(effective)
        validation = validate(output)
        (output / "remote_validation.json").write_text(json.dumps(validation, indent=2) + "\n")
    except Exception as exc:
        failure = {"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()}
        output.mkdir(parents=True, exist_ok=True)
        (output / "worker_failure.json").write_text(json.dumps(failure, indent=2) + "\n")
        print(json.dumps({"event": "omni_full_failure", "failure": failure}), flush=True)
    finally:
        receipt = {
            "provider": "huggingface",
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "elapsed_seconds": time.monotonic() - started,
            "timeout_seconds": cfg["worker_timeout_seconds"],
            "cloud_cost_usd": None,
            "provider_bill_verified": False,
            "cost_note": "Cost is null: elapsed setup/runtime is not a provider billing record.",
            "worker_failure": failure,
            "source_sha256": {
                str(path.relative_to(ROOT)): digest(path)
                for path in [Path(__file__), ROOT / "research/bridge_open/stage_fleurs.py", ROOT / "research/bridge_open/omni_extension/run_omni_full.py",
                             ROOT / "research/bridge_open/omni_extension/validate_run.py", config_path]
            },
        }
        if output.is_dir():
            (output / "hf_receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
            try:
                emit(output)
            except Exception as transport_exc:
                print(json.dumps({"event": "artifact_transport_failure", "failure": {
                    "type": type(transport_exc).__name__, "message": str(transport_exc),
                }}), flush=True)
    print(json.dumps({"event": "omni_full_complete", "validation": validation, "receipt": receipt}), flush=True)
    return code if validation.get("valid") else 2


if __name__ == "__main__":
    raise SystemExit(main())
