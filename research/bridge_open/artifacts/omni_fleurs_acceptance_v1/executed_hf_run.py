"""Run the bundled exact-input Omni acceptance cohort on an HF GPU worker."""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    import torch
    from research.bridge_open.omni_extension.artifact_transport import emit
    from research.bridge_open.omni_extension.run_omni_full import run
    from research.bridge_open.omni_extension.validate_run import validate

    started = time.monotonic()
    config_path = ROOT / "research/bridge_open/run_config.json"
    cfg = json.loads(config_path.read_text())
    if torch.__version__.split("+")[0] != "2.8.0" or not torch.cuda.is_available():
        raise RuntimeError("Requires PyTorch 2.8.0 and an available CUDA GPU")
    if cfg.get("inference_mode") != "acceptance" or len(cfg.get("conversation_ids", [])) != 9:
        raise ValueError("Portable baseline only permits the bounded 9-item acceptance cohort")
    manifest = ROOT / cfg["manifest_path"]
    if digest(manifest) != cfg["manifest_sha256"] or digest(manifest) != cfg["frozen_cohort_sha256"]:
        raise ValueError("Bundled cohort manifest hash mismatch")
    rows = [json.loads(line) for line in manifest.read_text().splitlines()]
    if {row["conversation_id"] for row in rows} != set(cfg["conversation_ids"]) or len(rows) != 9:
        raise ValueError("Bundled manifest IDs differ from acceptance config")
    for row in rows:
        for path_key, hash_key in (("audio_path", "audio_sha256"), ("reference_path", "reference_sha256")):
            path = ROOT / row[path_key]
            if not path.is_file() or digest(path) != row[hash_key]:
                raise ValueError(f"Bundled input hash mismatch: {row['conversation_id']} {path_key}")
    # Both runtime and independent validator resolve staged paths from ROOT.
    os.chdir(ROOT)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    cfg["manifest_path"] = str(manifest)
    cfg["output_dir"] = str(ROOT / "artifacts" / cfg["cloud_job_id"])
    effective = ROOT / "effective_config.json"
    effective.write_text(json.dumps(cfg, indent=2) + "\n")
    output = Path(cfg["output_dir"])
    code = 1
    validation = {"valid": False, "reason": "runner did not reach independent validation"}
    failure = None
    try:
        code = run(effective)
        validation = validate(output)
        (output / "remote_validation.json").write_text(json.dumps(validation, indent=2) + "\n")
    except Exception as exc:
        failure = {"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()}
        output.mkdir(parents=True, exist_ok=True)
        (output / "worker_failure.json").write_text(json.dumps(failure, indent=2) + "\n")
        print(json.dumps({"event": "omni_acceptance_failure", "failure": failure}), flush=True)
    finally:
        receipt = dict(provider="huggingface", gpu=torch.cuda.get_device_name(0), torch_version=torch.__version__,
                       cuda_version=torch.version.cuda, elapsed_seconds=time.monotonic() - started, timeout_seconds=1800,
                       cloud_cost_usd=None, provider_bill_verified=False,
                       cost_note="Cost is null: elapsed setup/runtime is not a provider billing record.",
                       worker_failure=failure)
        if output.is_dir():
            (output / "hf_receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
            try:
                emit(output)
            except Exception as transport_exc:
                print(json.dumps({"event": "artifact_transport_failure", "failure": {
                    "type": type(transport_exc).__name__, "message": str(transport_exc),
                }}), flush=True)
        else:
            print(json.dumps({"event": "artifact_transport_failure", "failure": {
                "type": "OutputDirectoryUnavailable", "message": str(output),
            }}), flush=True)
    print(json.dumps({"event": "omni_acceptance_complete", "validation": validation, "receipt": receipt}), flush=True)
    return code if validation.get("valid") else 2


if __name__ == "__main__":
    raise SystemExit(main())
