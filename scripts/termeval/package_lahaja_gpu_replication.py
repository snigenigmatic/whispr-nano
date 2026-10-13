"""Extract aggregate-only HF GPU evidence from the completed ORX run log."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
LOG = Path.home() / ".local/share/openresearch/run-logs/5f3ff635-6b53-4e85-a2a0-b655423caab3.log"
MPS = Path.home() / ".local/share/openresearch/lahaja-small-beam5-heldout-405-v1/summary.json"
OUT = ROOT / "results/night1/lahaja_beam5_hf_gpu_replication_20261013.json"
SELECTION_SHA256 = "6134e2a9eabd65f590bf4f7f7829626e520379a66e058d9dd51ce4acf36c4060"
MODEL_REVISION = "973afd24965f72e36ca33b3055d56a652f456b4d"
DATASET_REVISION = "d4ffd2ecbdd933e37c917ddcf620eef159ceb3a7"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    log = LOG.read_text()
    if log.count("===== RESULT SUMMARY =====") != 1 or log.count("==========================") != 1:
        raise ValueError("incomplete or ambiguous final ORX summary")
    preflights = [json.loads(line) for line in log.splitlines()
                  if line.startswith("{") and '"event": "preflight"' in line]
    if len(preflights) != 1:
        raise ValueError("missing or duplicate preflight event")
    preflight = preflights[0]
    summary_text = log.split("===== RESULT SUMMARY =====", 1)[1].split("==========================", 1)[0]
    gpu = json.loads(summary_text)
    mps = json.loads(MPS.read_text())
    for key, expected in {
        "status": "pilot_complete",
        "dataset_revision": DATASET_REVISION,
        "selection_sha256": SELECTION_SHA256,
        "model": "openai/whisper-small",
        "model_revision": MODEL_REVISION,
        "device": "cuda",
        "dtype": "torch.float16",
        "language": "hi",
        "task": "transcribe",
        "num_beams": 5,
        "max_new_tokens": 256,
    }.items():
        if gpu[key] != expected:
            raise ValueError(f"GPU {key} changed")
    for key in ("dataset_revision", "selection_sha256", "model", "model_revision", "dtype",
                "language", "task", "num_beams", "max_new_tokens"):
        if gpu[key] != mps[key]:
            raise ValueError(f"MPS versus GPU {key} mismatch")
    for key, expected in {
        "dataset_revision": DATASET_REVISION,
        "selection_sha256": SELECTION_SHA256,
        "selected_rows": 405,
        "selected_speakers": 106,
        "selected_states": 17,
        "model_revision": MODEL_REVISION,
    }.items():
        if preflight[key] != expected:
            raise ValueError(f"preflight {key} changed")
    for run in (gpu, mps):
        overall = run["overall"]
        if (overall["clips"], overall["speakers"], overall["ref_words"]) != (405, 106, 7339):
            raise ValueError("cohort counts changed")
        if abs(overall["wer"] - overall["word_errors"] / 7339) > 1e-12:
            raise ValueError("pooled WER arithmetic changed")
    result = {
        "status": "completed_aggregate_only_cloud_replication",
        "dataset": "ai4bharat/Lahaja",
        "dataset_revision": DATASET_REVISION,
        "selection_sha256": SELECTION_SHA256,
        "model": "openai/whisper-small",
        "model_revision": MODEL_REVISION,
        "num_beams": 5,
        "clips": 405,
        "speakers": 106,
        "reference_words": 7339,
        "huggingface": {
            "job_url": "https://huggingface.co/jobs/snigenigmatic/6ace098a24c0e0c53d7cea30",
            "orx_run_id": "5f3ff635-6b53-4e85-a2a0-b655423caab3",
            "code_commit": "7f06b432d975d6c5b2a32d951d4b414e7810342c",
            "device": "cuda",
            "gpu_class": "a100-large",
            "torch_version": preflight["torch"],
            "transformers_version": preflight["transformers"],
            "word_errors": gpu["overall"]["word_errors"],
            "wer_percent": 100 * gpu["overall"]["wer"],
            "generation_cap_count": gpu["decoder_quality"]["generated_token_cap_256_count"],
            "runner_elapsed_seconds": gpu["elapsed_s"],
            "log_sha256": _sha(LOG),
        },
        "local_mps": {
            "orx_run_id": "36217bd3-5caa-48ea-8521-0660ca15d7f0",
            "device": "mps",
            "word_errors": mps["overall"]["word_errors"],
            "wer_percent": 100 * mps["overall"]["wer"],
            "generation_cap_count": mps["decoder_quality"]["generated_token_cap_256_count"],
            "runner_elapsed_seconds": mps["elapsed_s"],
        },
        "gpu_minus_mps_wer_points": 100 * (gpu["overall"]["word_errors"] - mps["overall"]["word_errors"]) / 7339,
        "limits": [
            "ORX records the full cloud summary and preflight, but this run did not export its per-clip GPU predictions; we cannot independently re-score or form a paired GPU-versus-MPS interval.",
            "The two runs have different PyTorch versions and hardware; this aggregate discrepancy is unresolved and must not be called bitwise parity.",
            "The cloud runner time includes model/data setup and is not a controlled inference-latency benchmark or a verified bill.",
            "The MPS run remains the source of the independently validated per-clip comparison with greedy decoding and frozen repair.",
        ],
        "packaging_script_sha256": _sha(Path(__file__)),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"output": str(OUT), "gpu_wer_percent": result["huggingface"]["wer_percent"],
                      "gpu_minus_mps_wer_points": result["gpu_minus_mps_wer_points"]}))


if __name__ == "__main__":
    main()
