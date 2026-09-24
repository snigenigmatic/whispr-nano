"""Bounded H100 worker for the pinned OmniASR CTC pilot.

This module only defines the Modal app.  It performs no reservation or launch;
the existing launcher or an explicit operator action must call ``evaluate``.
"""
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parents[2]
DATA_MOUNT = "/Users/prateekraj/whispr-nano/data/bridge_open"
DATA_VOLUME = "whispr-bridge-public-data-v1"
app = modal.App("whispr-bridge-omni-ctc-pilot")
data_volume = modal.Volume.from_name(DATA_VOLUME, create_if_missing=True)
cache_volume = modal.Volume.from_name("ast-asr-cache", create_if_missing=True)

# Versions are constrained by the official omnilingual-asr pyproject at the
# pinned source revision. fairseq2 0.6.0 is the upper compatible release.
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git", "build-essential", "cmake", "libsndfile1")
    .pip_install(
        "torch==2.8.0",
        "torchaudio==2.8.0",
        "fairseq2==0.6.0",
        "fairseq2n==0.6",
        "numpy==1.26.4",
        "scipy==1.15.3",
        "soundfile==0.13.1",
        "jiwer==4.0.0",
        "huggingface-hub==0.36.2",
        "omnilingual-asr @ git+https://github.com/facebookresearch/omnilingual-asr.git@81f51e224ce9e74b02cc2a3eaf21b2d91d743455",
    )
    .env({"HF_HOME": "/cache/huggingface", "TOKENIZERS_PARALLELISM": "false"})
    .add_local_file(ROOT / "research/bridge_open/run_omni.py", "/root/project/research/bridge_open/run_omni.py")
    .add_local_file(ROOT / "research/bridge_open/scoring.py", "/root/project/research/bridge_open/scoring.py")
)


@app.function(
    image=image,
    gpu="H100",
    cpu=(4, 4),
    memory=(16384, 16384),
    timeout=900,
    startup_timeout=180,
    retries=0,
    max_containers=1,
    scaledown_window=2,
    volumes={DATA_MOUNT: data_volume, "/cache/huggingface": cache_volume},
)
def evaluate(config: dict) -> dict:
    import json
    import sys
    import time
    from pathlib import Path

    import torch

    sys.path.insert(0, "/root/project")
    from research.bridge_open.run_omni import run

    started = time.monotonic()
    if config.get("device") != "cuda" or config.get("max_wall_seconds", 9999) > 780:
        raise ValueError("Remote configuration exceeds fixed Omni worker contract")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    path = Path("/tmp/bridge_omni_config.json")
    config = {**config, "output_dir": f"/tmp/{config['cloud_job_id']}"}
    path.write_text(json.dumps(config, indent=2) + "\n")
    exit_code = run(path)
    elapsed = time.monotonic() - started
    files = {p.name: p.read_text() for p in Path(config["output_dir"]).iterdir() if p.is_file()}
    rate = 0.001097 + 4 * 0.0000131 + 16 * 0.00000222
    receipt = dict(
        exit_code=exit_code,
        function_wall_seconds=elapsed,
        estimated_function_resource_usd=elapsed * rate,
        gpu=torch.cuda.get_device_name(0),
        cpu_limit=4,
        memory_limit_gib=16,
        timeout_seconds=900,
        pricing_source="https://modal.com/pricing",
        provider_bill_verified=False,
        model_id=config.get("model_id"),
        model_revision=config.get("model_revision"),
    )
    print(json.dumps({"event": "cloud_receipt", **receipt}), flush=True)
    cache_volume.commit()
    return {"files": files, "receipt": receipt}
