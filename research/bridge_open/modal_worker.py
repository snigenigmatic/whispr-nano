"""Bounded H100 worker, dispatched only through the committed orx launcher."""
from pathlib import Path
import modal

ROOT = Path(__file__).resolve().parents[2]
DATA_MOUNT = '/Users/prateekraj/whispr-nano/data/bridge_open'
DATA_VOLUME = 'whispr-bridge-public-data-v1'
app = modal.App('whispr-bridge-open-repair')
data_volume = modal.Volume.from_name(DATA_VOLUME, create_if_missing=True)
cache_volume = modal.Volume.from_name('ast-asr-cache', create_if_missing=True)
image = (
    modal.Image.debian_slim(python_version='3.12')
    .pip_install('torch==2.8.0', 'transformers==4.57.6', 'numpy==2.2.6',
                 'scipy==1.15.3', 'soundfile==0.13.1', 'jiwer==4.0.0',
                 'huggingface-hub==0.36.2')
    .env({'HF_HOME':'/cache/huggingface', 'TOKENIZERS_PARALLELISM':'false'})
    .add_local_file(ROOT/'research/bridge_open/run_whisper.py', '/root/project/research/bridge_open/run_whisper.py')
    .add_local_file(ROOT/'research/bridge_open/scoring.py', '/root/project/research/bridge_open/scoring.py')
    .add_local_file(ROOT/'research/bridge_open/generation_output.py', '/root/project/research/bridge_open/generation_output.py')
    .add_local_file(ROOT/'research/bridge_open/native_whisper.py', '/root/project/research/bridge_open/native_whisper.py')
)


@app.function(image=image, gpu='H100', cpu=(4, 4), memory=(16384, 16384),
              timeout=900, startup_timeout=180, retries=0, max_containers=1,
              scaledown_window=2, volumes={DATA_MOUNT:data_volume, '/cache/huggingface':cache_volume})
def evaluate(config: dict) -> dict:
    import json
    import sys
    import time
    import torch
    sys.path.insert(0, '/root/project')
    from research.bridge_open.run_whisper import run
    started = time.monotonic()
    if config['device'] != 'cuda' or config.get('max_wall_seconds', 9999) > 780:
        raise ValueError('Remote configuration exceeds fixed worker contract')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    path = Path('/tmp/bridge_config.json')
    config = {**config, 'output_dir':f"/tmp/{config['cloud_job_id']}"}
    path.write_text(json.dumps(config, indent=2)+'\n')
    exit_code = run(path)
    elapsed = time.monotonic()-started
    files = {p.name:p.read_text() for p in Path(config['output_dir']).iterdir() if p.is_file()}
    # Published task rates checked 2026-09-22; excludes startup/image overhead.
    rate = 0.001097 + 4*0.0000131 + 16*0.00000222
    receipt = dict(exit_code=exit_code, function_wall_seconds=elapsed,
                   estimated_function_resource_usd=elapsed*rate, gpu=torch.cuda.get_device_name(0),
                   cpu_limit=4, memory_limit_gib=16, timeout_seconds=900,
                   pricing_source='https://modal.com/pricing', provider_bill_verified=False)
    print(json.dumps({'event':'cloud_receipt', **receipt}), flush=True)
    cache_volume.commit()
    return {'files':files, 'receipt':receipt}
