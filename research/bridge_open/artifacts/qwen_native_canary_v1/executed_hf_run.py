"""Stage exact public FLEURS inputs, run Qwen, validate, and export evidence."""
import hashlib
import json
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    import torch
    from research.bridge_open.artifact_transport import emit
    from research.bridge_open.run_qwen import run
    from research.bridge_open.stage_fleurs import stage
    from research.bridge_open.validate_run import validate

    started = time.monotonic()
    cfg = json.loads((ROOT / 'research/bridge_open/run_config.json').read_text())
    if torch.__version__.split('+')[0] != '2.8.0' or not torch.cuda.is_available():
        raise RuntimeError('Requires PyTorch 2.8.0 and an available CUDA GPU')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    frozen = ROOT / cfg['source_manifest']
    if digest(frozen) != cfg['source_manifest_sha256']:
        raise ValueError('Frozen FLEURS v2 manifest hash mismatch')
    all_rows = [json.loads(line) for line in frozen.read_text().splitlines()]
    wanted = set(cfg['conversation_ids'])
    selected = [row for row in all_rows if row['conversation_id'] in wanted]
    if len(selected) != len(wanted) or not selected or {row['conversation_id'] for row in selected} != wanted or any(row['language'] != 'hi' for row in selected):
        raise ValueError('Expected exactly the configured Hindi official-test recordings')
    # Keep the manifest byte-for-byte identical to the frozen v2 source.
    # stage() writes each selected recording/reference to the original absolute
    # paths recorded in that manifest, preserving independent local replay.
    stage(selected, ROOT / 'research/bridge_open/frozen_references.json')
    manifest = Path(cfg['manifest_path'])
    manifest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(frozen, manifest)
    if digest(manifest) != cfg['manifest_sha256']:
        raise ValueError('Configured manifest no longer matches frozen v2 bytes')
    cfg['output_dir'] = str(ROOT / 'artifacts' / cfg['cloud_job_id'])
    effective = ROOT / 'effective_config.json'
    effective.write_text(json.dumps(cfg, indent=2) + '\n')
    code = run(effective)
    out = Path(cfg['output_dir'])
    validation = validate(out)
    (out / 'remote_validation.json').write_text(json.dumps(validation, indent=2) + '\n')
    receipt = dict(provider='huggingface', gpu=torch.cuda.get_device_name(0), torch_version=torch.__version__,
        cuda_version=torch.version.cuda, elapsed_seconds=time.monotonic() - started, timeout_seconds=1800,
        cloud_cost_usd=None, provider_bill_verified=False,
        cost_note='Cost is null: elapsed setup/runtime is not a provider-billing record.',
        source_sha256={str(path.relative_to(ROOT)): digest(path) for path in (Path(__file__), ROOT / 'experiments/hf_entrypoint.sh')})
    (out / 'hf_receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
    emit(out)
    print(json.dumps(dict(event='hf_run_complete', validation=validation, receipt=receipt)), flush=True)
    return code if validation['valid'] else 2


if __name__ == '__main__':
    raise SystemExit(main())
