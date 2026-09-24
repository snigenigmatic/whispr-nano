"""Run the frozen study on HF Jobs and export complete verifiable evidence."""
import hashlib
import json
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    import torch
    from research.bridge_open import acquire
    from research.bridge_open.artifact_transport import emit
    from research.bridge_open.run_whisper import run, verify
    from research.bridge_open.validate_run import validate
    started = time.monotonic()
    cfg = json.loads((ROOT/'research/bridge_open/run_config.json').read_text())
    if torch.__version__.split('+')[0] != '2.8.0' or not torch.cuda.is_available():
        raise RuntimeError('Requires PyTorch 2.8.0 and an available CUDA GPU')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    manifest = Path(cfg['manifest_path'])
    frozen = ROOT/'research/bridge_open/frozen_corpus_v1.jsonl'
    verify(frozen, cfg['manifest_sha256'])
    rows = [json.loads(line) for line in frozen.read_text().splitlines()]
    selected = [r for r in rows if r['conversation_id'] in cfg['conversation_ids']]
    if len(selected) != len(cfg['conversation_ids']):
        raise ValueError('Frozen cohort mismatch')
    # Preserve original manifest bytes and paths to permit identical local replay.
    acquire.DATA = manifest.parent
    acquire.DATA.mkdir(parents=True, exist_ok=True)
    for row in selected:
        for key, hash_key in [('audio_path', 'audio_sha256'), ('reference_path', 'reference_sha256')]:
            source = ROOT/'frozen_inputs'/Path(row[key]).name
            verify(source, row[hash_key])
            shutil.copy2(source, row[key])
    shutil.copy2(frozen, manifest)
    cfg['output_dir'] = str(ROOT/'artifacts'/cfg['cloud_job_id'])
    effective = ROOT/'effective_config.json'
    effective.write_text(json.dumps(cfg, indent=2)+'\n')
    code = run(effective)
    out = Path(cfg['output_dir'])
    validation = validate(out)
    (out/'remote_validation.json').write_text(json.dumps(validation, indent=2)+'\n')
    receipt = dict(provider='huggingface', gpu=torch.cuda.get_device_name(0),
                   torch_version=torch.__version__, cuda_version=torch.version.cuda,
                   elapsed_seconds=time.monotonic()-started, timeout_seconds=1800,
                   hourly_price_usd=2.50, provider_bill_verified=False,
                   cost_note='Job Starting/Running time includes installation; this elapsed time is not the bill.',
                   source_sha256={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                                  for p in [Path(__file__), ROOT/'experiments/hf_entrypoint.sh']})
    (out/'hf_receipt.json').write_text(json.dumps(receipt, indent=2)+'\n')
    emit(out)
    print(json.dumps(dict(event='hf_run_complete', validation=validation, receipt=receipt)), flush=True)
    return code if validation['valid'] else 2


if __name__ == '__main__':
    raise SystemExit(main())
