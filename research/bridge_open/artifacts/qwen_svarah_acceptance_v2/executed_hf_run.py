"""Verify a frozen native-input bundle, run Qwen, and export evidence."""
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


def validate_svarah_bundle(cfg):
    """Validate the frozen 12-record English bundle without touching CUDA."""
    if cfg.get('input_transport') != 'native_svarah_bundle':
        raise ValueError('This child requires input_transport=native_svarah_bundle')
    if cfg.get('supported_languages') != ['en'] or cfg.get('transcribe_language') != 'English':
        raise ValueError('This child requires supported_languages=["en"] and transcribe_language="English"')
    if cfg.get('reference_qc_scope') != 'structural_not_human':
        raise ValueError('reference_qc_scope must be structural_not_human')
    frozen = Path(cfg['source_manifest'])
    if not frozen.is_absolute(): frozen = (ROOT / frozen).resolve()
    if digest(frozen) != cfg['source_manifest_sha256']:
        raise ValueError('Frozen source manifest hash mismatch')
    all_rows = [json.loads(line) for line in frozen.read_text().splitlines()]
    wanted_ids = cfg['conversation_ids']; wanted = set(wanted_ids)
    if len(wanted_ids) != 12 or len(wanted) != 12 or len(all_rows) != 12:
        raise ValueError('Svarah bundle must contain exactly 12 configured unique recordings')
    if {row['conversation_id'] for row in all_rows} != wanted:
        raise ValueError('Configured IDs must cover the entire 12-record manifest')
    if any(row.get('language') != 'en' or row.get('reference_qc', row.get('dataset_reference_qc')) != 'pass' for row in all_rows):
        raise ValueError('Every Svarah row must be English with structural reference_qc=pass')
    for row in all_rows:
        for key, expected in (('audio_path', row['audio_sha256']), ('reference_path', row['reference_sha256'])):
            raw = Path(row[key])
            if raw.is_absolute(): raise ValueError(f'{key} must be repository-relative')
            path = (ROOT / raw).resolve()
            if ROOT not in path.parents: raise ValueError(f'{key} escapes repository bundle: {raw}')
            if not path.is_file(): raise FileNotFoundError(path)
            if digest(path) != expected: raise ValueError(f'{key} hash mismatch: {path}')
    manifest = Path(cfg['manifest_path'])
    if not manifest.is_absolute(): manifest = (ROOT / manifest).resolve()
    manifest.parent.mkdir(parents=True, exist_ok=True)
    if manifest != frozen: shutil.copy2(frozen, manifest)
    if digest(manifest) != cfg['manifest_sha256']:
        raise ValueError('Configured manifest hash mismatch')
    return frozen, manifest, all_rows


def main():
    import torch
    from research.bridge_open.artifact_transport import emit
    from research.bridge_open.run_qwen import run
    from research.bridge_open.validate_run import validate

    started = time.monotonic()
    cfg = json.loads((ROOT / 'research/bridge_open/run_config.json').read_text())
    out = ROOT / 'artifacts' / cfg.get('cloud_job_id', 'qwen-preflight-failure')
    try:
        if torch.__version__.split('+')[0] != '2.8.0' or not torch.cuda.is_available():
            raise RuntimeError('Requires PyTorch 2.8.0 and an available CUDA GPU')
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        frozen, manifest, all_rows = validate_svarah_bundle(cfg)
        cfg['manifest_path'] = str(manifest)
        cfg['output_dir'] = str(out)
        effective = ROOT / 'effective_config.json'
        effective.write_text(json.dumps(cfg, indent=2) + '\n')
        code = run(effective)
        validation = validate(out)
        (out / 'remote_validation.json').write_text(json.dumps(validation, indent=2) + '\n')
        receipt = dict(provider='huggingface', gpu=torch.cuda.get_device_name(0), torch_version=torch.__version__,
        cuda_version=torch.version.cuda, elapsed_seconds=time.monotonic() - started, timeout_seconds=1800,
        cloud_cost_usd=None, provider_bill_verified=False,
        cost_note='Cost is null: elapsed setup/runtime is not a provider-billing record.',
        source_sha256={str(path.relative_to(ROOT)): digest(path) for path in (Path(__file__), ROOT / 'experiments/hf_entrypoint.sh')})
        (out / 'hf_receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
        try:
            emit(out)
        except Exception as exc:
            (out / 'artifact_transport_failure.json').write_text(json.dumps({'status':'transport_failure','type':type(exc).__name__,'message':str(exc)}, indent=2) + '\n')
            print(json.dumps({'event':'artifact_transport_failure','type':type(exc).__name__,'message':str(exc)}), flush=True)
            return 3
        print(json.dumps(dict(event='hf_run_complete', validation=validation, receipt=receipt)), flush=True)
        return code if validation['valid'] else 2
    except Exception as exc:
        out.mkdir(parents=True, exist_ok=True)
        failure = {'valid':False,'output_dir':str(out),'errors':[f'preflight_or_worker_failure: {type(exc).__name__}: {exc}']}
        (out / 'remote_validation.json').write_text(json.dumps(failure, indent=2) + '\n')
        (out / 'hf_receipt.json').write_text(json.dumps({'provider':'huggingface','status':'failed','elapsed_seconds':time.monotonic()-started,'cloud_cost_usd':None,'provider_bill_verified':False}, indent=2) + '\n')
        print(json.dumps({'event':'hf_run_failure','validation':failure}), flush=True)
        try: emit(out)
        except Exception as transport_exc:
            (out / 'artifact_transport_failure.json').write_text(json.dumps({'status':'transport_failure','type':type(transport_exc).__name__,'message':str(transport_exc)}, indent=2) + '\n')
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
