"""Run a frozen Svarah input-acceptance cohort through the accepted native CTC code."""
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def check_inputs(cfg):
    if cfg['source_dataset_id'] != 'ai4bharat/Svarah' or cfg['source_dataset_revision'] != 'ebbf7777fe771490696a3f7b007097606fa8c924':
        raise ValueError('Unexpected source dataset or revision')
    if cfg['inference_mode'] != 'acceptance' or cfg['supported_languages'] != ['en'] or cfg['expected_count'] != 12:
        raise ValueError('Expected 12-record native English acceptance condition')
    path = ROOT / cfg['manifest_path']
    if sha(path) != cfg['manifest_sha256']:
        raise ValueError('Frozen Svarah manifest hash mismatch')
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    ids = [r['conversation_id'] for r in rows]
    if len(ids) != len(set(ids)) or len(ids) != 12 or ids != cfg['conversation_ids']:
        raise ValueError('Configured cohort does not exactly match frozen manifest')
    for r in rows:
        if r['language'] != 'en' or r['reference_qc_scope'] != 'structural_not_human':
            raise ValueError('Language or reference-QC scope mismatch')
        for field, hashfield in [('audio_path', 'audio_sha256'), ('reference_path', 'reference_sha256')]:
            item = Path(r[field])
            if item.is_absolute() or '..' in item.parts or item.parts[:3] != ('research', 'bridge_open', 'svarah_inputs'):
                raise ValueError('Input path outside committed bundle')
            if sha(ROOT / item) != r[hashfield]:
                raise ValueError('Input content hash mismatch')
    return path


def main():
    import torch
    from research.bridge_open.omni_extension.run_omni_full import run
    from research.bridge_open.omni_extension.validate_run import validate
    from research.bridge_open.omni_extension.artifact_transport import emit
    started = time.monotonic()
    os.chdir(ROOT)
    source = ROOT / 'research/bridge_open/run_config.json'
    cfg = json.loads(source.read_text())
    out = ROOT / 'artifacts' / cfg['cloud_job_id']
    cfg['output_dir'] = str(out)
    validation = {'valid': False, 'errors': ['Run did not reach validation']}
    code = 1
    failure = None
    try:
        if torch.__version__.split('+')[0] != '2.8.0' or not torch.cuda.is_available():
            raise RuntimeError('Expected CUDA and pinned PyTorch 2.8.0')
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        frozen = check_inputs(cfg)
        cfg['manifest_path'] = str(frozen)
        effective = ROOT / 'effective_svarah_config.json'
        effective.write_text(json.dumps(cfg, indent=2) + '\n')
        code = run(effective)
        validation = validate(out)
        predictions = [json.loads(s) for s in (out/'predictions.jsonl').read_text().splitlines()]
        agreements = sum(r.get('api_token_agreement') is True for r in predictions)
        if agreements != 12:
            validation['valid'] = False
            validation.setdefault('errors', []).append('Not all 12 outputs match public native API')
        validation['api_token_agreements'] = agreements
    except Exception as exc:
        failure = dict(type=type(exc).__name__, message=str(exc), traceback=traceback.format_exc())
        print(json.dumps(dict(event='svarah_acceptance_failure', failure=failure)), flush=True)
    finally:
        out.mkdir(parents=True, exist_ok=True)
        receipt = dict(provider='huggingface', gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            torch_version=torch.__version__, cuda_version=torch.version.cuda, elapsed_seconds=time.monotonic()-started,
            worker_failure=failure, cloud_cost_usd=None, provider_bill_verified=False,
            source_sha256={str(p.relative_to(ROOT)): sha(p) for p in [Path(__file__),source,ROOT/'experiments/hf_entrypoint.sh']},
            cost_note='Elapsed runtime is not a provider billing record.')
        (out/'hf_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
        (out/'remote_validation.json').write_text(json.dumps(validation,indent=2)+'\n')
        emit(out)
        print(json.dumps(dict(event='svarah_acceptance_complete', validation=validation, receipt=receipt)),flush=True)
    return code if validation['valid'] else 2


if __name__ == '__main__':
    raise SystemExit(main())
