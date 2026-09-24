"""Run the bounded Omni resource-profile panel inside the HF container."""
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
    from research.bridge_open.core_study.resource_profile import FIXED_PANEL_IDS, FIXED_PANEL_MANIFEST_SHA256, read_fixed_panel
    if cfg.get('experiment_kind') != 'core_resource_profile' or cfg.get('expected_count') != 12:
        raise ValueError('Expected the fixed 12-record resource profile')
    if cfg.get('timeout_seconds') != 1800 or cfg.get('gpu_flavor') != 'a100-large':
        raise ValueError('Expected bounded 30-minute A100-large job contract')
    if cfg.get('gpu_contract') != 'NVIDIA A100 80GB':
        raise ValueError('Expected NVIDIA A100 80GB contract')
    if cfg.get('batch_size') != 1 or cfg.get('dtype') != 'bfloat16':
        raise ValueError('Expected BF16 batch-one resource condition')
    if cfg.get('public_api') != 'ASRInferencePipeline.transcribe' or cfg.get('public_api_calls_per_input_per_phase') != 1:
        raise ValueError('Expected exactly one official public API call per input phase')
    path = ROOT / cfg['manifest_path']
    if sha(path) != FIXED_PANEL_MANIFEST_SHA256 or cfg.get('manifest_sha256') != FIXED_PANEL_MANIFEST_SHA256:
        raise ValueError('Frozen Svarah manifest hash mismatch')
    rows = read_fixed_panel(path)
    ids = [r['conversation_id'] for r in rows]
    if len(ids) != len(set(ids)) or tuple(ids) != FIXED_PANEL_IDS or ids != cfg['conversation_ids']:
        raise ValueError('Configured cohort does not exactly match frozen manifest')
    for r in rows:
        if r['language'] != 'en' or r['reference_qc_scope'] != 'structural_not_human':
            raise ValueError('Language or reference-QC scope mismatch')
        for field, hashfield in [('audio_path', 'audio_sha256'), ('reference_path', 'reference_sha256')]:
            item = path.parent / ('wav' if field == 'audio_path' else 'references') / Path(r[field]).name
            if not item.is_file() or sha(item) != r[hashfield]:
                raise ValueError('Input content hash mismatch')
    return path


def validate_resource_artifact(out):
    from research.bridge_open.core_study.resource_profile import FIXED_PANEL_IDS, SCHEMA_VERSION
    required = ['config.json', 'environment.json', 'resource_profile.json', 'quality_predictions.jsonl', 'quality_summary.json']
    errors = [f'missing artifact: {name}' for name in required if not (out / name).is_file()]
    if errors:
        return {'valid': False, 'errors': errors}
    try:
        profile = json.loads((out / 'resource_profile.json').read_text())
        quality = [json.loads(line) for line in (out / 'quality_predictions.jsonl').read_text().splitlines()]
    except Exception as exc:
        return {'valid': False, 'errors': [f'invalid result JSON: {exc}']}
    if profile.get('schema_version') != SCHEMA_VERSION:
        errors.append('unexpected resource-profile schema')
    if profile.get('study', {}).get('conversation_ids') != list(FIXED_PANEL_IDS):
        errors.append('profile IDs differ from fixed panel')
    if len(profile.get('warmup_calls', [])) != 12 or len(profile.get('timed_calls', [])) != 36:
        errors.append('profile does not contain one warmup and three complete timed passes')
    ids = [record.get('conversation_id') for record in quality]
    if ids != list(FIXED_PANEL_IDS) or len(ids) != len(set(ids)):
        errors.append('quality output must contain each fixed reference exactly once in panel order')
    if any(record.get('timed_pass_index') != 3 for record in quality):
        errors.append('quality output must use final timed pass only')
    return {'valid': not errors, 'errors': errors, 'timed_calls': len(profile.get('timed_calls', [])), 'quality_records': len(quality)}


def main():
    import torch
    from research.bridge_open.core_study.run_omni_resource import run
    from research.bridge_open.omni_extension.artifact_transport import emit
    started = time.monotonic()
    os.chdir(ROOT)
    source = ROOT / 'research/bridge_open/core_study/omni_resource_config.json'
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
        run(effective)
        validation = validate_resource_artifact(out)
        code = 0 if validation['valid'] else 1
    except Exception as exc:
        failure = dict(type=type(exc).__name__, message=str(exc), traceback=traceback.format_exc())
        print(json.dumps(dict(event='svarah_acceptance_failure', failure=failure)), flush=True)
    finally:
        out.mkdir(parents=True, exist_ok=True)
        receipt = dict(provider='huggingface', gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            torch_version=torch.__version__, cuda_version=torch.version.cuda, elapsed_seconds=time.monotonic()-started,
            worker_failure=failure, cloud_cost_usd=None, provider_bill_verified=False,
            source_sha256={str(p.relative_to(ROOT)): sha(p) for p in [Path(__file__),source,ROOT/'experiments/hf_entrypoint.sh', ROOT/'research/bridge_open/core_study/run_omni_resource.py', ROOT/'research/bridge_open/core_study/resource_profile.py']},
            cost_note='Elapsed runtime is not a provider billing record.')
        (out/'hf_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
        container = dict(schema_version='container-receipt/v1', command='bash experiments/hf_entrypoint.sh',
                         image='pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime', gpu_flavor='a100-large',
                         timeout_seconds=1800, worker_failure=failure, validation=validation)
        (out/'container_receipt.json').write_text(json.dumps(container,indent=2)+'\n')
        (out/'remote_validation.json').write_text(json.dumps(validation,indent=2)+'\n')
        transport = dict(schema_version='artifact-transport/v1', transport='zlib/base64 JSON chunks emitted to job logs',
                         top_level=True, source_artifact_directory=str(out), includes_container_receipt=True)
        (out/'artifact_transport.json').write_text(json.dumps(transport,indent=2)+'\n')
        emit(out)
        print(json.dumps(dict(event='svarah_acceptance_complete', validation=validation, receipt=receipt)),flush=True)
    return code if validation['valid'] else 2


if __name__ == '__main__':
    raise SystemExit(main())
