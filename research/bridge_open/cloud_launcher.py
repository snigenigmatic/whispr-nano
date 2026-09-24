"""Reserve first, upload public data, dispatch one bounded H100 evaluation."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def main():
    from research.bridge_open.cloud_budget import BudgetLedger
    config_path = ROOT/'research/bridge_open/run_config.json'
    config = json.loads(config_path.read_text())
    if config.get('model_family') == 'omni_ctc':
        from research.bridge_open.modal_omni_worker import app, data_volume, evaluate, DATA_MOUNT
    elif config.get('model_family') == 'qwen3_asr':
        from research.bridge_open.modal_qwen_worker import app, data_volume, evaluate, DATA_MOUNT
    else:
        from research.bridge_open.modal_worker import app, data_volume, evaluate, DATA_MOUNT
    job_id = config['cloud_job_id']
    config_hash = hashlib.sha256(config_path.read_bytes()).hexdigest()
    ledger = BudgetLedger()
    # $1.50 covers the 900s execution cap, 180s startup and conservative overhead.
    ledger.reserve(job_id, 1.50, config_hash)
    manifest = Path(config['manifest_path'])
    rows = [json.loads(line) for line in manifest.read_text().splitlines()]
    selected = [r for r in rows if r['conversation_id'] in config['conversation_ids']]
    if len(selected) != len(config['conversation_ids']):
        raise ValueError('Upload cohort mismatch')
    paths = [manifest] + [Path(r[k]) for r in selected for k in ['audio_path','reference_path']]
    # Reuse only when the committed configuration opts in. The worker still
    # verifies the frozen manifest/audio/reference hashes before accepting output.
    if not config.get('reuse_verified_volume', False):
        with data_volume.batch_upload(force=True) as upload:
            for path in paths:
                upload.put_file(str(path), '/'+str(path.relative_to(DATA_MOUNT)))
    print(json.dumps({'event':'reserved_and_uploaded', 'job_id':job_id,
                      'reserved_usd':1.5, 'new_spending_cap_usd':15, 'files':len(paths), 'reused_verified_volume':config.get('reuse_verified_volume', False)}), flush=True)
    with app.run():
        result = evaluate.remote(config)
    out = ROOT/'research/bridge_open/artifacts'/job_id
    out.mkdir(parents=True, exist_ok=False)
    for name, contents in result['files'].items():
        if Path(name).name != name:
            raise ValueError('Invalid artifact name')
        (out/name).write_text(contents)
    print(json.dumps({'event':'effective_config', 'config':json.loads(result['files']['config.json'])}), flush=True)
    print(json.dumps({'event':'validated_pending_summary', 'summary':json.loads(result['files']['summary.json'])}), flush=True)
    receipt_text = json.dumps(result['receipt'], indent=2)+'\n'
    (out/'cloud_receipt.json').write_text(receipt_text)
    ledger.record(job_id, 'completed' if result['receipt']['exit_code']==0 else 'failed',
                  result['receipt']['estimated_function_resource_usd'],
                  hashlib.sha256(receipt_text.encode()).hexdigest())
    print(json.dumps({'event':'cloud_run_finished', 'artifact_dir':str(out),
                      **result['receipt']}), flush=True)
    return result['receipt']['exit_code']
