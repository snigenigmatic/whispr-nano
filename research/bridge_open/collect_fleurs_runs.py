"""Recover completed ORX/HF evidence and update the study's job index."""
import hashlib
import json
import sqlite3
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from research.bridge_open.artifact_transport import recover
from research.bridge_open.validate_run import validate


def main():
    registry_path = ROOT/'research/bridge_open/artifacts/fleurs_run_registry.json'
    registry = json.loads(registry_path.read_text())
    validator_sha = hashlib.sha256((ROOT/'research/bridge_open/validate_run.py').read_bytes()).hexdigest()
    db = sqlite3.connect(Path.home()/'.local/share/openresearch/orx.db')
    db.row_factory = sqlite3.Row
    for entry in registry:
        if 'run_id' not in entry:
            continue
        run = dict(db.execute('select * from runs where id=?',(entry['run_id'],)).fetchone())
        entry['provider_run_status'] = run['status']
        verified_states = ('done_validated', 'validated_incomplete')
        if entry.get('status') not in verified_states:
            entry['status'] = run['status']
        if run['status'] not in ('done','failed') or (entry['status'] in verified_states and entry.get('validator_sha256') == validator_sha):
            continue
        out = Path(entry['output_dir'])
        log = Path.home()/'.local/share/openresearch/run-logs'/f"{entry['run_id']}.log"
        try:
            if not out.exists():
                transport = recover(log,out)
                (out/'transport_receipt.json').write_text(json.dumps(transport,indent=2)+'\n')
            validation = validate(out)
            validation['validator_sha256'] = validator_sha
            previous = out/'independent_validation.json'
            if previous.exists() and json.loads(previous.read_text()).get('validator_sha256') != validator_sha:
                old_sha = hashlib.sha256(previous.read_bytes()).hexdigest()
                archived = out/f'independent_validation_previous_{old_sha[:12]}.json'
                if not archived.exists():
                    archived.write_bytes(previous.read_bytes())
            previous.write_text(json.dumps(validation,indent=2)+'\n')
            backend = json.loads(run['backend_json'])
            receipt = dict(run_id=run['id'],commit=run['commit_sha'],
                **{k:backend.get(k) for k in ['jobId','namespace','url','sourceDigest','sourcePath','sourceSize']})
            source = Path(receipt['sourcePath'])
            assert hashlib.sha256(source.read_bytes()).hexdigest()==receipt['sourceDigest']
            (out/'orx_source_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
            if not validation['valid']:
                raise ValueError(validation['errors'])
            summary=json.loads((out/'summary.json').read_text())
            entry['scored_count'] = summary['scored']
            entry['runtime_failure_count'] = summary['attempted'] - summary['scored']
            entry['status'] = 'done_validated' if not entry['runtime_failure_count'] else 'validated_incomplete'
            entry['validator_sha256'] = validator_sha
            entry.pop('validation_error', None)
            entry['per_language']={lang:{k:r[k] for k in ['scored_count','micro_wer','micro_cer','generation_limit_count']}
                for lang,r in summary['per_language'].items()}
            print(json.dumps(dict(event='fleurs_run_validated',**entry)),flush=True)
        except Exception as exc:
            entry['status']='evidence_needs_review'
            entry['validation_error']=str(exc)
            print(json.dumps(dict(event='fleurs_run_needs_review',**entry)),flush=True)
    registry_path.write_text(json.dumps(registry,indent=2)+'\n')
    lines=['# FLEURS job index','','[Hugging Face Jobs dashboard](https://huggingface.co/jobs) — sign in as `snigenigmatic`.','','Open a job for live provider status and logs. This local index is a snapshot. Progress counts completed inference batches; final validation occurs after artifact recovery.','','An HF failure can coexist with verified exported predictions. See [the validator correction](FLEURS_VALIDATION_CORRECTION.md) for the completed Assamese Turbo run.','','| Run | Clips | Inference progress | Local evidence | Provider run | HF job |','|---|---:|---:|---|---|---|']
    for entry in registry:
        label=entry.get('cloud_job_id',entry['phase'])
        link=f"[Open](https://huggingface.co/jobs/snigenigmatic/{entry['job_id']})" if entry.get('job_id') else 'Pending dispatch'
        count = entry.get('conversation_count', 36)
        progress = str(entry.get('scored_count', count)) if entry['status'] in ('done_validated', 'validated_incomplete') else 'Starting'
        if entry.get('run_id') and entry['status'] not in ('done_validated', 'validated_incomplete'):
            log = Path.home()/'.local/share/openresearch/run-logs'/f"{entry['run_id']}.log"
            completed = set()
            for line in log.read_text(errors='replace').splitlines() if log.exists() else []:
                if 'utterance_batch_done' in line:
                    try:
                        event = json.loads(line[line.index('{'):])
                        completed.update(event['conversation_ids'])
                    except (ValueError, KeyError):
                        continue
            if completed:
                progress = str(len(completed))
        lines.append(f"| {label} | {count} | {progress} | {entry['status']} | {entry.get('provider_run_status', 'pending')} | {link} |")
    (ROOT/'research/bridge_open/FLEURS_JOBS.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({'states':{e.get('cloud_job_id',e['phase']):e['status'] for e in registry}}))


if __name__ == '__main__':
    main()
