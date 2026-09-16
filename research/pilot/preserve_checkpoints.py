"""Download already-produced, audited checkpoints; never start compute.

Uses individual Modal volume files to avoid recursive directory-download errors.
All completed downloads have SHA256 receipts and are reused after restart.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import subprocess
import time
import zipfile

ROOT = Path(__file__).resolve().parents[2]
PILOT = ROOT / 'research/pilot'
OUTPUT = ROOT / 'output/checkpoints'
VOLUME = 'accent-selection-pilot-results'
MODAL = ROOT / '.venv/bin/modal'
REQUIRED = {'model.safetensors', 'config.json', 'generation_config.json',
            'processor_config.json', 'tokenizer.json', 'tokenizer_config.json'}


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def write(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp')
    with temporary.open('w') as stream:
        stream.write(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')
        stream.flush()
        import os
        os.fsync(stream.fileno())
    temporary.replace(path)


def checkpoint_source(name, planned):
    directory = PILOT / 'runs' / name
    summary_path, audit_path = directory / 'summary.json', directory / 'workflow_audit.json'
    if not summary_path.is_file() or not audit_path.is_file():
        return None
    # The controller writes large evidence files before appending its terminal
    # event. Do not race a partly written summary/audit merely because it exists.
    event_path = PILOT / 'ledger/events.jsonl'
    passed = False
    if event_path.is_file():
        for line in event_path.read_text().splitlines():
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if event.get('type') == 'recorded' and event.get('job_id') == name and event.get('outcome') == 'passed':
                passed = True
    if not passed:
        return None
    summary, audit = read(summary_path), read(audit_path)
    if summary.get('status') != 'completed' or audit.get('matching_registered_run') is not True:
        return None
    runs = audit.get('audit', {}).get('runs', [])
    if len(runs) != 1 or (runs[0]['arm_name'], runs[0]['seed']) != (planned['arm_name'], planned['seed']):
        raise ValueError(f'{name}: independent audit identifies another run')
    if (summary['config']['arm_name'], summary['config']['seed']) != (planned['arm_name'], planned['seed']):
        raise ValueError(f'{name}: summary identifies another run')
    summary_sha = sha(summary_path)
    if runs[0]['artifact_sha256']['summary.json'] != summary_sha:
        raise ValueError(f'{name}: summary changed since independent audit')
    expected = summary.get('checkpoint_model_sha256', '')
    if not re.fullmatch(r'[0-9a-f]{64}', expected):
        raise ValueError(f'{name}: missing exact saved model SHA256')
    return {'name': name, **planned, 'summary_path': str(summary_path),
            'summary_sha256': summary_sha, 'audit_path': str(audit_path),
            'audit_sha256': sha(audit_path), 'expected_model_sha256': expected,
            'model_id': summary['config']['model_id'], 'model_revision': summary['config']['model_revision'],
            'runtime': summary['runtime'], 'remote_directory': f'training/{name}/checkpoint_last'}


def verified_receipt(name, source=None):
    folder = OUTPUT / name
    path = folder / 'download_receipt.json'
    if not path.is_file():
        return None
    receipt = read(path)
    if receipt.get('status') != 'complete' or not REQUIRED <= set(receipt.get('files', {})):
        raise ValueError(f'{name}: malformed completed download receipt')
    if source and receipt['source']['summary_sha256'] != source['summary_sha256']:
        raise ValueError(f'{name}: existing checkpoint belongs to different source evidence')
    for filename, metadata in receipt['files'].items():
        destination = folder / filename
        if not destination.is_file() or sha(destination) != metadata['sha256']:
            raise ValueError(f'{name}: preserved file changed or disappeared: {filename}')
    if receipt['files']['model.safetensors']['sha256'] != receipt['source']['expected_model_sha256']:
        raise ValueError(f'{name}: model does not match audited training receipt')
    return receipt


def command(args, deadline, *, maximum=300):
    remaining = deadline - time.time()
    if remaining <= 0:
        raise TimeoutError('Checkpoint preservation deadline reached')
    result = subprocess.run([str(MODAL), *args], capture_output=True, text=True,
                            timeout=max(1, min(maximum, remaining)))
    if result.returncode:
        raise RuntimeError(f'Modal volume operation failed: {result.stderr or result.stdout}')
    return result.stdout


def download(source, deadline):
    name = source['name']
    existing = verified_receipt(name, source)
    if existing:
        return existing
    folder = OUTPUT / name
    folder.mkdir(parents=True, exist_ok=True)
    listing = json.loads(command(['volume', 'ls', VOLUME, '/' + source['remote_directory'], '--json'], deadline, maximum=60))
    filenames = {}
    prefix = PurePosixPath(source['remote_directory'])
    for item in listing:
        remote = PurePosixPath(item['filename'].lstrip('/'))
        if item['type'] != 'file':
            raise ValueError(f'Unexpected checkpoint subdirectory; recursive downloads prohibited: {remote}')
        if remote.parent != prefix or remote.name in filenames:
            raise ValueError(f'Unexpected/duplicate remote checkpoint filename: {remote}')
        if remote.name != 'training_state.pt':
            filenames[remote.name] = str(remote)
    if not REQUIRED <= set(filenames):
        raise ValueError(f'{name}: checkpoint files missing: {sorted(REQUIRED - set(filenames))}')
    progress_path = folder / 'download_progress.json'
    progress = read(progress_path) if progress_path.is_file() else {'source': source, 'files': {}}
    if progress['source']['summary_sha256'] != source['summary_sha256']:
        raise ValueError(f'{name}: incomplete download has different source evidence')
    # Read the model first so a bad weight artifact is detected before metadata.
    ordered = sorted(filenames, key=lambda value: (value != 'model.safetensors', value))
    for filename in ordered:
        destination = folder / filename
        prior = progress['files'].get(filename)
        if prior and destination.is_file():
            if sha(destination) != prior['sha256']:
                raise ValueError(f'{name}: completed download changed: {filename}')
            continue
        if destination.exists():
            raise ValueError(f'{name}: refusing to overwrite untracked local checkpoint file {filename}')
        temporary = folder / ('.' + filename + '.download-part')
        # Only this helper's explicitly named partial file may be replaced.
        command(['volume', 'get', VOLUME, '/' + filenames[filename], str(temporary), '--force'], deadline)
        checksum = sha(temporary)
        if filename == 'model.safetensors' and checksum != source['expected_model_sha256']:
            raise ValueError(f'{name}: downloaded model SHA256 differs from audited saved checkpoint')
        if filename.endswith('.json'):
            read(temporary)  # Reject a truncated/non-JSON processor or configuration file.
        temporary.replace(destination)
        progress['files'][filename] = {'sha256': checksum, 'bytes': destination.stat().st_size,
                                       'remote_path': filenames[filename], 'local_path': str(destination)}
        write(progress_path, progress)
    receipt = {'status': 'complete', 'source': source, 'files': progress['files'],
               'volume': VOLUME, 'checkpoint_directory': str(folder),
               'completed_at': datetime.now(timezone.utc).isoformat(),
               'optimizer_preserved_remotely': source['remote_directory'] + '/training_state.pt',
               'verification': 'Saved model.safetensors bytes match independently audited summary checkpoint_model_sha256'}
    write(folder / 'download_receipt.json', receipt)
    return receipt


def model_index(planned, completed, errors, deadline):
    return {'status': 'complete' if len(completed) == len(planned) else 'incomplete',
            'expected_runs': len(planned), 'completed_runs': len(completed),
            'missing_runs': sorted(set(planned) - set(completed)),
            'volume': VOLUME, 'deadline_utc': datetime.fromtimestamp(deadline, timezone.utc).isoformat(),
            'updated_at': datetime.now(timezone.utc).isoformat(), 'models': completed,
            'errors': errors, 'new_compute_launched': False,
            'scope': 'Durable final checkpoints from the registered development-only pilot; no publication claim'}


def build_archive(deadline):
    if time.time() >= deadline:
        return None
    archive = ROOT / 'output/trained_checkpoints.zip'
    receipt_path = ROOT / 'output/trained_checkpoints.archive.json'
    if archive.exists():
        if receipt_path.exists() and read(receipt_path)['sha256'] == sha(archive):
            return read(receipt_path)
        raise ValueError('Refusing to replace an existing unverified checkpoint archive')
    temporary = archive.with_name(archive.name + '.partial')
    paths = sorted(path for path in OUTPUT.rglob('*') if path.is_file() and '.download-part' not in path.name and not path.name.endswith('.tmp'))
    with zipfile.ZipFile(temporary, 'w', compression=zipfile.ZIP_STORED, allowZip64=True) as bundle:
        for path in paths:
            if time.time() >= deadline:
                raise TimeoutError('Archive deadline reached; verified individual checkpoints remain available')
            bundle.write(path, arcname=str(Path('checkpoints') / path.relative_to(OUTPUT)))
    temporary.replace(archive)
    receipt = {'status': 'complete', 'path': str(archive), 'sha256': sha(archive),
               'bytes': archive.stat().st_size, 'files': len(paths), 'compression': 'ZIP_STORED',
               'optimizer_included': False, 'created_at': datetime.now(timezone.utc).isoformat()}
    write(receipt_path, receipt)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--deadline-utc', default='2026-09-09T08:33:00+00:00')
    args = parser.parse_args()
    deadline = datetime.fromisoformat(args.deadline_utc).timestamp()
    schedule = read(PILOT / 'schedule.json')
    planned = {f"{r['arm_name']}-seed{r['seed']}": r for r in schedule['runs']}
    if len(planned) != 9:
        raise ValueError('Only the existing nine frozen runs may be preserved')
    OUTPUT.mkdir(parents=True, exist_ok=True)
    readme = OUTPUT / 'README.md'
    if not readme.exists():
        readme.write_text('''# Preserved trained Whisper checkpoints

Each `<arm>-seed<seed>/` directory contains the final `model.safetensors` plus
all saved tokenizer, processor, model, and generation configuration files from
the registered pilot. These are trained final weights, not the pretrained base.
`model_index.json` and each `download_receipt.json` record SHA256 hashes and the
independently audited source summary. Every model file is checked against the
worker's exact saved-checkpoint SHA256. Optimizer state remains on the existing
Modal results volume and is not included here.

Load a directory locally using the recorded runtime (Torch2.13.0,
Transformers5.16.1; see the per-model source runtime for the exact values):

```python
from transformers import WhisperForConditionalGeneration, WhisperProcessor
path = "/absolute/path/to/checkpoints/full_gold-seed0"
model = WhisperForConditionalGeneration.from_pretrained(path, local_files_only=True)
processor = WhisperProcessor.from_pretrained(path, local_files_only=True)
model.eval()
```

The saved generation configuration preserves Whisper suppression defaults. The
experiment also passed these explicit arguments on every call; supply them again
when decoding, because they are not stored as model defaults:

```python
decode_kwargs = dict(language="en", task="transcribe", return_timestamps=False,
                     max_new_tokens=224, do_sample=False, num_beams=1, use_cache=True)
# features = processor(audio_mono_16k, sampling_rate=16000, return_tensors="pt")
# token_ids = model.generate(input_features=features.input_features, **decode_kwargs)
```

Training and development manifests, full metrics, and independent evaluation code are in
the separate main evidence bundle. This directory contains development-pilot
checkpoints; it is not a claim of validated final-test generalization or
publication readiness. Whisper's underlying model/license terms still apply.
''')
    completed, errors, attempts, pending = {}, {}, {}, set()
    for name, plan in planned.items():
        source = checkpoint_source(name, plan)
        receipt = verified_receipt(name, source) if source else None
        if receipt:
            completed[name] = receipt
    futures = {}
    with ThreadPoolExecutor(max_workers=3) as executor:
        while time.time() < deadline and len(completed) < len(planned):
            for name, plan in planned.items():
                if name in completed or name in pending or attempts.get(name, 0) >= 3:
                    continue
                try:
                    source = checkpoint_source(name, plan)
                except (ValueError, KeyError) as error:
                    errors[name] = str(error)
                    attempts[name] = 3
                    continue
                if source:
                    attempts[name] = attempts.get(name, 0) + 1
                    pending.add(name)
                    futures[executor.submit(download, source, deadline)] = name
            for future in list(futures):
                if not future.done():
                    continue
                name = futures.pop(future)
                pending.remove(name)
                try:
                    completed[name] = future.result()
                    errors.pop(name, None)
                    print(json.dumps({'run': name, 'status': 'checkpoint_preserved',
                                      'model_sha256': completed[name]['files']['model.safetensors']['sha256']}), flush=True)
                except Exception as error:
                    errors[name] = str(error)
                    print(json.dumps({'run': name, 'status': 'download_failed', 'attempt': attempts[name], 'error': str(error)}), flush=True)
            write(OUTPUT / 'model_index.json', model_index(planned, completed, errors, deadline))
            if len(completed) < len(planned):
                time.sleep(min(5, max(0, deadline - time.time())))
        for future, name in futures.items():
            try:
                completed[name] = future.result()
                errors.pop(name, None)
            except Exception as error:
                errors[name] = str(error)
    final = model_index(planned, completed, errors, deadline)
    write(OUTPUT / 'model_index.json', final)
    print(json.dumps({'status': final['status'], 'completed_runs': len(completed), 'missing_runs': final['missing_runs']}), flush=True)
    if final['status'] == 'complete':
        archive = build_archive(deadline)
        print(json.dumps({'archive': archive}), flush=True)
    return 0 if final['status'] == 'complete' else 1


if __name__ == '__main__':
    raise SystemExit(main())
