"""Bounded Modal data preparation and controlled ASR pilot execution.

All paid calls are reserved locally before dispatch. No final-test audio is
downloaded or scored. A separate output volume protects historical artifacts.
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import modal


ROOT = Path(__file__).resolve().parents[2] if modal.is_local() else Path('/root')
app = modal.App('accent-selection-pilot')
data_volume = modal.Volume.from_name('accent-selection-pilot-data', create_if_missing=True)
out_volume = modal.Volume.from_name('accent-selection-pilot-results', create_if_missing=True)
cache_volume = modal.Volume.from_name('ast-asr-cache')
CPU_RATE = 2 * 0.0000131 + 8 * 0.00000222
T4_RATE = 0.000164 + 2 * 0.0000131 + 16 * 0.00000222
A10_RATE = 0.000306 + 2 * 0.0000131 + 16 * 0.00000222
cpu_image = modal.Image.debian_slim(python_version='3.12')
gpu_image = (
    modal.Image.debian_slim(python_version='3.12').apt_install('ffmpeg')
    .pip_install('torch==2.13.0', 'transformers==5.16.1', 'accelerate==1.11.0',
                 'datasets==4.4.1', 'torchcodec==0.16.0', 'soundfile==0.14.0',
                 'librosa==1.0.0', 'numpy==2.5.2')
    .pip_install('jiwer==4.0.0')
    .env({'HF_HOME': '/cache/huggingface', 'HF_HUB_ENABLE_HF_TRANSFER': '0'})
)
training_image = (gpu_image.pip_install('pytest==9.1.1')
                  .add_local_file(ROOT / 'research/pilot/training.py', '/root/research/pilot/training.py')
                  .add_local_file(ROOT / 'research/pilot/test_training.py', '/root/research/pilot/test_training.py')
                  .add_local_file(ROOT / 'research/pilot/frozen_english_normalizer.py', '/root/research/pilot/frozen_english_normalizer.py')
                  .add_local_file(ROOT / 'research/pilot/feature_cache.py', '/root/research/pilot/feature_cache.py')
                  .add_local_file(ROOT / 'paper/data/whisper_normalizer.json', '/root/normalizer.json'))


def reserve(run_id, configs, maximum_per_job):
    path = ROOT / 'research/pilot/budget.json'
    ledger = json.loads(path.read_text())
    current = sum(j['reserved_usd'] for j in ledger['new_jobs'])
    if ledger['prior_work_carryover_allowance_usd'] + current + maximum_per_job * len(configs) > ledger['authorized_total_usd']:
        raise RuntimeError('Pilot reservations would exceed total authorized spending')
    existing = {j['job_id'] for j in ledger['new_jobs']}
    if any(f'{run_id}/{i}' in existing for i in range(len(configs))):
        raise ValueError('A run ID cannot be reused; retries require a new reservation')
    for i, config in enumerate(configs):
        config['job_id'] = f'{run_id}/{i}'
        ledger['new_jobs'].append({'job_id': config['job_id'], 'reserved_usd': maximum_per_job,
                                  'status': 'reserved', 'config': config})
    path.write_text(json.dumps(ledger, indent=2) + '\n')


def record_result(result):
    path = ROOT / 'research/pilot/budget.json'
    ledger = json.loads(path.read_text())
    job = next(j for j in ledger['new_jobs'] if j['job_id'] == result['job_id'])
    job.update({'status': 'completed', 'wall_seconds': result['wall_seconds'],
                'estimated_resource_cost_usd': result['estimated_resource_cost_usd']})
    path.write_text(json.dumps(ledger, indent=2) + '\n')


@app.function(image=cpu_image, cpu=(2, 2), memory=(8192, 8192),
              timeout=1800, startup_timeout=300, retries=0, max_containers=6,
              scaledown_window=2, volumes={'/data': data_volume})
def prepare_speaker(config):
    import urllib.request
    import zipfile
    import wave
    start = time.monotonic()
    if config['split'] not in ('train', 'dev'):
        raise ValueError('Final-test audio must remain untouched')
    speaker = config['speaker_id']
    root = Path('/data/l2arctic')
    root.mkdir(parents=True, exist_ok=True)
    marker = root / f'{speaker}.verified.json'
    if marker.exists():
        prior = json.loads(marker.read_text())
        if prior['archive_sha256'] != config['archive_sha256']:
            raise ValueError('Existing speaker audio has different provenance')
        result = {**prior, 'job_id': config['job_id'], 'cache_hit': True}
    else:
        archive = root / f'{speaker}.zip.partial'
        digest, size = hashlib.sha256(), 0
        req = urllib.request.Request(config['archive_url'], headers={'User-Agent': 'whispr-nano-research/1.0'})
        with urllib.request.urlopen(req, timeout=120) as response, archive.open('wb') as target:
            while chunk := response.read(4 * 1024 * 1024):
                target.write(chunk)
                digest.update(chunk)
                size += len(chunk)
        if size != config['archive_bytes'] or digest.hexdigest() != config['archive_sha256']:
            raise ValueError(f'Archive provenance mismatch for {speaker}')
        with zipfile.ZipFile(archive) as z:
            for item in z.infolist():
                destination = (root / item.filename).resolve()
                if not destination.is_relative_to(root.resolve()):
                    raise ValueError('Unsafe ZIP member path')
                if not item.is_dir() and (item.filename.startswith(speaker + '/wav/') or item.filename.startswith(speaker + '/transcript/')):
                    z.extract(item, root)
        durations = {}
        for wav in sorted((root / speaker / 'wav').glob('*.wav')):
            with wave.open(str(wav)) as w:
                durations[wav.stem] = {'duration_s': w.getnframes() / w.getframerate(),
                                       'sample_rate': w.getframerate(), 'channels': w.getnchannels()}
        inventory_sha = hashlib.sha256('\n'.join(sorted(durations)).encode()).hexdigest()
        if len(durations) != config['expected_wav_count'] or inventory_sha != config['expected_stems_sha256']:
            raise ValueError(f'Unexpectedly incomplete audio archive: {speaker}, {len(durations)}')
        result = {'speaker_id': speaker, 'native_language': config['native_language'],
                  'split': config['split'], 'archive_sha256': config['archive_sha256'],
                  'archive_bytes': size, 'n_wav': len(durations), 'audio_metadata': durations,
                  'job_id': config['job_id'], 'cache_hit': False}
        marker.write_text(json.dumps(result, indent=2) + '\n')
        archive.unlink()
        data_volume.commit()
    result['wall_seconds'] = time.monotonic() - start
    result['estimated_resource_cost_usd'] = result['wall_seconds'] * CPU_RATE
    print(f"Prepared {speaker}: {result['n_wav']} clips in {result['wall_seconds']:.1f}s", flush=True)
    return result


@app.local_entrypoint()
def prepare(run_id: str = 'data-1'):
    speakers = json.loads((ROOT / 'research/pilot/data/speakers.json').read_text())
    metadata = [json.loads(x) for x in (ROOT / 'research/pilot/data/corpus_metadata.jsonl').read_text().splitlines()]
    output = ROOT / 'research/pilot/data/audio_validation'
    configs = [dict(s) for s in speakers if s['split'] in ('train', 'dev')
               and not (output / f"{s['speaker_id']}.json").exists()]
    for config in configs:
        stems = sorted(r['prompt_id'] for r in metadata if r['speaker_id'] == config['speaker_id'])
        config['expected_wav_count'] = len(stems)
        config['expected_stems_sha256'] = hashlib.sha256('\n'.join(stems).encode()).hexdigest()
        config['purpose'] = 'prepare_audio'
    reserve(run_id, configs, 0.12)
    output.mkdir(parents=True, exist_ok=True)
    failed = []
    for index, result in enumerate(prepare_speaker.map(configs, order_outputs=True, return_exceptions=True)):
        config = configs[index]
        if isinstance(result, Exception):
            failed.append({'speaker_id': config['speaker_id'], 'error': str(result)})
            continue
        record_result(result)
        (output / f"{result['speaker_id']}.json").write_text(json.dumps(result, indent=2) + '\n')
    if failed:
        print('Preparation failures (completed speakers retained):', failed)
        raise RuntimeError('Some audio archives still need preparation')
    print('Training/development audio prepared and hash-verified; no final-test audio downloaded.')


@app.function(image=gpu_image, gpu='A10G', cpu=(2, 2), memory=(16384, 16384),
              timeout=1200, startup_timeout=300, retries=0, max_containers=6,
              scaledown_window=2,
              volumes={'/data': data_volume, '/outputs': out_volume, '/cache/huggingface': cache_volume})
def teacher_speaker(config, rows):
    import numpy as np
    import soundfile as sf
    import torch
    import librosa
    import importlib.metadata
    from transformers import WhisperProcessor, WhisperForConditionalGeneration
    from transformers.utils import logging as hf_logging
    hf_logging.set_verbosity_error()
    start = time.monotonic()
    if not rows or any(r['split'] != 'train' or r['speaker_id'] != config['speaker_id'] for r in rows):
        raise ValueError('Teacher selection may only inspect the assigned training speaker')
    data_volume.reload()
    root = Path('/data/l2arctic')
    marker = json.loads((root / f"{config['speaker_id']}.verified.json").read_text())
    if marker['archive_sha256'] != config['archive_sha256']:
        raise ValueError('Teacher audio provenance mismatch')
    model_id = 'openai/whisper-large-v3'
    revision = '06f233fe06e710322aca913c1bc4249a0d71fce1'
    processor = WhisperProcessor.from_pretrained(model_id, revision=revision)
    model = WhisperForConditionalGeneration.from_pretrained(model_id, revision=revision, dtype=torch.float32).to('cuda').eval()
    torch.manual_seed(0)
    torch.set_num_threads(2)
    folder = Path('/outputs/teacher') / config['speaker_id']
    folder.mkdir(parents=True, exist_ok=True)
    predictions = []
    for begin in range(0, len(rows), 4):
        if time.monotonic() - start > 1120:
            raise TimeoutError('Teacher stopped before hard timeout; partial predictions preserved')
        batch = rows[begin:begin+4]
        arrays = []
        for r in batch:
            path = root / r['audio_path']
            audio, sr = sf.read(path, dtype='float32')
            if audio.ndim != 1 or len(audio) / sr > 30:
                raise ValueError('Audio outside frozen mono/short-utterance scope')
            if abs(len(audio) / sr - float(r['duration_s'])) > 0.01:
                raise ValueError('Manifest duration differs from decoded waveform')
            if sr != 16000:
                audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
            arrays.append(audio)
        feats = processor(arrays, sampling_rate=16000, return_tensors='pt',
                          padding='max_length', truncation=True, return_attention_mask=True)
        with torch.inference_mode():
            seqs = model.generate(input_features=feats.input_features.to('cuda'),
                                   attention_mask=feats.attention_mask.to('cuda'),
                                   language='en', task='transcribe', return_timestamps=False,
                                   do_sample=False, num_beams=1, max_new_tokens=224)
        hypotheses = processor.batch_decode(seqs, skip_special_tokens=True)
        if len(hypotheses) != len(batch):
            raise ValueError('Teacher output count mismatch')
        predictions.extend({**r, 'hypothesis_text': h.strip()} for r, h in zip(batch, hypotheses))
        if begin % 100 == 0 or len(predictions) == len(rows):
            (folder / 'predictions.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in predictions))
            out_volume.commit()
            print(f"Teacher {config['speaker_id']}: {len(predictions)}/{len(rows)} in {time.monotonic()-start:.1f}s", flush=True)
    result = {'job_id': config['job_id'], 'speaker_id': config['speaker_id'],
              'model_id': model_id, 'model_revision': revision, 'completed_rows': len(rows),
              'manifest_sha256': config['manifest_sha256'], 'archive_sha256': config['archive_sha256'],
              'generation_config': model.generation_config.to_dict(),
              'generation_overrides': {'language': 'en', 'task': 'transcribe', 'return_timestamps': False,
                                       'do_sample': False, 'num_beams': 1, 'max_new_tokens': 224},
              'versions': {p: importlib.metadata.version(p) for p in ['torch', 'transformers', 'librosa', 'soundfile']},
              'gpu_name': torch.cuda.get_device_name(), 'wall_seconds': time.monotonic() - start}
    result['estimated_resource_cost_usd'] = result['wall_seconds'] * A10_RATE
    (folder / 'runtime.json').write_text(json.dumps(result, indent=2) + '\n')
    out_volume.commit()
    return {**result, 'predictions': predictions}


@app.local_entrypoint()
def teacher(run_id: str = 'teacher-1'):
    path = ROOT / 'research/pilot/data/manifest.jsonl'
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    speakers = json.loads((ROOT / 'research/pilot/data/speakers.json').read_text())
    ledger = json.loads((ROOT / 'research/pilot/budget.json').read_text())
    launched = {j['config'].get('speaker_id') for j in ledger['new_jobs'] if j['config'].get('purpose') == 'teacher'}
    configs = [{**s, 'purpose': 'teacher', 'manifest_sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
               for s in speakers if s['split'] == 'train' and s['speaker_id'] not in launched
               and (ROOT / 'research/pilot/data/audio_validation' / f"{s['speaker_id']}.json").exists()]
    if not configs:
        print('No new verified training speakers ready.')
        return
    reserve(run_id, configs, 0.60)
    jobs = [(c, [r for r in rows if r['speaker_id'] == c['speaker_id'] and r['split'] == 'train']) for c in configs]
    output = ROOT / 'research/pilot/teacher_predictions'
    output.mkdir(parents=True, exist_ok=True)
    failed = []
    for result in teacher_speaker.starmap(jobs, order_outputs=False, return_exceptions=True):
        if isinstance(result, Exception):
            failed.append(str(result))
            continue
        predictions = result.pop('predictions')
        record_result(result)
        (output / f"{result['speaker_id']}.jsonl").write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in predictions))
        (output / f"{result['speaker_id']}.runtime.json").write_text(json.dumps(result, indent=2) + '\n')
    if failed:
        raise RuntimeError(f'Teacher failures, completed outputs retained: {failed}')


@app.function(image=training_image, gpu='T4', cpu=(2, 2), memory=(16384, 16384),
              timeout=3600, startup_timeout=300, retries=0, max_containers=3,
              scaledown_window=2,
              volumes={'/data': data_volume, '/outputs': out_volume, '/cache/huggingface': cache_volume})
def train_model(config, rows):
    import importlib.util
    import sys
    start = time.monotonic()
    spec = importlib.util.spec_from_file_location('pilot_training', '/root/research/pilot/training.py')
    module = importlib.util.module_from_spec(spec)
    sys.modules['pilot_training'] = module
    spec.loader.exec_module(module)
    import pytest
    if config.get('purpose') == 'training_smoke':
        if int(pytest.main(['/root/research/pilot/test_training.py', '-q', '-s'])) != 0:
            raise RuntimeError('Training preflight unit checks failed')
    data_volume.reload()
    train_config = dict(config['training'])
    train_config['normalizer_path'] = '/root/normalizer.json'
    # Replicate seeds run sequentially within each arm, so verified feature caches can be reused.
    folder = Path('/outputs/training') / config['run_name']
    train_config['feature_cache_dir'] = str(Path('/data/feature_cache') / config.get('cache_namespace', config['run_name']))
    import threading
    stop_progress = threading.Event()
    def report_progress():
        while not stop_progress.wait(30):
            history = folder / 'history.jsonl'
            updates = 0
            if history.exists():
                for line in history.read_text().splitlines():
                    try:
                        event = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if event.get('event') == 'train_update':
                        updates = max(updates, event['update'])
            print(f"Training {config['run_name']}: {updates}/{train_config['num_updates']} updates; elapsed {time.monotonic()-start:.0f}s", flush=True)
    progress = threading.Thread(target=report_progress, daemon=True)
    progress.start()
    try:
        summary = module.train(train_config, rows, '/data/l2arctic', folder)
    finally:
        stop_progress.set()
        progress.join(timeout=2)

    summary['job_id'] = config['job_id']
    summary['wall_seconds'] = time.monotonic() - start
    summary['estimated_resource_cost_usd'] = summary['wall_seconds'] * T4_RATE
    summary['checkpoint_model_sha256'] = module.sha256_file(folder / 'checkpoint_last/model.safetensors')
    (folder / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    out_volume.commit()
    data_volume.commit()
    predictions = {key: (folder / filename).read_text() for key, filename in summary['prediction_files'].items()}
    # Strip library-specific scalar subclasses (e.g. TorchVersion) at the RPC boundary.
    files = {name: (folder / name).read_text() for name in [
        'provenance.json', 'manifest.json', 'history.jsonl',
        'baseline_dev_metrics.json', 'final_dev_metrics.json']}
    return json.loads(json.dumps({'summary': summary, 'predictions': predictions, 'files': files}))


@app.local_entrypoint()
def train_smoke(run_id: str = 'train-smoke-1'):
    rows = [json.loads(line) for line in (ROOT / 'research/pilot/data/manifest.jsonl').read_text().splitlines()]
    ready = {p.stem for p in (ROOT / 'research/pilot/data/audio_validation').glob('*.json')}
    candidates = [r for r in rows if r['speaker_id'] in ready]
    small = [r for r in candidates if r['split'] == 'train'][:16] + [r for r in candidates if r['split'] == 'dev'][:12]
    if not any(r['split'] == 'dev' for r in small):
        raise ValueError('No verified development speaker audio yet')
    configs = [{'run_name': run_id, 'purpose': 'training_smoke',
                'training': {'arm_name': 'smoke_not_scientific_result', 'num_updates': 3,
                             'batch_size': 8, 'seed': 0, 'eval_every': 0}}]
    reserve(run_id, configs, 0.65)
    result = train_model.remote(configs[0], small)
    summary = result['summary']
    record_result(summary)
    out = ROOT / 'research/pilot/runs' / run_id
    out.mkdir(parents=True, exist_ok=True)
    (out / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    for key, contents in result['predictions'].items():
        (out / f'{key}_dev_predictions.jsonl').write_text(contents)
    print('Training smoke passed; not a scientific result.')


@app.local_entrypoint()
def train_registered(seed: int = 0):
    """Execute one predeclared three-arm seed wave; every dev query is precharged."""
    from research.pilot import workflow
    from research.pilot.selection import load_jsonl
    deadline = json.loads((ROOT / 'research/pilot/deadline.json').read_text())
    if time.time() + 600 > deadline['experiment_cutoff_epoch']:
        raise RuntimeError('Less than ten minutes remain before the hard compute cutoff; preserve current results.')
    workflow.initialize()
    schedule = json.loads((ROOT / 'research/pilot/schedule.json').read_text())
    runs = [r for r in schedule['runs'] if r['seed'] == seed]
    if len(runs) != 3:
        raise ValueError('Seed wave is not the predeclared three-arm comparison')
    configs = [{'run_name': workflow.run_name(r['arm_name'], seed), 'purpose': 'registered_training',
                'cache_namespace': 'frozen-v1', 'training': workflow.training_config(r['arm_name'], seed)} for r in runs]
    reserve(f'train-seed{seed}', configs, 0.95)
    dev = [r for r in load_jsonl(ROOT / 'research/pilot/data/manifest.jsonl') if r['split'] == 'dev']
    jobs = []
    for config in configs:
        train = load_jsonl(ROOT / 'research/pilot/selections' / f"{config['training']['arm_name']}.jsonl")
        workflow.before_dispatch(config)
        jobs.append((config, train + dev))
    failures = []
    for index, result in enumerate(train_model.starmap(jobs, order_outputs=True, return_exceptions=True)):
        config = configs[index]
        if isinstance(result, Exception):
            failures.append(str(workflow.record_failure(config, result)))
            continue
        try:
            summary = result['summary']
            record_result(summary)
            output = ROOT / 'research/pilot/runs' / config['run_name']
            output.mkdir(parents=True, exist_ok=True)
            if (output / 'summary.json').exists():
                raise ValueError('Refusing to overwrite a completed local result')
            (output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
            for label, contents in result['predictions'].items():
                (output / f'{label}_dev_predictions.jsonl').write_text(contents)
            for filename, contents in result['files'].items():
                (output / filename).write_text(contents)
            workflow.after_result(config, output)
            print(f"Completed registered run: {config['run_name']}", flush=True)
        except Exception as error:
            failures.append(str(workflow.record_failure(config, error)))
    if failures:
        raise RuntimeError(f'Infrastructure failure receipts retained: {failures}')
    print(f'All three registered arms completed for seed {seed}.')


@app.function(image=training_image, cpu=(2, 2), memory=(8192, 8192),
              timeout=1800, startup_timeout=300, retries=0, max_containers=18,
              scaledown_window=2,
              volumes={'/data': data_volume, '/cache/huggingface': cache_volume})
def prepare_features(config, rows):
    from research.pilot.feature_cache import prepare
    start = time.monotonic()
    data_volume.reload()
    result = prepare(rows, '/data/l2arctic', '/data/feature_cache/frozen-v1')
    result.update(job_id=config['job_id'], speaker_id=config['speaker_id'])
    result['wall_seconds'] = time.monotonic() - start
    result['estimated_resource_cost_usd'] = result['wall_seconds'] * CPU_RATE
    data_volume.commit()
    print(f"Cached {result['speaker_id']}: {result['rows']} feature arrays", flush=True)
    return json.loads(json.dumps(result))


@app.local_entrypoint()
def features(run_id: str = 'features-1'):
    rows = [json.loads(line) for line in (ROOT / 'research/pilot/data/manifest.jsonl').read_text().splitlines()]
    rows = [r for r in rows if r['split'] in ('train', 'dev')]
    configs = [{'speaker_id': s, 'purpose': 'cpu_feature_cache'} for s in sorted({r['speaker_id'] for r in rows})]
    reserve(run_id, configs, 0.10)
    output = ROOT / 'research/pilot/feature_receipts'
    output.mkdir(parents=True, exist_ok=True)
    jobs = [(config, [r for r in rows if r['speaker_id'] == config['speaker_id']]) for config in configs]
    results, failures = [], []
    for result in prepare_features.starmap(jobs, order_outputs=False, return_exceptions=True):
        if isinstance(result, Exception):
            failures.append(str(result))
            continue
        record_result(result)
        (output / f"{result['speaker_id']}.json").write_text(json.dumps(result, indent=2) + '\n')
        results.append(result)
    if failures:
        raise RuntimeError(f'Feature preparation failures: {failures}')
    signatures = {r['feature_signature'] for r in results}
    ids = [f['utt_id'] for r in results for f in r['files']]
    if len(signatures) != 1 or len(ids) != len(set(ids)) or set(ids) != {r['utt_id'] for r in rows}:
        raise ValueError('Feature cache coverage or signature mismatch')
    receipt = {'status': 'completed', 'rows': len(ids), 'speakers': len(results),
               'feature_signature': next(iter(signatures)), 'test_access': False,
               'speaker_receipt_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(output.glob('*.json'))}}
    (ROOT / 'research/pilot/feature_cache_receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print('All frozen training/development features prepared; no inference performed.')
