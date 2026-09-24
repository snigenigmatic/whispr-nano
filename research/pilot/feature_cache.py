"""CPU preparation of the exact content-addressed Whisper feature cache.

Public API: prepare(rows, audio_root, cache_dir) -> receipt. Uses the frozen
training.py feature recipe without modifying training or its schedule. Test and
shared-content rows are rejected. Imports of heavy libraries happen only inside
prepare(); helper tests can run without downloading a model.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import time
import uuid

MODEL_ID = 'openai/whisper-tiny'
MODEL_REVISION = '169d4a4341b33bc18d8881c4b69c2e104e1cc0af'
TRANSFORMERS_VERSION = '5.16.1'


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def feature_signature(versions, extractor_dict):
    # Keep this literal identical to training.py's feature_signature payload.
    return canonical_hash({'revision': MODEL_REVISION, 'transformers': versions['transformers'],
                           'numpy': versions['numpy'], 'scipy': versions['scipy'], 'soundfile': versions['soundfile'],
                           'downmix': 'arithmetic mean', 'resampling': 'scipy.signal.resample_poly to 16000',
                           'feature_extractor': extractor_dict})


def validate_rows(rows):
    rows = [dict(row) for row in rows]
    if not rows:
        raise ValueError('Feature preparation requires nonempty primary train/dev rows')
    seen = set()
    for row in rows:
        uid = row.get('utt_id')
        if not isinstance(uid, str) or not uid or uid in seen:
            raise ValueError(f'Missing or duplicate feature utterance ID: {uid}')
        seen.add(uid)
        if row.get('split') not in ('train', 'dev'):
            raise ValueError(f'Test/shared-content access prohibited: {uid}')
        duration = float(row.get('duration_s', 0))
        if not math.isfinite(duration) or not 0 < duration <= 30:
            raise ValueError(f'Invalid feature duration: {uid}')
        if not isinstance(row.get('audio_path'), str) or not row['audio_path']:
            raise ValueError(f'Missing feature audio path: {uid}')
    return sorted(rows, key=lambda row: row['utt_id'])


def validate_cached(path, np, shape):
    features = np.load(path, mmap_mode='r', allow_pickle=False)
    if features.shape != shape or features.dtype != np.dtype('float32') or not np.isfinite(features).all():
        raise ValueError(f'Invalid cached feature array: {path}')
    del features


def publish_features(path, extracted, np, shape):
    """Atomic publication; existing content must match rather than be replaced."""
    extracted = extracted.astype(np.float32)
    if extracted.shape != shape or not np.isfinite(extracted).all():
        raise ValueError(f'Extractor produced invalid shape or non-finite data: {path}')
    temporary = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    try:
        with temporary.open('xb') as stream:
            np.save(stream, extracted, allow_pickle=False)
        if path.exists():
            validate_cached(path, np, shape)
            if sha256_file(path) != sha256_file(temporary):
                raise ValueError(f'Concurrent cache publication differs: {path}')
        else:
            temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _prepare(rows, audio_root, cache_dir, processor, np, sf, scipy, versions):
    start = time.monotonic()
    rows = validate_rows(rows)
    root, cache = Path(audio_root).resolve(), Path(cache_dir).resolve()
    cache.mkdir(parents=True, exist_ok=True)
    signature = feature_signature(versions, processor.feature_extractor.to_dict())
    shape = (int(processor.feature_extractor.feature_size), int(processor.feature_extractor.nb_max_frames))
    if shape != (80, 3000):
        raise ValueError(f'Unexpected frozen Whisper-tiny feature shape: {shape}')
    records, created, reused = [], 0, 0
    for row in rows:
        uid = row['utt_id']
        path = (root / row['audio_path']).resolve()
        if not path.is_relative_to(root):
            raise ValueError(f'Audio path escapes audio_root: {uid}')
        checksum = sha256_file(path)
        if row.get('audio_sha256') and row['audio_sha256'] != checksum:
            raise ValueError(f'Audio checksum mismatch: {uid}')
        audio, sample_rate = sf.read(path, dtype='float32', always_2d=True)
        duration = len(audio) / sample_rate
        if not len(audio) or duration > 30.0 or not np.isfinite(audio).all():
            raise ValueError(f'Invalid or >30s audio (no truncation allowed): {uid}')
        if abs(duration - row['duration_s']) > max(0.01, 1 / sample_rate):
            raise ValueError(f'Manifest/audio duration mismatch: {uid}')
        feature_path = cache / (canonical_hash([checksum, signature]) + '.npy')
        if feature_path.exists():
            validate_cached(feature_path, np, shape)
            reused += 1
        else:
            audio = audio.mean(axis=1)
            if sample_rate != 16000:
                divisor = math.gcd(sample_rate, 16000)
                audio = scipy.signal.resample_poly(audio, 16000 // divisor, sample_rate // divisor).astype(np.float32)
            extracted = processor.feature_extractor(audio, sampling_rate=16000, return_tensors='np').input_features[0]
            publish_features(feature_path, extracted, np, shape)
            created += 1
        records.append({'utt_id': uid, 'split': row['split'], 'audio_sha256': checksum,
                        'duration_s': duration, 'original_sample_rate': sample_rate,
                        'feature_file': feature_path.name, 'feature_sha256': sha256_file(feature_path),
                        'feature_bytes': feature_path.stat().st_size})
    return {'status': 'completed', 'purpose': 'cpu_feature_preparation_no_inference',
            'rows': len(rows), 'utt_ids_sha256': canonical_hash([row['utt_id'] for row in rows]),
            'manifest_sha256': canonical_hash(rows), 'feature_signature': signature,
            'model_id': MODEL_ID, 'model_revision': MODEL_REVISION, 'runtime': versions,
            'feature_shape': list(shape), 'feature_dtype': 'float32', 'torch_threads': 2,
            'created_features': created, 'reused_features': reused,
            'cache_dir': str(cache), 'files': records, 'elapsed_s': time.monotonic() - start,
            'preparation_code_sha256': sha256_file(Path(__file__))}


def prepare(rows, audio_root, cache_dir):
    rows = validate_rows(rows)  # Reject held-out access before loading libraries/models.
    import numpy as np
    import scipy
    import scipy.signal
    import soundfile as sf
    import torch
    import transformers
    from transformers import WhisperProcessor
    if transformers.__version__ != TRANSFORMERS_VERSION:
        raise ValueError(f'Feature preparation requires transformers=={TRANSFORMERS_VERSION}')
    torch.set_num_threads(2)
    processor = WhisperProcessor.from_pretrained(MODEL_ID, revision=MODEL_REVISION,
                                                language='english', task='transcribe')
    processor.tokenizer.set_prefix_tokens(language='english', task='transcribe', predict_timestamps=False)
    versions = {'torch': torch.__version__, 'transformers': transformers.__version__, 'numpy': np.__version__,
                'scipy': scipy.__version__, 'soundfile': sf.__version__}
    return _prepare(rows, audio_root, cache_dir, processor, np, sf, scipy, versions)
