"""Acquire pinned public Humyn data; no inference and no SPIRE access."""
from __future__ import annotations

import concurrent.futures
import hashlib
import io
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import soundfile as sf
from huggingface_hub import hf_hub_download
from scipy.signal import resample_poly

REPO = 'humyn-labs/Indic-High-Fidelity-MultiSpeaker-ASR'
REVISION = '2e4879d569bd57f675c227a41fecf4a2e56f467c'
LANGUAGES = dict(Assamese='as', Bengali='bn', Bhojpuri='bho', Chhattisgarhi='hne',
                 Gujarati='gu', Haryanvi='bgc', Hindi='hi', Kannada='kn',
                 Malayalam='ml', Marathi='mr', Punjabi='pa', Tamil='ta', Telugu='te')
# Released label contains this spelling; retain released_language in every row.
LANGUAGES['Chattisgarhi'] = 'hne'
ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / 'data/bridge_open/humyn_public_v1'
ARTIFACTS = ROOT / 'research/bridge_open/artifacts'


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def timestamp(value) -> float:
    if isinstance(value, str) and ':' in value:
        parts = value.split(':')
        if len(parts) != 3:
            raise ValueError('Expected HH:MM:SS timestamp')
        hours, minutes, seconds = map(float, parts)
        if not (hours >= 0 and 0 <= minutes < 60 and 0 <= seconds < 60):
            raise ValueError('Invalid timestamp fields')
        result = hours * 3600 + minutes * 60 + seconds
    else:
        result = float(value)
    if not math.isfinite(result) or result < 0:
        raise ValueError('Invalid timestamp')
    return result


def prepare_shard(index: int):
    filename = f'data/train-{index:05d}-of-00022.parquet'
    path = hf_hub_download(REPO, filename, repo_type='dataset', revision=REVISION, token=False)
    shard_sha = digest(Path(path).read_bytes())
    rows = []
    for row_index, row in enumerate(pq.read_table(path).to_pylist()):
        cid = f'humyn-{index:02d}-{row_index:02d}'
        encoded = row['audio']['bytes']
        samples, sr = sf.read(io.BytesIO(encoded), dtype='float32', always_2d=True)
        if not len(samples) or not np.isfinite(samples).all():
            raise ValueError(f'{cid}: invalid decoded audio')
        duration = len(samples) / sr
        channels = samples.shape[1]
        mono = samples.mean(axis=1)
        if sr != 16000:
            divisor = math.gcd(sr, 16000)
            mono = resample_poly(mono, 16000 // divisor, sr // divisor).astype('float32')
        wav = DATA / f'{cid}.wav'
        sf.write(wav, mono, 16000, subtype='FLOAT')
        segments = json.loads(row['transcript_json'])
        issues = []
        if not isinstance(segments, list) or not segments:
            raise ValueError(f'{cid}: empty/non-array reference')
        parsed = []
        for j, segment in enumerate(segments):
            start, end = timestamp(segment['start']), timestamp(segment['end'])
            if end < start or end > duration + 0.25:
                issues.append(f'segment_{j}_timestamp_out_of_bounds')
            if not isinstance(segment['text'], str):
                raise ValueError(f'{cid}: non-string reference')
            parsed.append({**segment, 'start_seconds': start, 'end_seconds': end})
        # Stable chronological concatenation; overlapping speech remains in that order.
        ordered = sorted(parsed, key=lambda s: s['start_seconds'])
        reference = ' '.join(s['text'].strip() for s in ordered if s['text'].strip())
        if not reference:
            issues.append('empty_reference')
        refpath = DATA / f'{cid}.reference.json'
        refpath.write_text(json.dumps({'reference_text': reference, 'segments': parsed,
                                      'raw_transcript_json': row['transcript_json']}, ensure_ascii=False) + '\n')
        rows.append(dict(conversation_id=cid, source_repo=REPO, source_revision=REVISION,
                         split='train', usage='public_evaluation_pilot', source_file=row['file_name'],
                         source_shard=filename, shard_sha256=shard_sha,
                         source_audio_sha256=digest(encoded), audio_path=str(wav),
                         audio_sha256=digest(wav.read_bytes()), reference_path=str(refpath),
                         reference_sha256=digest(refpath.read_bytes()),
                         released_language=row['language'], language=LANGUAGES[row['language']],
                         duration_seconds=duration, source_sample_rate=sr, source_channels=channels,
                         sample_rate=16000, channel_policy='arithmetic_mean',
                         resampling='scipy.signal.resample_poly', license='CC-BY-4.0',
                         segment_count=len(parsed), reference_qc='pass' if not issues else 'reference_suspect',
                         qc_issues=issues, speaker_identity='file-local labels only'))
    print(json.dumps({'event': 'acquired_shard', 'shard': index, 'rows': len(rows)}), flush=True)
    return rows


def main():
    DATA.mkdir(parents=True, exist_ok=True)
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    manifest = DATA / 'corpus-v1.jsonl'
    if manifest.exists():
        raise SystemExit('Frozen manifest exists; refuse overwrite. Validate/reuse it or create a new version.')
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        rows = [r for group in pool.map(prepare_shard, range(22)) for r in group]
    if len(rows) != 37 or len({r['conversation_id'] for r in rows}) != 37:
        raise ValueError('Unexpected count or duplicate IDs')
    manifest.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows))
    receipt = dict(source_repo=REPO, source_revision=REVISION, manifest_path=str(manifest),
                   manifest_sha256=digest(manifest.read_bytes()), conversations=len(rows),
                   language_counts=dict(Counter(r['language'] for r in rows)),
                   total_duration_seconds=sum(r['duration_seconds'] for r in rows),
                   min_duration_seconds=min(r['duration_seconds'] for r in rows),
                   max_duration_seconds=max(r['duration_seconds'] for r in rows),
                   qc_counts=dict(Counter(r['reference_qc'] for r in rows)),
                   reference_suspect_ids=[r['conversation_id'] for r in rows if r['reference_qc']!='pass'],
                   bridge_membership='unverified', cloud_cost_usd=0,
                   acquisition_code_sha256=digest(Path(__file__).read_bytes()))
    (ARTIFACTS / 'corpus_receipt_v1.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(receipt, indent=2), flush=True)


if __name__ == '__main__':
    main()
