"""Independently compare bundled speech/reference pairs against cached Arrow rows."""
import argparse
import hashlib
import io
import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import soundfile as sf

ROOT = Path(__file__).resolve().parents[2]


def sha(data):
    return hashlib.sha256(data).hexdigest()


def validate(bundle, expected_count=12):
    bundle = Path(bundle)
    rows = [json.loads(line) for line in (bundle/'manifest.jsonl').read_text().splitlines()]
    prior_path = ROOT/'paper/data/svarah_metadata_raw.jsonl'
    if sha(prior_path.read_bytes()) != 'ae875faf9c12fecff64c765e2a116d413e897fc63c416f60f28f39adc9793d47':
        raise ValueError('Pinned prior metadata bytes changed')
    prior = {r['utt_id']: r for r in (json.loads(s) for s in prior_path.read_text().splitlines())}
    ids = [r['conversation_id'] for r in rows]
    if expected_count not in (12, 518):
        raise ValueError('Only the frozen 12-item acceptance or complete 518-item shard is supported')
    if len(ids) != expected_count or len(set(ids)) != expected_count:
        raise ValueError(f'Expected exactly {expected_count} unique recordings')
    if expected_count == 518 and set(ids) != {f'svarah_{i:05d}' for i in range(6138, 6656)}:
        raise ValueError('Complete recovered-shard cohort is missing or replacing global IDs')
    source_shards = {}
    checks = []
    for r in rows:
        shard = r['source_shard']
        if shard not in source_shards:
            path = ROOT/'data/bridge_open/svarah_cache_v1'/shard
            # This stage is intentionally restricted to the complete recovered shard.
            if shard != 'svarah-test-00002-of-00003.arrow' or sha(path.read_bytes()) != '7f8ec2d10d827bb2329ff96164b332bb09352b62a5e9c8fa059415c6b1ffad11':
                raise ValueError('Unverified source shard')
            with pa.memory_map(str(path),'r') as source:
                source_shards[shard] = pa.ipc.open_stream(source).read_all()
            if len(source_shards[shard]) != 518:
                raise ValueError('Source shard coverage changed')
        table = source_shards[shard]
        src = table.slice(r['source_row'],1).to_pylist()[0]
        metadata = prior[r['conversation_id']]
        prefix = Path('research/bridge_open/svarah_inputs')
        audio = bundle/Path(r['audio_path']).relative_to(prefix)
        reference = bundle/Path(r['reference_path']).relative_to(prefix)
        original = src['audio_filepath']['bytes']
        if Path(src['audio_filepath']['path']).name != Path(metadata['audio_path']).name:
            raise ValueError('Source audio basename does not match claimed global ID')
        if src['text'] != metadata['text']:
            raise ValueError('Source transcript does not match pinned global metadata')
        ref_text = json.loads(reference.read_text())['reference_text']
        if ref_text != src['text']:
            raise ValueError(f"{r['conversation_id']}: bundled reference differs from exact source Arrow row")
        if sha(ref_text.encode()) != r['reference_text_sha256'] or sha(ref_text.encode()) != r['source_sentence_id']:
            raise ValueError('Reference text hash or sentence identity mismatch')
        if sha(reference.read_bytes()) != r['reference_sha256']:
            raise ValueError('Reference file hash mismatch')
        if sha(original) != r['raw_audio_sha256'] or sha(audio.read_bytes()) != r['audio_sha256']:
            raise ValueError('Original or bundled audio hash mismatch')
        if audio.read_bytes() != original or r['resampling'] != 'none':
            raise ValueError('This native 16 kHz cohort must preserve original WAV bytes')
        waveform, sr = sf.read(io.BytesIO(original),dtype='float32')
        if sr != 16000 or waveform.ndim != 1 or not np.isfinite(waveform).all() or waveform.size == 0:
            raise ValueError('Malformed source waveform')
        if not 0 < float(np.max(np.abs(waveform))) <= 1 or abs(waveform.size/sr-r['duration_seconds']) > 1e-9:
            raise ValueError('Waveform peak or duration mismatch')
        checks.append(dict(conversation_id=r['conversation_id'], source_shard=shard,
            source_row=r['source_row'], global_dataset_index=metadata['dataset_index'],
            source_audio_and_reference_match=True))
    if expected_count == 518 and {r['source_row'] for r in rows} != set(range(518)):
        raise ValueError('Complete recovered-shard cohort does not cover every source row exactly once')
    return dict(valid=True, records=expected_count, source_records=518, scope='limited recovered shard; structural not human validation',
        manifest_sha256=sha((bundle/'manifest.jsonl').read_bytes()), checks=checks)


if __name__ == '__main__':
    p=argparse.ArgumentParser();p.add_argument('bundle');p.add_argument('--output');p.add_argument('--expected-count',type=int,default=12);a=p.parse_args()
    result=validate(a.bundle,a.expected_count)
    if a.output:Path(a.output).write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
