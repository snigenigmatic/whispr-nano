"""Prepare the complete three-language FLEURS official test after protocol freeze."""
from __future__ import annotations

import concurrent.futures
import csv
import hashlib
import io
import json
from pathlib import Path
import tarfile

import numpy as np
import requests
import soundfile as sf
from huggingface_hub import hf_hub_download

ROOT = Path(__file__).resolve().parents[2]
REPO = 'google/fleurs'
REVISION = '70bb2e84b976b7e960aa89f1c648e09c59f894dd'
CONFIGS = ('hi_in', 'bn_in', 'as_in')
COUNT = None
DATA = ROOT / 'data/bridge_open/fleurs_test_v1'
OUT = ROOT / 'research/bridge_open/artifacts/fleurs_test_v1'


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def prepare(config):
    source = f'data/{config}/audio/test.tar.gz'
    meta = requests.get(f'https://huggingface.co/api/datasets/{REPO}/tree/{REVISION}/data/{config}/audio', timeout=30)
    meta.raise_for_status()
    archive_meta = next(x for x in meta.json() if x['path'] == source)
    assert archive_meta['size'] < 2_000_000_000
    tsv = Path(hf_hub_download(REPO, f'data/{config}/test.tsv', repo_type='dataset', revision=REVISION, token=False))
    # Quote characters belong to the transcript; the release TSV does not use CSV quoting.
    records = list(csv.reader(io.StringIO(tsv.read_text()), delimiter='\t', quoting=csv.QUOTE_NONE))
    assert all(len(r) == 7 for r in records)
    # Selection uses immutable file identity, never reference content or model scores.
    selected = sorted(records, key=lambda r: hashlib.sha256(f'12:{config}:{r[1]}'.encode()).hexdigest())[:COUNT]
    names = {r[1]: r for r in selected}
    assert len(names) == len(records)
    print(json.dumps(dict(event='downloading_test_archive', config=config, bytes=archive_meta['size'])), flush=True)
    archive = Path(hf_hub_download(REPO, source, repo_type='dataset', revision=REVISION, token=False))
    archive_sha = digest(archive)
    assert archive.stat().st_size == archive_meta['size']
    assert archive_sha == archive_meta['lfs']['oid']
    rows = []
    found = set()
    with tarfile.open(archive, 'r:gz') as tar:
        for member in tar:
            name = Path(member.name).name
            if name not in names or not member.isfile():
                continue
            assert name not in found
            found.add(name)
            encoded = tar.extractfile(member).read()
            samples, sr = sf.read(io.BytesIO(encoded), dtype='float32')
            raw = names[name]
            assert samples.ndim == 1 and sr == 16000 and len(samples) == int(raw[5])
            assert np.isfinite(samples).all() and len(samples) > 0
            assert raw[2].strip() and raw[3].strip()
            cid = f'fleurs-{config}-test-{Path(name).stem}'
            audio = DATA / f'{cid}.wav'
            audio.write_bytes(encoded)  # exact original bytes; no normalization/downmix
            reference = DATA / f'{cid}.reference.json'
            reference.write_text(json.dumps(dict(reference_text=raw[3], raw_transcription=raw[2],
                source_sentence_id=raw[0], source_filename=name), ensure_ascii=False) + '\n')
            peak = float(np.abs(samples).max())
            zero_fraction = float(np.mean(samples == 0))
            issues = []
            if peak == 0:
                issues.append('all_zero_audio')
            if zero_fraction > 0.8:
                issues.append('over_80_percent_exact_zero_samples')
            if len(samples) / sr > 30:
                issues.append('over_30_seconds')
            rows.append(dict(conversation_id=cid, recording_unit='utterance', source_repo=REPO,
                source_revision=REVISION, source_config=config, split='test',
                source_split_name='test', usage='frozen_official_test_v1', license='CC-BY-4.0',
                source_shard=source, source_archive_sha256=archive_sha, source_member=member.name,
                source_metadata_sha256=digest(tsv), source_sentence_id=raw[0],
                audio_path=str(audio), audio_sha256=digest(audio), reference_path=str(reference),
                reference_sha256=digest(reference), language=config.split('_')[0],
                released_language=config, duration_seconds=len(samples)/sr, sample_rate=sr,
                source_sample_rate=sr, source_channels=1, channel_policy='original_mono',
                resampling='none', released_gender=raw[6], speaker_identity=None,
                reference_qc='pass' if not issues else 'reference_suspect', qc_issues=issues,
                qc_scope='Automated integrity, waveform and metadata checks only; not human verification.',
                peak=peak, rms=float(np.sqrt(np.mean(samples.astype('float64')**2))),
                exact_zero_fraction=zero_fraction, clipped_fraction=float(np.mean(np.abs(samples) >= 0.999))))
    assert found == set(names)
    return dict(config=config, test_split_total_rows=len(records), source_archive_sha256=archive_sha,
        source_archive_bytes=archive_meta['size'], selected_rows=sorted(rows, key=lambda r:r['conversation_id']))


def main():
    manifest = DATA / 'corpus-v1.jsonl'
    protocol = ROOT / 'research/bridge_open/FLEURS_TEST_PROTOCOL_V1.md'
    assert protocol.is_file(), 'Freeze protocol before acquisition'
    if manifest.exists():
        raise SystemExit('Frozen FLEURS manifest exists; refusing replacement.')
    DATA.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        batches = list(pool.map(prepare, CONFIGS))
    rows = [r for b in batches for r in b['selected_rows']]
    assert len(rows) == sum(b['test_split_total_rows'] for b in batches)
    manifest.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows))
    report = dict(source_repo=REPO, source_revision=REVISION, manifest_path=str(manifest),
        manifest_sha256=digest(manifest), code_sha256=digest(__file__),
        selection='All official-test filenames in each of three languages, ordered by SHA256(12:config:filename).',
        conversations=len(rows), total_duration_seconds=sum(r['duration_seconds'] for r in rows),
        qc_pass=sum(r['reference_qc']=='pass' for r in rows),
        qc_suspect_ids=[r['conversation_id'] for r in rows if r['reference_qc']!='pass'],
        source_archives=[{k:v for k,v in b.items() if k!='selected_rows'} for b in batches],
        limits=['Read speech, not a conversational BRIDGE replication.',
            'Full official-test acquisition; no model results claimed.',
            'No speaker IDs in these TSVs; sentence IDs are not speaker IDs.',
            'Automated QC does not establish transcript correctness.',
            'Public benchmark; model training/evaluation exposure must be disclosed.'],
        test_audio_or_references_opened=True, cloud_cost_usd=0, protocol_sha256=digest(protocol))
    (OUT/'receipt.json').write_text(json.dumps(report, indent=2) + '\n')
    (OUT/'corpus-v1.jsonl').write_bytes(manifest.read_bytes())
    print(json.dumps(dict(event='prepared_fleurs_test', **report)), flush=True)


if __name__ == '__main__':
    main()
