"""CPU-only source verification and exact-silence audit; no ASR-based exclusions."""
from __future__ import annotations

import hashlib
import io
import json
import math
import platform
from pathlib import Path

import numpy as np
import pyarrow
import pyarrow.parquet as pq
import scipy
import soundfile as sf
from huggingface_hub import hf_hub_download
from scipy.signal import resample_poly

ROOT = Path(__file__).resolve().parents[2]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    cfg = json.loads((ROOT / 'research/bridge_open/run_config.json').read_text())
    manifest = Path(cfg['manifest_path'])
    assert sha(manifest) == cfg['manifest_sha256']
    out = ROOT / cfg['analysis_output']
    out.mkdir(parents=True, exist_ok=False)
    cached_name, cached_rows = None, None
    results = []
    for row in map(json.loads, manifest.read_text().splitlines()):
        assert sha(row['audio_path']) == row['audio_sha256']
        assert sha(row['reference_path']) == row['reference_sha256']
        if cached_name != row['source_shard']:
            path = hf_hub_download(row['source_repo'], row['source_shard'],
                repo_type='dataset', revision=row['source_revision'], token=False,
                local_files_only=True)
            assert sha(path) == row['shard_sha256']
            cached_rows = pq.read_table(path).to_pylist()
            cached_name = row['source_shard']
        original = cached_rows[int(row['conversation_id'].rsplit('-', 1)[1])]
        encoded = original['audio']['bytes']
        assert hashlib.sha256(encoded).hexdigest() == row['source_audio_sha256']
        source, source_sr = sf.read(io.BytesIO(encoded), dtype='float32', always_2d=True)
        audio, sr = sf.read(row['audio_path'], dtype='float32')
        assert sr == 16000 and audio.ndim == 1
        assert np.isfinite(source).all() and np.isfinite(audio).all()
        reconstructed = source.mean(axis=1)
        if source_sr != sr:
            divisor = math.gcd(source_sr, sr)
            reconstructed = resample_poly(reconstructed, sr // divisor,
                source_sr // divisor).astype('float32')
        equal = bool(np.array_equal(reconstructed, audio))
        assert equal, f"{row['conversation_id']}: source reconstruction differs"
        references = json.loads(Path(row['reference_path']).read_text())['segments']
        segments = []
        for index, segment in enumerate(references):
            start, end = segment['start_seconds'], segment['end_seconds']
            in_bounds = 0 <= start < end <= len(audio) / sr
            samples = audio[max(0, math.floor(start * sr)):min(len(audio), math.ceil(end * sr))]
            all_zero = bool(in_bounds and len(samples) and not np.any(samples))
            segments.append(dict(index=index, start_seconds=start, end_seconds=end,
                duration_seconds=end-start, in_bounds=in_bounds,
                nonempty_reference=bool(segment['text'].strip()),
                raw_whitespace_word_count=len(segment['text'].split()),
                all_zero_audio=all_zero,
                zero_sample_fraction=float(np.mean(samples == 0)) if len(samples) else None))
        silent_text = [s for s in segments if s['nonempty_reference'] and s['all_zero_audio']]
        one_second_windows = [audio[i:i+sr] for i in range(0, len(audio), sr)]
        result = dict(conversation_id=row['conversation_id'], language=row['language'],
            duration_seconds=len(audio)/sr, source_sample_rate=source_sr,
            source_channels=source.shape[1], source_zero_fraction=float(np.mean(source == 0)),
            source_channel_rms=np.sqrt(np.mean(source.astype('float64')**2, axis=0)).tolist(),
            prepared_zero_fraction=float(np.mean(audio == 0)),
            all_zero_one_second_windows=sum(not np.any(w) for w in one_second_windows),
            prepared_peak=float(np.max(np.abs(audio))),
            samplewise_equal_to_reconstructed_source=equal,
            source_parquet_sha256=row['shard_sha256'], source_audio_sha256=row['source_audio_sha256'],
            prepared_audio_sha256=row['audio_sha256'], reference_sha256=row['reference_sha256'],
            previous_reference_qc=row['reference_qc'],
            silent_nonempty_reference_segments=len(silent_text),
            silent_reference_raw_words=sum(s['raw_whitespace_word_count'] for s in silent_text),
            total_reference_raw_words=sum(s['raw_whitespace_word_count'] for s in segments),
            segments=segments)
        results.append(result)
        print(json.dumps(dict(event='audio_content_audit', **{k:v for k,v in result.items()
            if k not in ('segments', 'source_parquet_sha256', 'source_audio_sha256',
                'prepared_audio_sha256', 'reference_sha256', 'source_channel_rms')})), flush=True)
    report = dict(config=cfg, code_sha256=sha(__file__), python_version=platform.python_version(),
        numpy_version=np.__version__, scipy_version=scipy.__version__,
        pyarrow_version=pyarrow.__version__, soundfile_version=sf.__version__,
        cloud_cost_usd=0, conversations=len(results),
        files_with_silent_nonempty_reference_segments=sum(r['silent_nonempty_reference_segments'] > 0 for r in results),
        criteria='Exact zero at every sample within a fully in-bounds nonempty transcript interval, using floor(start*sr) and ceil(end*sr). No ASR outputs or WER used. Silence alone is not an error; nonempty transcript over silence requires adjudication.',
        interpretation='These findings concern this pinned public release and do not establish problems in the private BRIDGE evaluation corpus. Unflagged files have not received human transcript validation.',
        rows=results)
    (out / 'evidence.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k:v for k,v in report.items() if k != 'rows'}), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
