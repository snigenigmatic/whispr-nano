"""Pinned, reference-independent Whisper pilot with append-only evidence."""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import transformers
from huggingface_hub import snapshot_download
from transformers import WhisperForConditionalGeneration, WhisperProcessor

from research.bridge_open.scoring import normalize_v1, score
from research.bridge_open.generation_output import generation_rows


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def now():
    return datetime.now(timezone.utc).isoformat()


def verify(path, expected):
    if sha(path) != expected:
        raise ValueError(f'Content hash mismatch: {path}')


def run(config_path):
    config_path = Path(config_path)
    cfg = json.loads(config_path.read_text())
    if transformers.__version__ != '4.57.6' or importlib.metadata.version('transformers') != '4.57.6':
        raise RuntimeError('Pilot requires matching imported and installed Transformers 4.57.6')
    if cfg['chunk_seconds'] != 30 or cfg['batch_size'] < 1:
        raise ValueError('This frozen condition requires 30-second chunks and positive batch size')
    manifest_path = Path(cfg['manifest_path'])
    verify(manifest_path, cfg['manifest_sha256'])
    records = [json.loads(line) for line in manifest_path.read_text().splitlines()]
    by_id = {r['conversation_id']: r for r in records}
    if len(by_id) != len(records) or len(set(cfg['conversation_ids'])) != len(cfg['conversation_ids']):
        raise ValueError('Duplicate conversation IDs')
    rows = [by_id[cid] for cid in cfg['conversation_ids']]
    for row in rows:
        if row['reference_qc'] != 'pass' or row['language'] not in cfg['supported_languages']:
            raise ValueError('QC failure or unsupported language in configured cohort')
        verify(row['audio_path'], row['audio_sha256'])
    out = Path(cfg['output_dir'])
    out.mkdir(parents=True, exist_ok=False)
    # Freeze effective configuration before opening any model predictions.
    (out / 'config.json').write_text(json.dumps(cfg, indent=2) + '\n')
    torch.manual_seed(cfg['seed'])
    np.random.seed(cfg['seed'])
    torch.set_num_threads(4)
    device = cfg['device']
    if device == 'mps' and not torch.backends.mps.is_available():
        raise RuntimeError('Configured MPS device unavailable; refuse silent fallback')
    if device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('Configured CUDA device unavailable; refuse silent fallback')
    started = time.monotonic()
    snapshot = Path(snapshot_download(cfg['model_id'], revision=cfg['model_revision'],
                    allow_patterns=['*.json', '*.safetensors', '*.txt'], token=False))
    weight_hashes = {p.name: sha(p) for p in sorted(snapshot.glob('*.safetensors'))}
    auxiliary_hashes = {p.name: sha(p) for p in sorted(snapshot.iterdir())
                        if p.is_file() and p.suffix in {'.json', '.txt'}}
    if not weight_hashes:
        raise RuntimeError('No safetensors weights in pinned snapshot')
    processor = WhisperProcessor.from_pretrained(snapshot, local_files_only=True)
    model = WhisperForConditionalGeneration.from_pretrained(snapshot, local_files_only=True,
                torch_dtype=torch.float32, attn_implementation='sdpa').to(device).eval()
    model.generation_config.forced_decoder_ids = None
    evidence = dict(started_at=now(), config_sha256=sha(config_path),
                    manifest_sha256=sha(manifest_path), weight_sha256=weight_hashes,
                    auxiliary_file_sha256=auxiliary_hashes,
                    code_sha256={p.name: sha(p) for p in [Path(__file__), Path(__file__).with_name('scoring.py'), Path(__file__).with_name('generation_output.py')] + ([Path(__file__).with_name('native_whisper.py')] if cfg.get('decoding_mode') == 'native_longform' else [])},
                    packages={n: importlib.metadata.version(n) for n in ['torch', 'transformers', 'numpy', 'scipy', 'soundfile', 'jiwer', 'huggingface-hub']},
                    platform=platform.platform(), device=device, dtype='float32',
                    imported_transformers=transformers.__version__, imported_torch=torch.__version__,
                    model_load_seconds=time.monotonic()-started, cloud_cost_usd=None if cfg.get('executor') in {'modal', 'hf'} else 0)
    (out / 'environment.json').write_text(json.dumps(evidence, indent=2)+'\n')
    print(json.dumps({'event':'run_started', 'config':cfg, 'environment':evidence}), flush=True)
    all_results = []
    with (out / 'predictions.jsonl').open('x') as predictions:
        for row in rows:
            record = dict(record_type='conversation_prediction', schema_version='1.0',
                          conversation_id=row['conversation_id'], language=row['language'],
                          model_id=cfg['model_id'], model_revision=cfg['model_revision'],
                          condition_id=cfg.get('condition_id', 'audio_only_fixed_30s_no_overlap'), attempt=1,
                          audio_sha256=row['audio_sha256'], reference_sha256=row['reference_sha256'],
                          started_at=now(), status='inference_error', chunks=[], failure=None)
            conv_started = time.monotonic()
            try:
                audio, sr = sf.read(row['audio_path'], dtype='float32')
                if sr != 16000 or audio.ndim != 1:
                    raise ValueError('Expected prevalidated mono 16kHz WAV')
                # Absolute audio-clock boundaries only. References are not opened here.
                size = int(cfg['chunk_seconds'] * sr)
                chunks = [(i, audio[i:i+size]) for i in range(0, len(audio), size)]
                max_memory = 0
                if cfg.get('decoding_mode') == 'native_longform':
                    from research.bridge_open.native_whisper import decode_native
                    record['chunks'] = [decode_native(model, processor, audio, sr, cfg, row['language'], started)]
                    chunks = []
                    torch.cuda.synchronize()
                    max_memory = torch.cuda.max_memory_allocated()
                for start in range(0, len(chunks), cfg['batch_size']):
                    if time.monotonic() - started > cfg.get('max_wall_seconds', 3600):
                        raise TimeoutError('Configured local run wall-time limit reached')
                    group = chunks[start:start+cfg['batch_size']]
                    inputs = processor([a for _,a in group], sampling_rate=sr,
                                       return_tensors='pt', return_attention_mask=True)
                    with torch.inference_mode():
                        generated = model.generate(input_features=inputs.input_features.to(device),
                                             attention_mask=inputs.attention_mask.to(device),
                                             language=row['language'], task='transcribe',
                                             do_sample=False, num_beams=cfg.get('num_beams', 1),
                                             return_timestamps=cfg.get('return_timestamps', False),
                                             return_dict_in_generate=True,
                                             max_new_tokens=cfg['max_new_tokens'])
                    if device == 'mps':
                        torch.mps.synchronize()
                        max_memory = max(max_memory, torch.mps.driver_allocated_memory())
                    elif device == 'cuda':
                        torch.cuda.synchronize()
                        max_memory = max(max_memory, torch.cuda.max_memory_allocated())
                    # Whisper's default wrapper strips EOS. The raw generate
                    # ModelOutput preserves it, so termination checks are valid.
                    decoded_rows = generation_rows(generated, model.config.eos_token_id)
                    texts = processor.batch_decode([r['token_ids'] for r in decoded_rows], skip_special_tokens=True)
                    for (offset, samples), text, decoded in zip(group, texts, decoded_rows):
                        record['chunks'].append(dict(start_seconds=offset/sr,
                                  end_seconds=(offset+len(samples))/sr, text_raw=text,
                                  **decoded))
                    print(json.dumps({'event':'batch_done','conversation_id':row['conversation_id'],
                                     'chunks_done':len(record['chunks']), 'chunks_total':len(chunks)}), flush=True)
                record['text_raw'] = ' '.join(c['text_raw'].strip() for c in record['chunks'])
                record['text_for_scoring'] = normalize_v1(record['text_raw'])
                record['status'] = 'success' if record['text_for_scoring'] else 'empty_output'
                if any(not chunk['eos_present'] for chunk in record['chunks']):
                    record['status'] = 'generation_limit'
                    record['failure'] = {'type':'generation_truncated',
                                         'message':'All audio processed; at least one chunk reached generation cap without EOS. Retain bounded-output errors in pilot scores.'}
                record['sampled_max_mps_driver_bytes'] = max_memory if device == 'mps' else None
                record['peak_cuda_allocated_bytes'] = max_memory if device == 'cuda' else None
                record['duration_seconds'] = len(audio)/sr
                # References become available only after full-conversation generation.
                verify(row['reference_path'], row['reference_sha256'])
                reference = json.loads(Path(row['reference_path']).read_text())['reference_text']
                record['metrics'] = score(reference, record['text_raw'])
            except Exception as exc:
                record['status'] = 'inference_error'
                record['failure'] = {'type':type(exc).__name__, 'message':str(exc)}
                traceback.print_exc()
            record['ended_at'] = now()
            record['wall_seconds'] = time.monotonic()-conv_started
            if 'duration_seconds' in record:
                record['real_time_factor'] = record['wall_seconds']/record['duration_seconds']
            predictions.write(json.dumps(record, ensure_ascii=False)+'\n')
            predictions.flush()
            all_results.append(record)
            print(json.dumps({'event':'conversation_done', **{k:v for k,v in record.items() if k not in ['chunks','text_raw','text_for_scoring']}}), flush=True)
    success = [r for r in all_results if r['status']=='success']
    # Model-quality failures must not disappear from an aggregate. Include all
    # completely processed audio, even if its decoded text repeated/hit a cap.
    scored = [r for r in all_results if r['status'] in {'success','empty_output','generation_limit'} and 'metrics' in r]
    aggregates = {}
    for language in sorted({r['language'] for r in rows}):
        group = [r for r in scored if r['language']==language]
        if not group:
            continue
        word_errors = sum(sum(r['metrics'][k] for k in ['substitutions','deletions','insertions']) for r in group)
        ref_words = sum(r['metrics']['ref_words'] for r in group)
        character_errors = sum(r['metrics']['character_errors'] for r in group)
        ref_chars = sum(r['metrics']['ref_chars'] for r in group)
        aggregates[language] = dict(estimand='all fully processed conversations including empty/capped text; runtime failures reported separately',
            conversation_ids=[r['conversation_id'] for r in group],
            scored_count=len(group), attempted_count=sum(r['language']==language for r in rows),
            clean_completion_count=sum(r['status']=='success' for r in group),
            generation_limit_count=sum(r['status']=='generation_limit' for r in group),
            word_errors=word_errors, ref_words=ref_words, character_errors=character_errors, ref_chars=ref_chars,
            micro_wer=word_errors/ref_words, micro_cer=character_errors/ref_chars,
            mean_conversation_wer=sum(r['metrics']['wer'] for r in group)/len(group),
            mean_conversation_wil=sum(r['metrics']['wil'] for r in group)/len(group))
    summary = dict(status='completed' if len(success)==len(rows) else 'completed_with_quality_flags' if len(scored)==len(rows) else 'completed_with_runtime_failures',
                   scope='engineering/public-data pilot; not BRIDGE-matched or population evidence',
                   attempted=len(rows), successful=len(success), scored=len(scored),
                   conversation_ids=cfg['conversation_ids'],
                   per_language=aggregates,
                   results=[{k:r.get(k) for k in ['conversation_id','language','status','metrics','wall_seconds','real_time_factor']} for r in all_results],
                   total_wall_seconds=time.monotonic()-started, cloud_cost_usd=None if cfg.get('executor') in {'modal', 'hf'} else 0)
    (out / 'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    print(json.dumps({'event':'final_summary', **summary}), flush=True)
    return 0 if len(scored)==len(rows) else 1


if __name__ == '__main__':
    raise SystemExit(run(Path(__file__).with_name('run_config.json')))
