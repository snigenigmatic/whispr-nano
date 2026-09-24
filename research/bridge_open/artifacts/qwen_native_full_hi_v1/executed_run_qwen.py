"""Pinned, reference-independent Qwen3-ASR pilot with append-only evidence."""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import time
import traceback
import types
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import transformers
from huggingface_hub import snapshot_download
from qwen_asr import Qwen3ASRModel

from research.bridge_open.scoring import normalize_v1, score
from research.bridge_open.qwen_trace import token_termination


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
    if cfg.get('inference_api') != 'Qwen3ASRModel.transcribe':
        raise ValueError('This runner requires the documented Qwen transcribe API')
    if cfg['batch_size'] < 1:
        raise ValueError('batch_size must be positive')
    manifest_path = Path(cfg['manifest_path'])
    verify(manifest_path, cfg['manifest_sha256'])
    records = [json.loads(line) for line in manifest_path.read_text().splitlines()]
    by_id = {r['conversation_id']: r for r in records}
    if len(by_id) != len(records) or len(set(cfg['conversation_ids'])) != len(cfg['conversation_ids']):
        raise ValueError('Duplicate conversation IDs')
    rows = [by_id[cid] for cid in cfg['conversation_ids']]
    validated_audio = {}
    for row in rows:
        if row['reference_qc'] != 'pass' or row['language'] not in cfg['supported_languages']:
            raise ValueError('QC failure or unsupported language in configured cohort')
        verify(row['audio_path'], row['audio_sha256'])
        verify(row['reference_path'], row['reference_sha256'])
        audio, sr = sf.read(row['audio_path'], dtype='float32')
        peak = float(np.max(np.abs(audio))) if audio.size else 0.0
        if sr != 16000 or audio.ndim != 1:
            raise ValueError(f"{row['conversation_id']}: expected prevalidated mono 16kHz WAV")
        if not np.isfinite(audio).all() or peak <= 0.0 or peak > 1.0:
            raise ValueError(f"{row['conversation_id']}: non-finite, silent, or out-of-range samples")
        if len(audio) / sr > cfg.get('native_max_audio_seconds', 1200):
            raise ValueError('This trace-mapping contract requires one native chunk per input')
        validated_audio[row['conversation_id']] = (audio, sr, peak)
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
    if cfg['num_beams'] != 1 or cfg['return_timestamps'] or cfg['supported_languages'] != ['hi']:
        raise ValueError('This pilot freezes Hindi text-only Qwen inference')
    qwen_dtype = getattr(torch, cfg.get('torch_dtype', 'bfloat16'))
    if qwen_dtype not in {torch.float32, torch.float16, torch.bfloat16}:
        raise ValueError('unsupported torch_dtype')
    qwen = Qwen3ASRModel.from_pretrained(str(snapshot), local_files_only=True,
                dtype=qwen_dtype, device_map=cfg.get('device_map', 'cuda:0'), attn_implementation='sdpa',
                max_inference_batch_size=cfg['batch_size'], max_new_tokens=cfg['max_new_tokens'])
    processor, model = qwen.processor, qwen.model.eval()
    if model.training or model.generation_config.do_sample or model.generation_config.num_beams != 1:
        raise ValueError('Native Qwen requires eval mode and pinned greedy generation settings')
    expected_eos = json.loads((snapshot/'generation_config.json').read_text())['eos_token_id']
    if cfg['eos_token_ids'] != expected_eos:
        raise ValueError('Configured EOS differs from pinned model generation config')
    generation_capture = []
    original_generate = model.generate
    def observed_generate(module, *args, **kwargs):
        output = original_generate(*args, **kwargs)
        input_ids = kwargs.get('input_ids')
        sequences = getattr(output, 'sequences', output)
        if sequences is None or input_ids is None:
            generation_capture.append({'observation_status': 'missing_sequences_or_prompt'})
            return output
        raw = sequences.detach().cpu().tolist() if hasattr(sequences, 'detach') else sequences.tolist()
        prompt_length = int(input_ids.shape[1])
        continuation = [seq[prompt_length:] for seq in raw]
        try:
            decoded = processor.batch_decode(continuation, skip_special_tokens=True,
                                             clean_up_tokenization_spaces=False)
        except Exception:
            decoded = None
        termination = [token_termination(seq, cfg['eos_token_ids'], cfg['max_new_tokens']) for seq in continuation]
        generation_capture.append(dict(observation_status='captured', trace_schema=2, prompt_tokens=prompt_length,
            batch_size=len(continuation), raw_sequences=raw, new_token_sequences=continuation,
            new_token_counts=[t['new_token_counts'] for t in termination],
            max_new_tokens=cfg['max_new_tokens'],
            cap_observed=[t['cap_observed'] for t in termination],
            raw_decoded_text=decoded,
            eos_present=[t['eos_present'] for t in termination]))
        return output
    model.generate = types.MethodType(observed_generate, model)
    package_files = {}
    for package_file in importlib.metadata.files('qwen-asr') or []:
        candidate = Path(importlib.metadata.distribution('qwen-asr').locate_file(package_file))
        if candidate.is_file():
            package_files[str(package_file)] = sha(candidate)
    evidence = dict(started_at=now(), config_sha256=sha(config_path),
                    manifest_sha256=sha(manifest_path), weight_sha256=weight_hashes,
                    auxiliary_file_sha256=auxiliary_hashes,
                    qwen_asr_package_file_sha256=package_files,
                    code_sha256={p.name: sha(p) for p in [Path(__file__), Path(__file__).with_name('scoring.py'), Path(__file__).with_name('qwen_trace.py')]},
                    packages={n: importlib.metadata.version(n) for n in ['torch', 'transformers', 'numpy', 'scipy', 'soundfile', 'jiwer', 'huggingface-hub', 'qwen-asr', 'accelerate']},
                    platform=platform.platform(), device=device, dtype=cfg.get('torch_dtype', 'bfloat16'), parameter_count=sum(p.numel() for p in model.parameters()),
                    imported_transformers=transformers.__version__, imported_torch=torch.__version__,
                    model_eval=not model.training, actual_model_dtype=str(model.dtype),
                    effective_generation_config=model.generation_config.to_dict(),
                    model_load_seconds=time.monotonic()-started, cloud_cost_usd=None if cfg.get('executor') in {'modal', 'hf'} else 0)
    (out / 'environment.json').write_text(json.dumps(evidence, indent=2)+'\n')
    print(json.dumps({'event':'run_started', 'config':cfg, 'environment':evidence}), flush=True)
    all_results = []
    with (out / 'predictions.jsonl').open('x') as predictions:
        for group_start in range(0, len(rows), cfg['batch_size']):
            group_rows = rows[group_start:group_start + cfg['batch_size']]
            group_id = f"group-{group_start // cfg['batch_size']:04d}"
            group_started = time.monotonic()
            records = []
            audios = []
            try:
                for row in group_rows:
                    record = dict(record_type='conversation_prediction', schema_version='1.1',
                        conversation_id=row['conversation_id'], language=row['language'],
                        model_id=cfg['model_id'], model_revision=cfg['model_revision'],
                        condition_id=cfg.get('condition_id', 'qwen_native_transcribe'), attempt=1,
                        audio_sha256=row['audio_sha256'], reference_sha256=row['reference_sha256'],
                        started_at=now(), status='inference_error', chunks=[], failure=None,
                        inference_group_id=group_id)
                    audio, sr, peak = validated_audio[row['conversation_id']]
                    audios.append((audio, sr)); records.append((row, record, peak))
                if time.monotonic() - started > cfg.get('max_wall_seconds', 3600):
                    raise TimeoutError('Configured local run wall-time limit reached')
                generation_capture.clear()
                results = qwen.transcribe(audios, language=['Hindi'] * len(audios),
                                          return_time_stamps=False)
                if not isinstance(results, list) or len(results) != len(records):
                    raise ValueError(f'Qwen transcribe cardinality mismatch: expected {len(records)}, got {len(results) if isinstance(results, list) else type(results).__name__}')
                trace_rows = [[] for _ in records]
                total_trace_rows = 0
                for capture in generation_capture:
                    raw_rows = capture.get('new_token_sequences', [])
                    if capture.get('observation_status') != 'captured':
                        raise ValueError('Qwen generate observation missing raw sequences')
                    if len(raw_rows) != len(records):
                        raise ValueError(f"Qwen generate trace cardinality mismatch: expected {len(records)}, got {len(raw_rows)}")
                    total_trace_rows += len(raw_rows)
                    for index in range(len(records)):
                        per_record = dict(capture)
                        for key in ('raw_sequences', 'new_token_sequences', 'raw_decoded_text', 'new_token_counts', 'cap_observed', 'eos_present'):
                            value = per_record.get(key)
                            if isinstance(value, list) and len(value) == len(records):
                                per_record[key] = value[index]
                        per_record['record_index'] = index
                        trace_rows[index].append(per_record)
                if total_trace_rows != len(records):
                    raise ValueError(f'Qwen generate trace total mismatch: expected {len(records)}, got {total_trace_rows}')
                for index, ((row, record, peak), result) in enumerate(zip(records, results)):
                    text = result.text or ''
                    record['chunks'].append(dict(start_seconds=0.0, end_seconds=len(audios[index][0])/16000,
                        text_raw=text, language_detected=result.language,
                        token_evidence='captured_from_generate', eos_present=None,
                        generation_sequences=[], generation_trace=trace_rows[index],
                        generation_observation='raw generate output captured; API text remains primary'))
                    record['text_raw'] = text.strip()
                    record['text_for_scoring'] = normalize_v1(record['text_raw'])
                    record['status'] = 'success' if record['text_for_scoring'] else 'empty_output'
                    record['termination_semantics'] = 'qwen_transcribe_api'
                    record['sampled_max_mps_driver_bytes'] = None
                    record['peak_cuda_allocated_bytes'] = None
                    record['duration_seconds'] = len(audios[index][0])/16000
                    record['audio_peak_abs'] = peak
                    reference = json.loads(Path(row['reference_path']).read_text())['reference_text']
                    record['metrics'] = score(reference, record['text_raw'])
            except Exception as exc:
                for row, record, _peak in records:
                    record['status'] = 'inference_error'
                    record['failure'] = {'type':type(exc).__name__, 'message':str(exc)}
                if not records:
                    raise
                traceback.print_exc()
            group_wall = time.monotonic() - group_started
            for row, record, _peak in records:
                record['ended_at'] = now()
                record['wall_seconds'] = None
                record['timing_semantics'] = 'shared batch wall time; no per-record latency estimate'
                record['group_wall_seconds'] = group_wall
                record['group_size'] = len(group_rows)
                record['group_audio_seconds'] = sum(len(a[0])/a[1] for a in audios)
                record['group_real_time_factor'] = group_wall / record['group_audio_seconds']
                record['real_time_factor'] = None
                predictions.write(json.dumps(record, ensure_ascii=False) + '\n')
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
        aggregates[language] = dict(estimand='all fully processed conversations; runtime failures reported separately',
            conversation_ids=[r['conversation_id'] for r in group],
            scored_count=len(group), attempted_count=sum(r['language']==language for r in rows),
            clean_completion_count=sum(r['status']=='success' for r in group),
            generation_limit_count=None,
            generation_limit_observable=False,
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
