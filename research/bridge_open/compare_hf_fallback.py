"""Recompute the matched HF fallback comparison from complete saved evidence."""
import json
import hashlib
from collections import Counter
from pathlib import Path
from research.bridge_open.validate_run import validate

ROOT = Path(__file__).resolve().parent
NAMES = ['hf-bengali-turbo-native-portability-v1', 'hf-bengali-turbo-repetition-fallback-v1']


def read(name):
    path = ROOT/'artifacts'/name
    validation = validate(path)
    if not validation['valid']:
        raise ValueError(validation)
    files = {n: json.loads((path/n).read_text()) for n in ['config.json', 'environment.json', 'hf_receipt.json']}
    lines = (path/'predictions.jsonl').read_text().splitlines()
    if len(lines) != 1:
        raise ValueError('Requires exactly one matched conversation')
    row = json.loads(lines[0])
    chunk = row['chunks'][0]
    attempts = [a for step in chunk['native_trace'] for a in step['attempts']]
    details = dict(name=name, metrics=row['metrics'], inference_seconds=row['wall_seconds'],
        selected_calls=len(chunk['generation_sequences']),
        selected_without_eos=sum(50257 not in s for s in chunk['generation_sequences']),
        total_attempts=len(attempts), attempt_temperatures=dict(Counter(str(a['temperature']) for a in attempts)),
        effective_beams=dict(Counter(str(a['effective_num_beams']) for a in attempts)),
        exhausted_schedules=sum(step['attempts'][-1]['needs_fallback'] for step in chunk['native_trace']),
        artifact_hashes={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in path.iterdir() if p.is_file()})
    return files, row, details


def main():
    a, pa, sa = read(NAMES[0]); b, pb, sb = read(NAMES[1])
    allowed = {'cloud_job_id', 'condition_id', 'output_dir', 'temperature', 'compression_ratio_threshold'}
    changed = {k for k in a['config.json'] | b['config.json'] if a['config.json'].get(k) != b['config.json'].get(k)}
    if not changed <= allowed:
        raise ValueError(f'Unexpected condition change: {changed-allowed}')
    for key in ['code_sha256','weight_sha256','auxiliary_file_sha256','packages','dtype']:
        if a['environment.json'][key] != b['environment.json'][key]:
            raise ValueError(f'Runtime or implementation mismatch: {key}')
    for key in ['gpu', 'torch_version', 'cuda_version', 'source_sha256']:
        if a['hf_receipt.json'][key] != b['hf_receipt.json'][key]:
            raise ValueError(f'Provider/entrypoint mismatch: {key}')
    for key in ['conversation_id','audio_sha256','reference_sha256','duration_seconds']:
        if pa[key] != pb[key]:
            raise ValueError(f'Unpaired input: {key}')
    result = dict(scope='One exposed Bengali development conversation, seed12; no population or significance inference.',
        baseline=sa, fallback=sb, changed_config_keys=sorted(changed),
        delta_wer_points=100*(pb['metrics']['wer']-pa['metrics']['wer']),
        delta_cer_points=100*(pb['metrics']['cer']-pa['metrics']['cer']))
    out=ROOT/'artifacts/hf_fallback_comparison';out.mkdir(exist_ok=False)
    (out/'comparison.json').write_text(json.dumps(result,indent=2)+'\n')
    lines=['# Matched HF repetition-fallback diagnostic','',
        'One previously exposed Bengali conversation (humyn-01-01, 15.05 minutes), inference seed 12. Both conditions use identical frozen audio, references, Turbo weights, code, package versions, A100 GPU, float32 precision and scoring.','',
        '| Condition | WER | CER | Selected calls without EOS | Total decoding attempts | Inference time |',
        '|---|---:|---:|---:|---:|---:|']
    for label,s in [('No fallback',sa),('Compression fallback',sb)]:
        lines.append(f"| {label} | {100*s['metrics']['wer']:.2f}% | {100*s['metrics']['cer']:.2f}% | {s['selected_without_eos']}/{s['selected_calls']} | {s['total_attempts']} | {s['inference_seconds']:.1f} s |")
    lines.extend(['',f"Fallback changes WER by {result['delta_wer_points']:+.2f} percentage points and CER by {result['delta_cer_points']:+.2f} points. Negative values favor fallback.",'',
        'The changed recipe uses the Transformers token-byte compression threshold 1.35 and temperatures 0, 0.2, 0.4, 0.6, 0.8, 1.0. Previous-token context, log-probability fallback and silence skipping are disabled in both conditions. The token cap is unchanged. This compares the Transformers native temperature-fallback recipe. Its sampling transition also leaves later temperature-zero windows at beam 1; the observed WER change cannot be attributed to the compression threshold alone. It is not a controlled beam-5 comparison or a test of all recommended Whisper decoding options.','',
        f"Effective beam counts across fallback attempts: {sb['effective_beams']}. Sampling switches to one beam; describing every attempt as beam 5 would be incorrect. {sb['exhausted_schedules']} windows exhausted their temperature schedule while still triggering the detector.",'',
        'All rejected and selected sequences are retained. Hash checks, full input coverage, compression ratios, retry decisions and WER/CER were independently replayed. This is a development diagnostic, not a new ASR method, a held-out improvement, demographic evidence or a publication acceptance claim.'])
    (out/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
