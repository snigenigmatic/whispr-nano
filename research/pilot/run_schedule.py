"""Execute the frozen pilot once, with explicit receipts and a delivery deadline."""
from datetime import datetime, timezone
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parents[2]
PILOT = ROOT / 'research/pilot'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume-prepared', action='store_true')
    args = parser.parse_args()
    deadline = json.loads((PILOT / 'deadline.json').read_text())
    speakers = json.loads((PILOT / 'data/speakers.json').read_text())
    training_speakers = {s['speaker_id'] for s in speakers if s['split'] == 'train'}
    while True:
        ready = {p.name.removesuffix('.runtime.json') for p in (PILOT / 'teacher_predictions').glob('*.runtime.json')}
        if training_speakers <= ready and (PILOT / 'feature_cache_receipt.json').exists():
            break
        if time.time() + 1200 >= deadline['experiment_cutoff_epoch']:
            raise TimeoutError('Insufficient time remains for registered training after preparation')
        time.sleep(10)
    print('Preparation complete; verifying teacher provenance and freezing selection.', flush=True)
    all_predictions = []
    for speaker in sorted(training_speakers):
        path = PILOT / 'teacher_predictions' / f'{speaker}.jsonl'
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        runtime = json.loads(path.with_suffix('.runtime.json').read_text())
        if runtime['model_revision'] != '06f233fe06e710322aca913c1bc4249a0d71fce1':
            raise ValueError('Teacher revision changed')
        if runtime['manifest_sha256'] != hashlib.sha256((PILOT / 'data/manifest.jsonl').read_bytes()).hexdigest():
            raise ValueError('Teacher data identity changed')
        if runtime['completed_rows'] != len(rows) or any(r['split'] != 'train' or r['speaker_id'] != speaker for r in rows):
            raise ValueError('Teacher returned incomplete or unassigned rows')
        all_predictions.extend(rows)
    predictions = PILOT / 'teacher_predictions/all_train.jsonl'
    if args.resume_prepared:
        report = json.loads((PILOT / 'selections/selection_report.json').read_text())
        if report['teacher_predictions_sha256'] != hashlib.sha256(predictions.read_bytes()).hexdigest():
            raise ValueError('Prepared teacher evidence changed')
        for arm, checksum in report['output_sha256'].items():
            if checksum != hashlib.sha256((PILOT / 'selections' / f'{arm}.jsonl').read_bytes()).hexdigest():
                raise ValueError('Prepared selection changed')
    else:
        if predictions.exists() or (PILOT / 'selections').exists():
            raise ValueError('This controller refuses to replace existing selection; explicitly resume prepared inputs')
        predictions.write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in all_predictions))
        subprocess.run([str(ROOT / '.venv/bin/python'), '-m', 'research.pilot.selection',
                        '--predictions', str(predictions)], cwd=ROOT, check=True)
    for seed in (0, 1, 2):
        if (PILOT / 'ledger').exists():
            from research.pilot.loop import status
            states = status(PILOT / 'ledger')['jobs']
            if all(states.get(f'{arm}-seed{seed}') == 'passed' for arm in ('full_gold', 'filtered10_gold', 'random_matched_gold')):
                print(f'Preserving completed seed {seed}.', flush=True)
                continue
        log = PILOT / f'train-seed{seed}.log'
        print(f"Starting seed {seed} at {datetime.now(timezone.utc).isoformat()}", flush=True)
        with log.open('x') as stream:
            process = subprocess.run([str(ROOT / '.venv/bin/modal'), 'run',
                    'research/pilot/modal_runner.py::train_registered', '--seed', str(seed)],
                    stdout=stream, stderr=subprocess.STDOUT, cwd=ROOT)
        if process.returncode:
            raise RuntimeError(f'Seed wave {seed} stopped with code {process.returncode}; inspect {log}')
        print(f'All three arms completed for seed {seed}.', flush=True)
    from research.pilot.workflow import finalize
    print(json.dumps(finalize(), indent=2), flush=True)
    print('Frozen experiment complete; full evidence available for review.', flush=True)


if __name__ == '__main__':
    main()
