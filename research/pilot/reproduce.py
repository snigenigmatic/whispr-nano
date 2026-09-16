"""Offline audit of the delivered evidence; never schedules compute or inference."""
import json
from pathlib import Path

from research.pilot.evaluate_runs import evaluate, evidence_markdown, compare, sha

ROOT = Path(__file__).resolve().parents[2]
PILOT = ROOT / 'research/pilot'


def main():
    schedule = json.loads((PILOT / 'schedule.json').read_text())
    original = json.loads((PILOT / 'results/audit.json').read_text())
    if original['source_sha256']['evaluator'] != sha(PILOT / 'evaluate_runs.py'):
        raise ValueError('The delivered independent evaluator has changed')
    directories = [PILOT / 'runs' / f"{r['arm_name']}-seed{r['seed']}" for r in schedule['runs']]
    result = evaluate(directories, PILOT / 'data/manifest.jsonl', PILOT / 'selections', schedule)
    compare(result['status'], original['status'], 'Audit completion status')
    compare(result['paired_contrasts'], original['paired_contrasts'], 'Published paired contrasts')
    expected = {(r['arm_name'], r['seed']): r for r in original['runs']}
    for run in result['runs']:
        key = (run['arm_name'], run['seed'])
        for field in ('metrics', 'exposure', 'artifact_sha256', 'initial_state_sha256', 'dev_manifest_sha256'):
            compare(run[field], expected[key][field], f'{key}.{field}')
    output = ROOT / 'tmp/reproduction'
    output.mkdir(parents=True, exist_ok=True)
    (output / 'audit.json').write_text(json.dumps(result, indent=2) + '\n')
    (output / 'evidence.md').write_text(evidence_markdown(result))
    print(f"Offline verification passed for {len(result['runs'])} runs; all published metrics and exposures reproduced.")
    print('No model inference, network request, or paid compute was used.')


if __name__ == '__main__':
    main()
