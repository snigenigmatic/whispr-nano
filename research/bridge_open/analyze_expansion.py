"""CPU-only frozen evidence replay for the multilingual expansion."""
import csv
import hashlib
import json
from pathlib import Path

from research.bridge_open.analyze_benchmark import main as aggregate_main
from research.bridge_open.validate_run import validate

ROOT = Path(__file__).resolve().parents[2]


def main():
    cfg = json.loads((ROOT / 'research/bridge_open/run_config.json').read_text())
    assert cfg['executor'] == 'analysis_expansion'
    output = ROOT / 'research/bridge_open/artifacts/expanded_benchmark'
    output.mkdir(parents=True, exist_ok=False)
    directories, validations = [], []
    for item in cfg['input_runs']:
        directory = ROOT / item['path']
        for name, expected in item['sha256'].items():
            if hashlib.sha256((directory / name).read_bytes()).hexdigest() != expected:
                raise ValueError(f'Frozen input hash mismatch: {directory / name}')
        result = validate(directory)
        if not result['valid']:
            raise ValueError(result)
        validations.append(result)
        directories.append(str(directory))
        run_cfg = json.loads((directory / 'config.json').read_text())
        summary = json.loads((directory / 'summary.json').read_text())
        print(json.dumps(dict(event='validated_input', name=directory.name,
                              config=run_cfg, summary=summary, validation=result)), flush=True)
    (output / 'validation.json').write_text(json.dumps(validations, indent=2) + '\n')
    aggregate_main(directories + ['--output-json', str(output / 'comparison.json'),
                                 '--output-csv', str(output / 'conversations.csv')])
    result = json.loads((output / 'comparison.json').read_text())
    with (output / 'per_language.csv').open('w', newline='') as stream:
        fields = ['model_id', 'condition_id', 'language', 'attempted', 'scored',
                  'ref_words', 'word_errors', 'micro_wer', 'micro_cer', 'capped_chunks']
        writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader()
        for run in result['runs']:
            for lang, metrics in run['summary']['languages'].items():
                convs = [c for c in run['conversations'] if c['language'] == lang]
                writer.writerow(dict(model_id=run['model_id'], condition_id=run['condition_id'],
                    language=lang, attempted=len(convs), scored=metrics['scored'],
                    ref_words=metrics['ref_words'], word_errors=metrics['word_errors'],
                    micro_wer=metrics['micro_wer'], micro_cer=metrics['micro_cer'],
                    capped_chunks=sum(c['capped_chunks'] for c in convs)))
    manifest = [json.loads(line) for line in Path(cfg['manifest_path']).read_text().splitlines()]
    with (output / 'coverage.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=['conversation_id', 'language', 'reference_qc',
            'qc_issues', 'model_id', 'execution_status', 'role'])
        writer.writeheader()
        for run in result['runs']:
            by_id = {r['conversation_id']: r for r in run['conversations']}
            for row in manifest:
                writer.writerow(dict(conversation_id=row['conversation_id'], language=row['language'],
                    reference_qc=row['reference_qc'], qc_issues=json.dumps(row['qc_issues']),
                    model_id=run['model_id'],
                    execution_status=by_id.get(row['conversation_id'], {}).get('status',
                        'excluded_pre_run_qc' if row['reference_qc'] != 'pass' else 'not_run'),
                    role='development' if row['language'] == 'hi' else 'exploratory_expansion'))
    print(json.dumps(dict(event='final_comparison', all_valid=True,
        runs=[dict(model=r['model_id'], summary=r['summary']) for r in result['runs']])), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
