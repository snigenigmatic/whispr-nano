"""Generate a comparison only after independently validating every run."""
import argparse
import json
from pathlib import Path
from validate_run import validate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('runs', nargs='+', type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    rows = []
    cohort = None
    for directory in args.runs:
        report = validate(directory)
        if not report['valid']:
            raise ValueError(report)
        cfg = json.loads((directory/'config.json').read_text())
        summary = json.loads((directory/'summary.json').read_text())
        predictions = [json.loads(line) for line in (directory/'predictions.jsonl').read_text().splitlines()]
        key = (cfg['manifest_sha256'], tuple(cfg['conversation_ids']))
        if cohort is not None and key != cohort:
            raise ValueError('Refuse comparison across different cohorts')
        cohort = key
        total_words = sum(p['metrics']['ref_words'] for p in predictions)
        total_chars = sum(p['metrics']['ref_chars'] for p in predictions)
        errors = sum(sum(p['metrics'][k] for k in ['substitutions','deletions','insertions']) for p in predictions)
        char_errors = sum(p['metrics']['character_errors'] for p in predictions)
        receipt = json.loads((directory/'cloud_receipt.json').read_text()) if (directory/'cloud_receipt.json').exists() else None
        rows.append(dict(run=directory.name, model=cfg['model_id'], device=cfg['device'],
            beams=cfg.get('num_beams',1), timestamps=cfg.get('return_timestamps',False), batch_size=cfg['batch_size'],
            wer_percent=100*errors/total_words, cer_percent=100*char_errors/total_chars,
            capped_chunks=sum(not c['eos_present'] for p in predictions for c in p['chunks']),
            total_chunks=sum(len(p['chunks']) for p in predictions),
            per_conversation_wer={p['conversation_id']:p['metrics']['wer'] for p in predictions},
            wall_seconds=summary['total_wall_seconds'],
            estimated_function_resource_usd=receipt['estimated_function_resource_usd'] if receipt else 0,
            provider_bill_verified=False if receipt else None))
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(dict(scope='three-conversation public Hindi engineering pilot; no held-out or significance claim',cohort_manifest_sha256=cohort[0],runs=rows),indent=2)+'\n')
    for row in rows:
        print(f"{row['run']}: WER {row['wer_percent']:.2f}%, CER {row['cer_percent']:.2f}%, capped {row['capped_chunks']}/{row['total_chunks']}, resource estimate ${row['estimated_function_resource_usd']:.3f}")


if __name__ == '__main__':
    main()
