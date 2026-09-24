"""CPU-only evidence replay; no provider APIs, training, or generation."""
import hashlib
import json
from pathlib import Path
from research.bridge_open.validate_run import validate
from research.bridge_open.scoring import score

ROOT = Path(__file__).resolve().parents[2]


def main():
    cfg = json.loads((ROOT/'research/bridge_open/run_config.json').read_text())
    assert cfg['executor'] == 'analysis'
    output = ROOT/'research/bridge_open/artifacts/validated_comparison'
    output.mkdir(parents=True, exist_ok=False)
    rows = []
    for item in cfg['input_runs']:
        directory = ROOT/item['path']
        for filename, expected in item['sha256'].items():
            actual = hashlib.sha256((directory/filename).read_bytes()).hexdigest()
            if actual != expected:
                raise ValueError(f'Input evidence changed: {directory/filename}')
        validation = validate(directory)
        if not validation['valid']:
            raise ValueError(validation)
        run_cfg = json.loads((directory/'config.json').read_text())
        summary = json.loads((directory/'summary.json').read_text())
        predictions = [json.loads(line) for line in (directory/'predictions.jsonl').read_text().splitlines()]
        receipt_path = directory/'cloud_receipt.json'
        receipt = json.loads(receipt_path.read_text()) if receipt_path.exists() else None
        record = dict(name=directory.name, config=run_cfg, validation=validation,
            summary=summary, cloud_receipt=receipt,
            capped_chunks=sum(not c['eos_present'] for p in predictions for c in p['chunks']),
            total_chunks=sum(len(p['chunks']) for p in predictions))
        if run_cfg.get('model_family') == 'qwen3_asr':
            manifest = {r['conversation_id']:r for r in [json.loads(line) for line in Path(run_cfg['manifest_path']).read_text().splitlines()]}
            secondary = []
            changed_chunks = 0
            for prediction in predictions:
                ref = json.loads(Path(manifest[prediction['conversation_id']]['reference_path']).read_text())['reference_text']
                parsed = ' '.join(c['text_official_parser'].strip() for c in prediction['chunks'])
                if parsed != prediction['text_official_parser']:
                    raise ValueError('Official parser text concatenation mismatch')
                metrics = score(ref, parsed)
                if metrics != prediction['metrics_official_parser']:
                    raise ValueError('Official parser secondary metric mismatch')
                secondary.append(metrics)
                changed_chunks += sum(c['text_raw'].strip() != c['text_official_parser'].strip() for c in prediction['chunks'])
            record['official_parser_sensitivity'] = dict(changed_chunks=changed_chunks,
                micro_wer=sum(m['substitutions']+m['deletions']+m['insertions'] for m in secondary)/sum(m['ref_words'] for m in secondary))
        print(json.dumps({'event':'validated_run', **record}), flush=True)
        rows.append(record)
    cohort_keys = {(r['config']['manifest_sha256'], tuple(r['config']['conversation_ids'])) for r in rows}
    if len(cohort_keys) != 1:
        raise ValueError('Comparison cohort mismatch')
    result = dict(scope='three public Hindi development conversations; descriptive only',runs=rows)
    (output/'comparison.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'event':'final_comparison','all_valid':True,'runs':[
        dict(name=r['name'],wer=r['summary']['per_language']['hi']['micro_wer'],
             cer=r['summary']['per_language']['hi']['micro_cer'],capped_chunks=r['capped_chunks']) for r in rows]}),flush=True)
    return 0
