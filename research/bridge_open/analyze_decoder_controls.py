"""Replay frozen decoder evidence and compare exact matched conversations."""
import csv
import hashlib
import json
from pathlib import Path
from research.bridge_open.validate_run import validate
ROOT=Path(__file__).resolve().parents[2]


def main():
    cfg=json.loads((ROOT/'research/bridge_open/run_config.json').read_text())
    out=ROOT/cfg['analysis_output'];out.mkdir(parents=True,exist_ok=False)
    rows=[];details=[];validations=[];shared_ids=None;shared_refs=None
    for item in cfg['input_runs']:
        directory=ROOT/item['path']
        for name,expected in item['sha256'].items():
            if hashlib.sha256((directory/name).read_bytes()).hexdigest()!=expected:
                raise ValueError(f'Input changed: {directory/name}')
        valid=validate(directory)
        if not valid['valid']:raise ValueError(valid)
        validations.append(valid)
        config=json.loads((directory/'config.json').read_text())
        preds=[json.loads(l) for l in (directory/'predictions.jsonl').read_text().splitlines()]
        summary=json.loads((directory/'summary.json').read_text())
        ids={p['conversation_id'] for p in preds}
        refs={p['conversation_id']:p['reference_sha256'] for p in preds}
        if shared_ids is None:shared_ids,shared_refs=ids,refs
        if ids!=shared_ids or refs!=shared_refs:raise ValueError('Unmatched cohort or reference')
        calls=[s for p in preds for c in p['chunks'] for s in c['generation_sequences']]
        eos=config.get('eos_token_ids',[50257])
        caps=sum(not any(t in s for t in eos) for s in calls)
        m={key:sum(p['metrics'][key] for p in preds) for key in ['substitutions','deletions','insertions','ref_words','character_errors','ref_chars']}
        n=m['ref_words']
        row=dict(recipe=item['label'],model_id=config['model_id'],condition=config['condition_id'],
                 conversations=len(preds),audio_seconds=sum(p['duration_seconds'] for p in preds),
                 wer_percent=100*(m['substitutions']+m['deletions']+m['insertions'])/n,
                 cer_percent=100*m['character_errors']/m['ref_chars'],
                 substitution_percent=100*m['substitutions']/n,deletion_percent=100*m['deletions']/n,
                 insertion_percent=100*m['insertions']/n,capped_calls=caps,total_calls=len(calls),
                 inference_wall_seconds=sum(p['wall_seconds'] for p in preds),**m)
        if (directory/'cloud_receipt.json').exists():
            row['function_resource_estimate_usd']=json.loads((directory/'cloud_receipt.json').read_text())['estimated_function_resource_usd']
        rows.append(row)
        for p in preds:
            details.append(dict(recipe=item['label'],conversation_id=p['conversation_id'],
                                wer_percent=100*p['metrics']['wer'],cer_percent=100*p['metrics']['cer']))
        print(json.dumps(dict(event='validated_decoder_input',effective_config=config,summary=summary,validation=valid)),flush=True)
    for name,data in [('comparison',rows),('per_conversation',details)]:
        (out/f'{name}.json').write_text(json.dumps(data,indent=2)+'\n')
        with (out/f'{name}.csv').open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(data[0]));writer.writeheader();writer.writerows(data)
    (out/'validation.json').write_text(json.dumps(validations,indent=2)+'\n')
    text=['# Controlled decoding results','',
          'Same three already-exposed Hindi conversations; 6,503 reference words. All source hashes, audio coverage, token termination and scores independently checked. No model training. These are development results, not population estimates.','',
          '| Recipe | WER % | CER % | Substitution % | Deletion % | Insertion % | Capped generation calls |',
          '|---|---:|---:|---:|---:|---:|---:|']
    for r in rows:
        text.append(f"| {r['recipe']} | {r['wer_percent']:.2f} | {r['cer_percent']:.2f} | {r['substitution_percent']:.2f} | {r['deletion_percent']:.2f} | {r['insertion_percent']:.2f} | {r['capped_calls']}/{r['total_calls']} |")
    text += ['', 'WER is corpus-level word errors divided by reference words. Capped generations remain in scores. Each native generation call can yield several timestamp segments; count underlying calls, not output segments. Fixed windows retain standard zero-padding to the Whisper encoder length.', '',
             'The native recipe uses batch1, timestamp-based sequential segmentation and no previous-token conditioning or temperature fallback; fixed recipes use batch8. This compares inference recipes, not a one-variable native-decoder ablation. Only the fixed15s versus fixed30s comparison changes one decoding setting. Runtime is descriptive and includes different batching. No significance claim or confidence interval is made from three conversations.', '',
             'Native coverage is verified from actual input-frame windows and advancing seek offsets. Timestamp segment endpoints alone are not used to infer audio coverage. All text is scored without repetition cleanup, transliteration or numeral conversion.', '',
             'The run configuration freezes each input artifact hash. Raw predictions, checkpoints identified by immutable revisions and weight hashes, native traces, resource receipts and provider billing evidence remain available in the surrounding evidence package. Resource receipts are estimates, not final provider bills.']
    (out/'RESULTS.md').write_text('\n'.join(text)+'\n')
    print(json.dumps(dict(event='final_decoder_comparison',rows=rows,all_valid=True)),flush=True)
    return 0
