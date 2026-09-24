"""Exact-pair replay of the frozen 22-conversation native decoder expansion."""
import csv
import hashlib
import json
import statistics
from pathlib import Path
from research.bridge_open.validate_run import validate
ROOT=Path(__file__).resolve().parents[2]
KEYS=['model_id','model_revision','condition_id','device','seed','chunk_seconds','batch_size','max_new_tokens','num_beams','return_timestamps','decoding_mode']


def load_group(items):
    predictions={};contract=None
    for item in items:
        path=ROOT/item['path']
        for name,digest in item['sha256'].items():
            if hashlib.sha256((path/name).read_bytes()).hexdigest()!=digest:raise ValueError('Frozen source changed')
        v=validate(path)
        if not v['valid']:raise ValueError(v)
        c=json.loads((path/'config.json').read_text());env=json.loads((path/'environment.json').read_text())
        current={k:c.get(k) for k in KEYS}
        current.update(weights=env['weight_sha256'],dtype=env['dtype'],scorer=env['code_sha256']['scoring.py'])
        if contract is None:contract=current
        if current!=contract:raise ValueError('Mixed recipe or weights within group')
        rows=[json.loads(l) for l in (path/'predictions.jsonl').read_text().splitlines()]
        for row in rows:
            cid=row['conversation_id']
            if cid in predictions:raise ValueError('Duplicate conversation shard')
            predictions[cid]=row
        print(json.dumps(dict(event='validated_native_expansion_input',config=c,validation=v,summary=json.loads((path/'summary.json').read_text()))),flush=True)
    return predictions


def call_stats(p):
    calls=[s for c in p['chunks'] for s in c['generation_sequences']]
    return len(calls),sum(50257 not in s for s in calls)


def aggregate(preds):
    metrics={k:sum(p['metrics'][k] for p in preds) for k in ['substitutions','deletions','insertions','ref_words','character_errors','ref_chars','hyp_words']}
    n=metrics['ref_words']
    return dict(conversations=len(preds),wer_percent=100*sum(metrics[k] for k in ['substitutions','deletions','insertions'])/n,
                cer_percent=100*metrics['character_errors']/metrics['ref_chars'],**metrics)


def main():
    cfg=json.loads((ROOT/'research/bridge_open/run_config.json').read_text())
    fixed=load_group(cfg['fixed_runs']);native=load_group(cfg['native_runs'])
    if set(fixed)!=set(native) or len(fixed)!=22:raise ValueError('Requires exact paired 22 cohort')
    out=ROOT/cfg['analysis_output'];out.mkdir(parents=True,exist_ok=False)
    paired=[]
    for cid in sorted(fixed):
        a,b=fixed[cid],native[cid]
        if any(a[k]!=b[k] for k in ['reference_sha256','audio_sha256','language','model_id','model_revision']):raise ValueError('Unpaired source')
        ac,ae=call_stats(a);bc,be=call_stats(b);m=a['metrics'];n=m['ref_words']
        if n!=b['metrics']['ref_words']:raise ValueError('Mismatched denominator')
        trace=b['chunks'][0]['native_trace'];hop=b['chunks'][0]['feature_hop_samples']
        r=dict(conversation_id=cid,language=a['language'],audio_seconds=a['duration_seconds'],
               fixed_wer=100*m['wer'],native_wer=100*b['metrics']['wer'],delta_wer=100*(b['metrics']['wer']-m['wer']),
               fixed_cer=100*m['cer'],native_cer=100*b['metrics']['cer'],delta_cer=100*(b['metrics']['cer']-m['cer']),
               fixed_capped_calls=ae,fixed_calls=ac,native_capped_calls=be,native_calls=bc,
               fixed_output_ref_ratio=m['hyp_words']/n,native_output_ref_ratio=b['metrics']['hyp_words']/n,
               native_final_seek=trace[-1]['next_seek_frame'],native_total_frames=b['chunks'][0]['total_feature_frames'],
               native_timestamp_segments=len(b['chunks'][0]['native_segments']),native_coverage_valid=True)
        for k in ['substitutions','deletions','insertions']:r['delta_'+k+'_per100']=100*(b['metrics'][k]-m[k])/n
        paired.append(r)
    languages=[]
    for lang in sorted({r['language'] for r in paired}):
        ids=[r['conversation_id'] for r in paired if r['language']==lang]
        a=aggregate([fixed[i] for i in ids]);b=aggregate([native[i] for i in ids])
        languages.append(dict(language=lang,conversations=len(ids),ref_words=a['ref_words'],fixed_wer=a['wer_percent'],native_wer=b['wer_percent'],delta_wer=b['wer_percent']-a['wer_percent'],fixed_cer=a['cer_percent'],native_cer=b['cer_percent'],delta_cer=b['cer_percent']-a['cer_percent']))
    deltas=[r['delta_wer'] for r in paired];quartiles=statistics.quantiles(deltas,n=4,method='inclusive')
    uncapped=[r for r in paired if r['fixed_capped_calls']==0 and r['native_capped_calls']==0]
    summary=dict(scope='descriptive paired development comparison; no population or demographic inference',conversation_count=22,languages=len(languages),
                 improved_conversations=sum(d<0 for d in deltas),worsened_conversations=sum(d>0 for d in deltas),
                 median_delta_wer=statistics.median(deltas),q25_delta_wer=quartiles[0],q75_delta_wer=quartiles[2],
                 fixed_micro=aggregate(list(fixed.values())),native_micro=aggregate(list(native.values())),
                 paired_uncapped_sensitivity=dict(count=len(uncapped),ids=[r['conversation_id'] for r in uncapped],median_delta_wer=statistics.median([r['delta_wer'] for r in uncapped]) if uncapped else None),
                 note='Uncapped subset is post-outcome selected and only a diagnostic; primary estimates retain every output.')
    for name,rows in [('per_conversation',paired),('per_language',languages)]:
        with (out/f'{name}.csv').open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    (out/'comparison.json').write_text(json.dumps(dict(summary=summary,per_language=languages,per_conversation=paired),indent=2)+'\n')
    lines=['# Native versus fixed-window Turbo across nine languages','',
           'All 22 matched conversations were fully processed and independently replayed against immutable source hashes. This is the same public development cohort used previously. No held-out or demographic claim is supported.','',
           '| Language | Conversations | Fixed 30 s WER % | Native WER % | Difference (points) | Fixed CER % | Native CER % |','|---|---:|---:|---:|---:|---:|---:|']
    for r in languages:lines.append(f"| {r['language']} | {r['conversations']} | {r['fixed_wer']:.2f} | {r['native_wer']:.2f} | {r['delta_wer']:+.2f} | {r['fixed_cer']:.2f} | {r['native_cer']:.2f} |")
    lines += ['',f"Native decoding improves WER on {summary['improved_conversations']}/22 conversations. Median paired change: {summary['median_delta_wer']:+.2f} points (descriptive IQR {quartiles[0]:+.2f} to {quartiles[2]:+.2f}).",'',
              'Negative differences favor native decoding. No confidence interval or significance claim: speaker independence is unverified and each language has few conversations. Cross-language pooled scores in JSON describe this exact file mixture only.','',
              'Both recipes use the same Turbo weights, float32, H100, beam 5 and scoring. Native uses timestamp-guided sequential decoding, batch 1, no previous-token context and no temperature fallback. Fixed uses independent 30 s audio windows, batch 8. This is a recipe comparison; batching and segmentation are not isolated. All capped generations stay in the primary scores.','',
              f"Uncapped-in-both diagnostic subset: {len(uncapped)}/22 conversations. This subset is selected on model outcomes and cannot replace the primary comparison.", '',
              'The per-conversation CSV includes S/D/I changes, output/reference length ratios, capped underlying calls and the native input-coverage trace summary. The independent validator checks every trace transition and final frame coverage; it does not infer coverage from predicted timestamp endpoints.']
    (out/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(event='final_native_expansion_comparison',summary=summary,per_language=languages)),flush=True)
    return 0
