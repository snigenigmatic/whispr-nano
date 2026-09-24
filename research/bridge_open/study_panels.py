"""Local-only diagnostic panels on validated, frozen ASR predictions.

These exploratory analyses do not implement Humyn's unpublished metrics.
Oracle selection uses evaluation references and is never deployment evidence.
"""
import csv
import hashlib
import json
import math
from itertools import combinations
from pathlib import Path

from research.bridge_open.analyze_benchmark import analyze
from research.bridge_open.validate_run import validate

ROOT = Path(__file__).resolve().parents[2]


def aggregate_errors(rows):
    if not rows:
        raise ValueError('Cannot aggregate an empty cohort')
    keys = ['hits', 'substitutions', 'deletions', 'insertions', 'ref_words', 'hyp_words']
    totals = {k: sum(r['metrics'][k] for r in rows) for k in keys}
    h, s, d, i, n, m = (totals[k] for k in keys)
    if n <= 0 or h + s + d != n or h + s + i != m:
        raise ValueError('Inconsistent word alignment counts')
    return {**totals, 'count': len(rows), 'micro_wer': (s+d+i)/n,
            'substitution_rate': s/n, 'deletion_rate': d/n, 'insertion_rate': i/n,
            'hyp_ref_length_ratio': m/n, 'pooled_wil': 1-h*h/(n*m) if m else 1.0,
            'mean_conversation_wil': sum(r['metrics']['wil'] for r in rows)/len(rows)}


def oracle_panel(models):
    """Compare selection levels on one exact common cohort and word denominator."""
    if not models or any(not rows for rows in models.values()):
        raise ValueError('Empty model cohort')
    maps = {name: {r['conversation_id']: r for r in rows} for name, rows in models.items()}
    ids = set(next(iter(maps.values())))
    for name, rows in models.items():
        if len(maps[name]) != len(rows) or set(maps[name]) != ids:
            raise ValueError('Oracle requires identical unique conversation sets')
    first = next(iter(maps.values()))
    for cid in ids:
        expected = (first[cid]['language'], first[cid]['metrics']['ref_words'])
        if any((r[cid]['language'], r[cid]['metrics']['ref_words']) != expected for r in maps.values()):
            raise ValueError('Oracle reference denominators or languages disagree')
    ids = sorted(ids)
    n = sum(first[c]['metrics']['ref_words'] for c in ids)
    def errors(row):
        return sum(row['metrics'][k] for k in ['substitutions', 'deletions', 'insertions'])
    total_errors = {name: sum(errors(rows[c]) for c in ids) for name, rows in maps.items()}
    best = min(total_errors, key=lambda name: (total_errors[name], name))
    per_language, language_errors = {}, 0
    for lang in sorted({first[c]['language'] for c in ids}):
        group = [c for c in ids if first[c]['language'] == lang]
        selected = min(maps, key=lambda name: (sum(errors(maps[name][c]) for c in group), name))
        per_language[lang] = selected
        language_errors += sum(errors(maps[selected][c]) for c in group)
    selected_by_file = {c: min(maps, key=lambda name: (errors(maps[name][c]), name)) for c in ids}
    file_errors = sum(errors(maps[selected_by_file[c]][c]) for c in ids)
    return dict(scope='in-sample reference oracle; optimistic lower bounds, not a learned router',
                conversation_ids=ids, ref_words=n, fixed_model=best,
                fixed_model_micro_wer=total_errors[best]/n,
                language_oracle_micro_wer=language_errors/n,
                conversation_oracle_micro_wer=file_errors/n,
                selected_by_language=per_language, selected_by_conversation=selected_by_file,
                all_fixed_model_micro_wer={name: e/n for name,e in total_errors.items()})


def correlation_panel(models):
    result = []
    for a,b in combinations(models, 2):
        left = {r['conversation_id']:r['metrics']['wer'] for r in models[a]}
        right = {r['conversation_id']:r['metrics']['wer'] for r in models[b]}
        ids = sorted(set(left)&set(right)); value = None
        reason = 'fewer than five conversations' if len(ids)<5 else None
        if reason is None:
            x, y = [left[c] for c in ids], [right[c] for c in ids]
            mx, my = sum(x)/len(x), sum(y)/len(y)
            denom = math.sqrt(sum((v-mx)**2 for v in x)*sum((v-my)**2 for v in y))
            if denom:
                value = sum((u-mx)*(v-my) for u,v in zip(x,y))/denom
            else:
                reason = 'constant scores'
        result.append(dict(a=a,b=b,conversation_ids=ids,pearson_wer=value,suppression_reason=reason))
    return result


def main():
    cfg = json.loads((ROOT/'research/bridge_open/run_config.json').read_text())
    assert cfg['executor'] == 'study_panels'
    print(json.dumps(dict(event='effective_config',config=cfg)),flush=True)
    directories=[]
    for item in cfg['input_runs']:
        directory=ROOT/item['path']
        for name,expected in item['sha256'].items():
            if hashlib.sha256((directory/name).read_bytes()).hexdigest()!=expected:
                raise ValueError(f'Frozen evidence changed: {directory/name}')
        check=validate(directory)
        if not check['valid']: raise ValueError(check)
        directories.append(directory)
    evidence=analyze(directories)
    runs={r['model_id']:r for r in evidence['runs']}
    if len(runs)!=len(evidence['runs']): raise ValueError('Ambiguous multiple recipes per model')
    per_language=[]
    for model,run in runs.items():
        for lang in sorted(run['summary']['languages']):
            rows=[r for r in run['conversations'] if r['language']==lang]
            per_language.append(dict(model_id=model,language=lang,**aggregate_errors(rows)))
    broad=['openai/whisper-small','openai/whisper-large-v3-turbo','facebook/omniASR-CTC-1B']
    cohorts={}
    for name,model_ids,language in [('hindi_development',list(runs),'hi'),('shared_nine_languages',broad,None)]:
        eligible={m:{r['conversation_id']:r for r in runs[m]['conversations']
                     if language is None or r['language']==language} for m in model_ids}
        ids=sorted(set.intersection(*(set(rows) for rows in eligible.values())))
        models={m:[eligible[m][c] for c in ids] for m in model_ids}
        cohorts[name]=dict(models=model_ids,conversation_ids=ids,
                          errors={m:aggregate_errors(rows) for m,rows in models.items()},
                          oracle=oracle_panel(models),correlations=correlation_panel(models),
                          per_file_range=[dict(conversation_id=c,language=eligible[model_ids[0]][c]['language'],
                              min_wer=min(eligible[m][c]['metrics']['wer'] for m in model_ids),
                              max_wer=max(eligible[m][c]['metrics']['wer'] for m in model_ids)) for c in ids])
    result=dict(contract='open_study_panels_v1',scope='post-hoc descriptive public-corpus diagnostics',
                limitations=['not BRIDGE-matched metrics','oracle uses evaluation references',
                             'Pearson is pooled, descriptive and not language-adjusted',
                             'small cohorts and unknown cross-file speaker overlap',
                             'high deletion is an alignment finding, not proof of acoustic truncation'],
                per_language=per_language,cohorts=cohorts,cloud_calls=0)
    out=ROOT/'research/bridge_open/artifacts/study_panels'
    out.mkdir(parents=True,exist_ok=False)
    (out/'panels.json').write_text(json.dumps(result,indent=2)+'\n')
    with (out/'error_profiles.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(per_language[0]));writer.writeheader();writer.writerows(per_language)
    print(json.dumps(dict(event='final_study_panels',**result)),flush=True)
    return 0
