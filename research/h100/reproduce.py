"""Portable offline verification of all nine H100 runs and the fixed decision."""
import json
from pathlib import Path
import statistics
from research.pilot.evaluate_runs import evaluate,compare,sha,evidence_markdown

ROOT=Path(__file__).resolve().parents[2]
H=ROOT/'research/h100'
P=ROOT/'research/pilot'


def main():
    original=json.loads((H/'results/audit.json').read_text())
    schedule=json.loads((H/'schedule.json').read_text())
    if original['source_sha256']['evaluator']!=sha(P/'evaluate_runs.py'):raise ValueError('Evaluator changed')
    dirs=[H/'runs'/f"{r['arm_name']}-seed{r['seed']}" for r in schedule['runs']]
    audit=evaluate(dirs,P/'data/manifest.jsonl',P/'selections',schedule)
    compare(audit['status'],'valid_complete','All nine runs required')
    compare(audit['paired_contrasts'],original['paired_contrasts'],'Published contrasts')
    expected={(r['arm_name'],r['seed']):r for r in original['runs']}
    for run in audit['runs']:
        if 'H100' not in run['runtime']['device_name']:raise ValueError('Mixed hardware')
        for field in ('metrics','exposure','artifact_sha256','initial_state_sha256','dev_manifest_sha256','runtime'):
            compare(run[field],expected[(run['arm_name'],run['seed'])][field],field)
    e=[r['filtered_minus_random_macro_wer_pp'] for r in audit['paired_contrasts'] if r['normalization']=='english']
    m=[r['filtered_minus_random_macro_wer_pp'] for r in audit['paired_contrasts'] if r['normalization']=='minimal']
    d=json.loads((H/'results/decision.json').read_text())
    compare(d['mean_english_contrast_pp'],statistics.mean(e),'Mean primary contrast')
    compare(d['mean_minimal_contrast_pp'],statistics.mean(m),'Mean sensitivity contrast')
    compare(d['pilot_positive'],statistics.mean(e)>=.5 and sum(x>0 for x in e)>=2 and statistics.mean(m)>=0,'Registered decision')
    target=ROOT/'tmp/h100-reproduction';target.mkdir(parents=True,exist_ok=True)
    (target/'audit.json').write_text(json.dumps(audit,indent=2)+'\n')
    (target/'evidence.md').write_text(evidence_markdown(audit))
    print('Verified all9 H100 runs, every published metric/exposure and the registered decision. No inference, network, or paid compute.')


if __name__=='__main__':main()
