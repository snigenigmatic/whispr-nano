"""Prove the offline acceptance gate rejects corrupted in-memory evidence."""
import copy
import json
from pathlib import Path
from unittest.mock import patch
import research.bridge_open.finish_qwen_comparison as finisher


def main():
    run=finisher.ART/'qwen_native_full_hi_v1'
    original=finisher.jsonl
    def missing(rows): rows.pop()
    def duplicate(rows): rows[-1]=copy.deepcopy(rows[0])
    def metric(rows): rows[0]['metrics']['wer']+=0.01
    def termination(rows):
        t=rows[0]['chunks'][0]['generation_trace'][0]
        t['cap_observed']=not t['cap_observed']
    checks={}
    for name, mutate in [('missing_prediction',missing),('duplicate_id',duplicate),('wrong_wer',metric),('false_cap_flag',termination)]:
        def changed(path):
            rows=original(path)
            if Path(path)==run/'predictions.jsonl':mutate(rows)
            return rows
        with patch.object(finisher,'jsonl',changed):
            try:finisher.validate_qwen(run)
            except AssertionError:checks[name]='rejected'
            else:raise AssertionError('Acceptance gate allowed '+name)
    out=finisher.ART/'fleurs_qwen_comparison_v1/negative_validation_checks.json'
    out.write_text(json.dumps(dict(scope='Intentional in-memory corruptions; original artifacts unchanged',checks=checks),indent=2)+'\n')
    print(json.dumps(checks))

if __name__=='__main__':main()
