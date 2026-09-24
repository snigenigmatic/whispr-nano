"""Finish this already-dispatched batch without launching or retrying compute.

Collect immutable job evidence, reject incomplete cohorts, and recompute the
complete comparison once both models finish. This is finite local orchestration.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from research.bridge_open.analyze_fleurs import analyze

ARTIFACTS = ROOT / 'research/bridge_open/artifacts'
REGISTRY = ARTIFACTS / 'fleurs_run_registry.json'
STATUS = ROOT / 'research/bridge_open/FLEURS_BATCH_STATUS.md'
OUTPUT = ARTIFACTS / 'fleurs_whisper_full_test_v2'
MODELS = {'openai/whisper-large-v3-turbo', 'openai/whisper-large-v3'}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def status(message):
    now = datetime.now(timezone.utc).isoformat()
    STATUS.write_text(f'# FLEURS batch status\n\nUpdated: {now}\n\n{message}\n\n'
                      'See [all job links](FLEURS_JOBS.md). This finalizer never launches GPU jobs or retries inference.\n')
    print(json.dumps({'time': now, 'message': message}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--once', action='store_true')
    args = parser.parse_args()
    expected_runs = {e['run_id'] for e in json.loads(REGISTRY.read_text())
                     if e.get('phase') == 'official_test' and e.get('model_id') in MODELS}
    if len(expected_runs) != 16:
        raise ValueError('Expected the 16 already-dispatched official-test jobs')
    source_hashes = {str(p.relative_to(ROOT)): digest(p) for p in (
        Path(__file__), ROOT/'research/bridge_open/analyze_fleurs.py',
        ROOT/'research/bridge_open/validate_run.py', ROOT/'research/bridge_open/scoring.py')}
    deadline = time.monotonic() + 2400
    db = sqlite3.connect(Path.home()/'.local/share/openresearch/orx.db')
    while True:
        if any(digest(ROOT/p) != value for p, value in source_hashes.items()):
            raise RuntimeError('Analysis source changed during supervision; refusing mixed provenance')
        subprocess.run([sys.executable, str(ROOT/'research/bridge_open/collect_fleurs_runs.py')],
                       cwd=ROOT, check=True, stdout=subprocess.DEVNULL)
        selected = [e for e in json.loads(REGISTRY.read_text()) if e.get('run_id') in expected_runs]
        verified = sum(e['status'] == 'done_validated' for e in selected)
        states = {run: db.execute('select status from runs where id=?', (run,)).fetchone()[0]
                  for run in expected_runs}
        active = sum(value not in {'done', 'failed', 'cancelled'} for value in states.values())
        if verified == len(expected_runs):
            registry = ARTIFACTS/'fleurs_whisper_full_registry.json'
            registry.write_text(json.dumps(selected, indent=2)+'\n')
            manifest = ROOT/'data/bridge_open/fleurs_test_v1/corpus-v2.jsonl'
            if digest(manifest) != '967d8dca02f9f729c1ed7548dce7283cf9dfc62623815a7c131f71acb13ce2c7':
                raise ValueError('Frozen manifest changed')
            analyze(manifest, registry, OUTPUT)
            receipt = {'source_sha256': source_hashes, 'registry_sha256': digest(registry),
                       'manifest_sha256': digest(manifest), 'run_ids': sorted(expected_runs),
                       'note': 'Offline recomputation only. All 2,322 recordings required per model.'}
            (OUTPUT/'analysis_receipt.json').write_text(json.dumps(receipt, indent=2)+'\n')
            status('**Complete:** all 16 jobs have independently verified evidence. The two-model comparison contains all 2,322 recordings per model. '
                   'Read [the results](artifacts/fleurs_whisper_full_test_v2/RESULTS.md).')
            return 0
        if active == 0:
            failed = [e.get('cloud_job_id', e['run_id']) for e in selected if e['status'] != 'done_validated']
            status('**Needs review:** all provider jobs ended, but complete evidence did not validate for: '+', '.join(failed)+
                   '. No incomplete comparison was produced, and no GPU retry was launched.')
            return 2
        status(f'**Running:** {verified}/{len(expected_runs)} official-test jobs have independently verified evidence; '
               f'{active} provider jobs remain active. Complete results will be written only after all recordings validate.')
        if args.once:
            return 0
        if time.monotonic() > deadline:
            status('**Finalizer timed out:** provider state needs review. No comparison or GPU retry was produced.')
            return 2
        time.sleep(30)


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except Exception as exc:
        status(f'**Finalizer stopped:** {type(exc).__name__}: {exc}. No GPU retry was launched.')
        raise
