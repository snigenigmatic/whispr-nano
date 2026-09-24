"""Reserve a bounded HF job, then launch it exclusively through ORX."""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
LEDGER = Path.home()/'.local/share/openresearch/whispr-bridge-open/hf-budget-20260922.json'
CAP = 50.0
RESERVE = 1.50


def reserve(path, key, commit):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.with_suffix('.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        ledger = json.loads(path.read_text()) if path.exists() else dict(provider='huggingface', cap_usd=CAP, reservations={})
        if ledger['provider'] != 'huggingface' or ledger['cap_usd'] != CAP:
            raise ValueError('HF ledger contract mismatch')
        amounts = [r['reserved_usd'] for r in ledger['reservations'].values()]
        if any(type(a) not in (float, int) or a != RESERVE for a in amounts):
            raise ValueError('Invalid reservation')
        if key in ledger['reservations'] or sum(amounts)+RESERVE > CAP:
            raise ValueError('Duplicate launch or HF budget exhausted')
        ledger['reservations'][key] = dict(reserved_usd=RESERVE, commit=commit,
            reserved_at=datetime.now(timezone.utc).isoformat(), flavor='a100-large',
            timeout_seconds=1800, price_per_minute_usd=0.041667,
            provider_bill_verified=False)
        fd, tmp = tempfile.mkstemp(dir=path.parent)
        with os.fdopen(fd, 'w') as f:
            json.dump(ledger, f, indent=2); f.write('\n'); f.flush(); os.fsync(f.fileno())
        os.replace(tmp, path)
        return ledger


def main():
    exp_id = sys.argv[1]
    def git(*args):
        return subprocess.check_output(['git', *args], cwd=ROOT, text=True).strip()
    if git('status', '--porcelain'):
        raise RuntimeError('Commit all code before launch')
    status = subprocess.check_output(['orx','exp','status',exp_id],text=True)
    branch = git('branch', '--show-current')
    if f'branch:   {branch}\n' not in status:
        raise RuntimeError('Experiment branch differs from prepared checkout')
    if 'command:  bash experiments/hf_entrypoint.sh\n' not in status:
        raise RuntimeError('Unexpected experiment execution contract')
    if '(running,' in status or '(queued,' in status:
        raise RuntimeError('Experiment already active')
    ledger = reserve(LEDGER, exp_id + ':' + git('rev-parse','HEAD'), git('rev-parse','HEAD'))
    print(json.dumps(dict(event='hf_budget_reserved', cap_usd=CAP,
        total_reserved_usd=sum(r['reserved_usd'] for r in ledger['reservations'].values()))), flush=True)
    # Failed/ambiguous dispatches retain their full reservation; no auto-retry.
    return subprocess.call(['orx','exp','run',exp_id,'--backend','hf','--flavor','a100-large',
        '--timeout','30m','--image','pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime'],cwd=ROOT)


if __name__ == '__main__':
    raise SystemExit(main())
