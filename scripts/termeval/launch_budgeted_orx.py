#!/usr/bin/env python3
"""Reserve the full timeout cost before launching an ORX Hugging Face job.

Reservations stay counted even for failed submissions until provider billing is
reconciled. The Modal rate is a conservative allowance, not a measured price.
Only GPU shapes whose charges are bounded by these rates are allowed here.
Modal launches are disabled here: ORX uses Sandboxes, whereas this project now
requires Modal Apps with bounded function lifetimes and no warm containers.
"""
import argparse
from datetime import datetime, timezone
import fcntl
import json
import math
from pathlib import Path
import subprocess
import uuid

CAP=20.0
RATES={('hf','a100-large'):2.5,('modal','h100'):8.0}
LEDGER=Path.home()/'.local/share/openresearch/cloud-budget-20261013.json'
ORX='/Users/prateekraj/.cargo/bin/orx'


def reservation(provider,flavor,minutes):
    if type(minutes) is not int or not 1<=minutes<=120: raise ValueError('Invalid timeout')
    return math.ceil((RATES[(provider,flavor)]*minutes/60+0.25)*100)/100


def main():
    p=argparse.ArgumentParser()
    p.add_argument('experiment');p.add_argument('--provider',choices=['hf','modal'],required=True)
    p.add_argument('--flavor',required=True);p.add_argument('--minutes',type=int,required=True)
    p.add_argument('--dry-run',action='store_true');a=p.parse_args()
    if a.provider=='modal':
        raise SystemExit('Modal Sandboxes are disabled for this project. Use a bounded Modal App; this ORX backend creates Sandboxes.')
    amount=reservation(a.provider,a.flavor,a.minutes)
    LEDGER.parent.mkdir(parents=True,exist_ok=True)
    with LEDGER.with_suffix('.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        data=json.loads(LEDGER.read_text()) if LEDGER.exists() else dict(schema=1,caps={'hf':CAP,'modal':CAP},reservations=[])
        assert data['caps']=={'hf':CAP,'modal':CAP}
        used=sum(r['reserved_usd'] for r in data['reservations'] if r['provider']==a.provider)
        if used+amount>CAP: raise SystemExit(f'Budget refused: {used}+{amount}>{CAP}')
        row=dict(id=str(uuid.uuid4()),experiment=a.experiment,provider=a.provider,flavor=a.flavor,
            timeout_minutes=a.minutes,reserved_usd=amount,created_at=datetime.now(timezone.utc).isoformat(),
            status='reserved',rate_usd_per_hour=RATES[(a.provider,a.flavor)])
        print(json.dumps(dict(reservation=row,provider_total_reserved=used+amount)),flush=True)
        if a.dry_run:return
        data['reservations'].append(row)
        tmp=LEDGER.with_suffix('.tmp');tmp.write_text(json.dumps(data,indent=2)+'\n');tmp.replace(LEDGER)
    command=[ORX,'exp','run',a.experiment,'--backend',a.provider,'--flavor',a.flavor,'--timeout',f'{a.minutes}m']
    result=subprocess.run(command,text=True,capture_output=True)
    # Keep a receipt without logging any account environment or credentials.
    receipt=LEDGER.parent/f"cloud-launch-{row['id']}.txt"
    receipt.write_text(result.stdout+'\n'+result.stderr)
    print(result.stdout,end='');print(result.stderr,end='')
    raise SystemExit(result.returncode)

if __name__=='__main__':main()
