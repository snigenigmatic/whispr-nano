"""One-shot local deadline timer; stops only this pilot's logged Modal apps."""
import json
import os
from pathlib import Path
import re
import subprocess
import time

ROOT = Path(__file__).resolve().parents[2]
PILOT = ROOT / 'research/pilot'
deadline = json.loads((PILOT / 'deadline.json').read_text())
(PILOT / 'deadline_watchdog.pid').write_text(str(os.getpid()))
cutoff = deadline['experiment_cutoff_epoch']
print(f"Timer active. Compute cutoff: {deadline['experiment_cutoff']}; delivery: {deadline['delivery_deadline']}", flush=True)
while time.time() < cutoff:
    time.sleep(min(30, cutoff - time.time()))
apps = set()
for pattern in ('teacher-ready-*.log', 'train-seed*.log', 'features-*.log'):
    for log in PILOT.glob(pattern):
        apps.update(re.findall(r'https://modal.com/apps/snigenigmatic/main/(ap-[A-Za-z0-9]+)', log.read_text(errors='replace')))
records = []
for app_id in sorted(apps):
    result = subprocess.run([str(ROOT / '.venv/bin/modal'), 'app', 'stop', app_id],
                            capture_output=True, text=True, timeout=45)
    records.append({'app_id': app_id, 'returncode': result.returncode,
                    'output': (result.stdout + result.stderr)[-2000:]})
(PILOT / 'compute_cutoff.json').write_text(json.dumps({'stopped_at_epoch': time.time(), 'apps': records}, indent=2) + '\n')
print('Experiment cutoff reached. Preserve available evidence and finish the report.', flush=True)
