"""User-authorized extension cutoff; stop only this H100 study."""
import json,os,re,subprocess,time
from pathlib import Path
H=Path(__file__).resolve().parent
ROOT=H.parents[1]
deadline=json.loads((H/'deadline.json').read_text())
(H/'timer.pid').write_text(str(os.getpid()))
print(json.dumps(deadline),flush=True)
while time.time()<deadline['experiment_cutoff_epoch']:
    time.sleep(min(20,deadline['experiment_cutoff_epoch']-time.time()))
apps=set(re.findall(r'https://modal.com/apps/snigenigmatic/main/(ap-[A-Za-z0-9]+)',(H/'run.log').read_text()))
receipts=[]
for app in sorted(apps):
    r=subprocess.run([str(ROOT/'.venv/bin/modal'),'app','stop',app,'--yes'],capture_output=True,text=True,timeout=30)
    receipts.append({'app':app,'returncode':r.returncode,'output':r.stdout+r.stderr})
(H/'cutoff.json').write_text(json.dumps({'time':time.time(),'apps':receipts},indent=2)+'\n')
