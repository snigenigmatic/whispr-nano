"""Run one committed study through a bounded Modal App, never a Sandbox.

The command is fixed. Only tracked files in the requested Git revision travel
to the user's Modal workspace; HF authentication is injected as a Modal Secret.
The same private ledger counts old ORX reservations and these native App runs.
"""
import argparse
from datetime import datetime,timezone
import fcntl
import hashlib
import json
import math
from pathlib import Path
import shlex
import subprocess
import tempfile
import uuid
from launch_budgeted_orx import LEDGER,CAP

RATES={'L4':2.,'H100':8.}
IMAGE='pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime'


def function_options(gpu,minutes):
    if gpu not in RATES or type(minutes) is not int or not 1<=minutes<=60:
        raise ValueError('Unsupported GPU or timeout')
    return dict(gpu=gpu,cpu=(4,4),memory=(16384,16384),timeout=minutes*60,startup_timeout=90,
        min_containers=0,max_containers=1,buffer_containers=0,scaledown_window=2,
        retries=0,single_use_containers=True,serialized=True,include_source=False)


def reserve(experiment,commit,gpu,minutes,dry_run):
    options=function_options(gpu,minutes)
    amount=math.ceil((RATES[gpu]*minutes/60+.25)*100)/100
    LEDGER.parent.mkdir(parents=True,exist_ok=True)
    with LEDGER.with_suffix('.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        data=json.loads(LEDGER.read_text()) if LEDGER.exists() else dict(schema=1,caps={'hf':CAP,'modal':CAP},reservations=[])
        assert data['caps']=={'hf':CAP,'modal':CAP}
        used=sum(r['reserved_usd'] for r in data['reservations'] if r['provider']=='modal')
        if used+amount>CAP:raise SystemExit(f'Modal budget refused: {used}+{amount}>{CAP}')
        row=dict(id=str(uuid.uuid4()),experiment=experiment,commit=commit,provider='modal',
            runtime='App.function',flavor=gpu,timeout_minutes=minutes,reserved_usd=amount,
            rate_usd_per_hour=RATES[gpu],created_at=datetime.now(timezone.utc).isoformat(),status='reserved')
        print(json.dumps(dict(reservation=row,provider_total_reserved=used+amount)),flush=True)
        if not dry_run:
            data['reservations'].append(row);tmp=LEDGER.with_suffix('.tmp')
            tmp.write_text(json.dumps(data,indent=2)+'\n');tmp.replace(LEDGER)
    return row,options


def main():
    p=argparse.ArgumentParser();p.add_argument('experiment');p.add_argument('--repository',type=Path,required=True)
    p.add_argument('--revision',required=True);p.add_argument('--gpu',choices=RATES,required=True)
    p.add_argument('--minutes',type=int,required=True);p.add_argument('--dry-run',action='store_true');a=p.parse_args()
    commit=subprocess.check_output(['git','rev-parse','--verify',a.revision+'^{commit}'],cwd=a.repository,text=True).strip()
    # The study guard needs time to export before the provider kills the worker.
    config=json.loads(subprocess.check_output(['git','show',commit+':research/lahaja/cloud_study.json'],cwd=a.repository,text=True))
    if config['max_inference_seconds']>a.minutes*60-180:raise SystemExit('Inference guard needs at least 180 seconds of provider headroom')
    row,options=reserve(a.experiment,commit,a.gpu,a.minutes,a.dry_run)
    if a.dry_run:return
    import modal
    env_file=Path.home()/'.openresearch/env'
    token=next(shlex.split(l.split('=',1)[1])[0] for l in env_file.read_text().splitlines()
               if l.split('=',1)[0].strip()=='export HF_TOKEN')
    receipt_path=LEDGER.parent/f"modal-app-{row['id']}.json"
    log_path=LEDGER.parent/'run-logs'/f"modal-app-{row['id']}.log";log_path.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='whispr-app-source-') as td:
        archive=Path(td)/'source.tar'
        subprocess.run(['git','archive','--format=tar','--output',str(archive),commit],cwd=a.repository,check=True)
        archive_sha=hashlib.sha256(archive.read_bytes()).hexdigest()
        image=(modal.Image.from_registry(IMAGE)
            .pip_install('transformers==5.16.1','pyarrow==25.0.1','soundfile==0.14.0',
                'openai-whisper==20250625','jiwer==4.0.0','numpy==2.4.6','huggingface-hub==1.29.0')
            .add_local_file(archive,'/opt/study-source.tar',copy=True))
        app=modal.App('whispr-cloud-study-app')

        @app.function(image=image,secrets=[modal.Secret.from_dict({'HF_TOKEN':token})],**options)
        def worker():
            import hashlib,subprocess,tarfile,time
            from pathlib import Path
            started=time.monotonic();root=Path('/tmp/study');root.mkdir()
            source=Path('/opt/study-source.tar')
            if hashlib.sha256(source.read_bytes()).hexdigest()!=archive_sha:raise ValueError('Code archive hash mismatch')
            with tarfile.open(source) as tf:tf.extractall(root,filter='data')
            proc=subprocess.run(['bash','experiments/cloud_entrypoint.sh'],cwd=root,text=True,capture_output=True)
            return dict(returncode=proc.returncode,stdout=proc.stdout,stderr=proc.stderr,worker_seconds=time.monotonic()-started)

        receipt=dict(**row,source_archive_sha256=archive_sha,image=IMAGE,function_options=options,
                     launcher_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),provider_bill_verified=False)
        try:
            with modal.enable_output(),app.run(detach=False):
                receipt['app_id']=app.app_id;receipt['status']='running'
                receipt_path.write_text(json.dumps(receipt,indent=2)+'\n')
                result=worker.remote()
                log_path.write_text(result.pop('stdout')+'\n'+result.pop('stderr'))
                receipt.update(result,status='completed' if result['returncode']==0 else 'failed',log=str(log_path))
        except BaseException:
            receipt['status']='failed';raise
        finally:
            receipt['finished_at']=datetime.now(timezone.utc).isoformat()
            receipt_path.write_text(json.dumps(receipt,indent=2)+'\n')
        print(json.dumps(receipt,indent=2),flush=True)
        raise SystemExit(receipt['returncode'])


if __name__=='__main__':main()
