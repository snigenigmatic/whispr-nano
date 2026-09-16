"""Preserve verified H100 output tensors without launching compute."""
import json
from pathlib import Path
import time
from concurrent.futures import ThreadPoolExecutor,as_completed
import zipfile
from research.pilot import preserve_checkpoints as helper
from research.pilot import loop

ROOT=Path(__file__).resolve().parents[2]
H=ROOT/'research/h100'
OUT=ROOT/'output/h100'
helper.OUTPUT=OUT/'checkpoints'


def main():
    OUT.mkdir(parents=True,exist_ok=True);helper.OUTPUT.mkdir(exist_ok=True)
    deadline=json.loads((H/'deadline.json').read_text())['deadline_epoch']-120
    expected=[f'{arm}-seed{seed}' for arm in ('full_gold','filtered10_gold','random_matched_gold') for seed in (0,1,2)]
    done={};errors={}
    while time.time()<deadline and len(done)<9:
        if not (H/'ledger').exists():time.sleep(5);continue
        status=loop.status(H/'ledger');sources=[]
        for name in expected:
            arm,seed=name.rsplit('-seed',1)
            if name in done or status['jobs'].get(arm)!='passed':continue
            summary_path=H/'runs'/name/'summary.json';summary=helper.read(summary_path)
            audit_path=H/f'{arm}.audit.json';audit=helper.read(audit_path)
            match=next(r for r in audit['runs'] if r['arm_name']==arm and r['seed']==int(seed))
            if helper.sha(summary_path)!=match['artifact_sha256']['summary.json']:raise ValueError('Audited summary changed')
            sources.append({'name':name,'arm_name':arm,'seed':int(seed),'summary_path':str(summary_path),
                            'summary_sha256':helper.sha(summary_path),'audit_path':str(audit_path),'audit_sha256':helper.sha(audit_path),
                            'expected_model_sha256':summary['checkpoint_model_sha256'],'model_id':summary['config']['model_id'],
                            'model_revision':summary['config']['model_revision'],'runtime':summary['runtime'],
                            'remote_directory':f'h100-v1/{name}/checkpoint_last'})
        with ThreadPoolExecutor(max_workers=3) as pool:
            futures={pool.submit(helper.download,s,deadline):s['name'] for s in sources}
            for future in as_completed(futures):
                name=futures[future]
                try:done[name]=future.result();print(json.dumps({'preserved':name}),flush=True)
                except Exception as error:errors[name]=str(error);print(json.dumps({'error':name,'detail':str(error)}),flush=True)
        helper.write(OUT/'model_index.json',{'status':'complete' if len(done)==9 else 'incomplete','completed_runs':len(done),'expected_runs':9,
            'models':done,'errors':errors,'new_compute_launched':False})
        if len(done)<9:time.sleep(10)
    (OUT/'CHECKPOINTS.md').write_text(f'# H100 checkpoints\n\n{len(done)}/9 final models verified against independently audited saved-weight SHA-256 values.\n\nEach directory contains model.safetensors and all saved processor/configuration files. Optimizer states remain in the Modal results volume under h100-v1/<arm>-seed<seed>/checkpoint_last/.\n')
    archive=OUT/'trained_checkpoints.zip'
    with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_STORED,allowZip64=True) as z:
        for path in sorted(helper.OUTPUT.rglob('*')):
            if path.is_file() and '.download-part' not in path.name:z.write(path,path.relative_to(OUT))
        for path in (OUT/'model_index.json',OUT/'CHECKPOINTS.md'):z.write(path,path.name)
    with zipfile.ZipFile(archive) as z:
        if z.testzip():raise ValueError('Checkpoint archive CRC failure')
    receipt={'path':str(archive),'bytes':archive.stat().st_size,'sha256':helper.sha(archive),'completed_runs':len(done),'crc_verified':True}
    helper.write(OUT/'checkpoint_archive_receipt.json',receipt);print(json.dumps(receipt),flush=True)


if __name__=='__main__':main()
