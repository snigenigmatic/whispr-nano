"""Recover immutable ORX result transport and executed source without relaunching."""
import argparse
import hashlib
import gzip
import json
import sqlite3
import tarfile
from pathlib import Path
from research.bridge_open.artifact_transport import recover


def collect(runid,out):
    out=Path(out);db=Path.home()/'.local/share/openresearch/orx.db'
    c=sqlite3.connect(f'file:{db}?mode=ro',uri=True);c.row_factory=sqlite3.Row
    r=dict(c.execute('select * from runs where id=?',(runid,)).fetchone())
    if r['status'] not in ['done','failed','cancelled']:raise ValueError('Run has not finished')
    b=json.loads(r['backend_json']);receipt=dict(run_id=runid,commit=r['commit_sha'],provider_status=r['status'],
        **{k:b.get(k) for k in ['jobId','namespace','url','sourceDigest','sourcePath','sourceSize']})
    source=Path(receipt['sourcePath'])
    assert hashlib.sha256(source.read_bytes()).hexdigest()==receipt['sourceDigest']
    if not out.exists():
        transport=recover(Path.home()/'.local/share/openresearch/run-logs'/f'{runid}.log',out)
        (out/'transport_receipt.json').write_text(json.dumps(transport,indent=2)+'\n')
    else:assert (out/'transport_receipt.json').exists(),'Refuse unverified existing output'
    # Keep the exact ORX source snapshot portable; gzip does not change tar bytes.
    archived=gzip.compress(source.read_bytes(),mtime=0)
    (out/'source_snapshot.tar.gz').write_bytes(archived)
    receipt['portable_archive']='source_snapshot.tar.gz'
    receipt['portable_archive_sha256']=hashlib.sha256(archived).hexdigest()
    (out/'orx_source_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    # Select individual regular files; never extract an arbitrary archive path.
    wanted={'experiments/audit.py':'executed_audit.py','experiments/diagnostic_v1.py':'diagnostic_v1.py',
        'experiments/hf_entrypoint.sh':'executed_entrypoint.sh','experiments/hf_run.py':'executed_hf_run.py',
        'research/bridge_open/run_qwen.py':'executed_run_qwen.py',
        'research/bridge_open/qwen_trace.py':'executed_qwen_trace.py',
        'research/bridge_open/scoring.py':'executed_scoring.py',
        'research/bridge_open/run_config.json':'committed_run_config.json',
        'experiments/test_qwen_trace.py':'executed_test_qwen_trace.py',
        'experiments/test_native_run.py':'executed_test_native_run.py',
        'research/bridge_open/validate_run.py':'executed_validate_run.py',
        'CORRECTION_PROTOCOL.md':'CORRECTION_PROTOCOL.md'}
    with tarfile.open(source,'r:*') as tar:
        for member in tar:
            name=member.name.removeprefix('./')
            if name in wanted:
                assert member.isfile()
                (out/wanted[name]).write_bytes(tar.extractfile(member).read())
    return receipt


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('run_id');p.add_argument('output');a=p.parse_args()
    print(json.dumps(collect(a.run_id,a.output),indent=2))
