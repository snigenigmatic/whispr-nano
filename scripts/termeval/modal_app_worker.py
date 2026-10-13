"""Importable Modal worker; source code works across Python 3.11 and 3.12."""


def run_committed_study(archive_sha):
    import hashlib
    from pathlib import Path
    import subprocess
    import tarfile
    import time

    started=time.monotonic();root=Path('/tmp/study');root.mkdir()
    source=Path('/opt/study-source.tar')
    if hashlib.sha256(source.read_bytes()).hexdigest()!=archive_sha:
        raise ValueError('Code archive hash mismatch')
    with tarfile.open(source) as tf:tf.extractall(root,filter='data')
    proc=subprocess.run(['bash','experiments/cloud_entrypoint.sh'],cwd=root,text=True,capture_output=True)
    return dict(returncode=proc.returncode,stdout=proc.stdout,stderr=proc.stderr,
                worker_seconds=time.monotonic()-started)
