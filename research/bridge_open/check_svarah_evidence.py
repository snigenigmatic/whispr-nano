"""Adversarial acceptance checks, including Python optimization mode."""
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / 'research/bridge_open/artifacts/qwen_svarah_acceptance_v2'


def run():
    checks = []
    with tempfile.TemporaryDirectory(prefix='svarah-evidence-') as tmp:
        directory = Path(tmp)/'candidate'
        shutil.copytree(SOURCE, directory)
        mutations = (
            ('summary_scored', 'summary.json', lambda x: x.update(scored=0)),
            ('not_complete', 'orx_source_receipt.json', lambda x: x.update(provider_status='running')),
            ('archive_escape', 'orx_source_receipt.json', lambda x: x.update(portable_archive='../outside.tar.gz')),
            ('changed_runner', 'environment.json', lambda x: x['code_sha256'].update({'run_qwen.py':'0'*64})),
        )
        for name, filename, modify in mutations:
            path = directory/filename
            original = path.read_bytes()
            contents = json.loads(original)
            modify(contents)
            path.write_text(json.dumps(contents))
            result = subprocess.run([sys.executable, '-O', '-m', 'research.bridge_open.accept_svarah_runs', str(directory)],
                                    cwd=ROOT, text=True, capture_output=True)
            path.write_bytes(original)
            if result.returncode == 0 or 'ValueError:' not in result.stderr:
                raise RuntimeError(f'{name} was not rejected explicitly: {result.stderr[-500:]}')
            checks.append(dict(case=name, rejected=True, python_optimization=True,
                               reason=result.stderr.splitlines()[-1]))
    target=ROOT/'research/bridge_open/artifacts/svarah_evidence_checks_v1'
    target.mkdir(exist_ok=True)
    (target/'negative_checks.json').write_text(json.dumps(dict(valid=True,cases=checks),indent=2)+'\n')
    print(json.dumps(checks,indent=2))


if __name__=='__main__':run()
