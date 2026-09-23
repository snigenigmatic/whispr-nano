import ast, hashlib, json
from pathlib import Path

HERE = Path(__file__).parent
ROOT = HERE.parents[2]

def test_configs():
    manifest = ROOT / 'data/bridge_open/fleurs_test_v1/corpus-v2.jsonl'
    digest = hashlib.sha256(manifest.read_bytes()).hexdigest()
    rows = [json.loads(x) for x in manifest.read_text().splitlines()]
    for name, count in [('config_canary.json', 3), ('config_full.json', 2322)]:
        cfg = json.loads((HERE / name).read_text())
        assert cfg['manifest_sha256'] == digest
        assert len(cfg['conversation_ids']) == count
        assert len(set(cfg['conversation_ids'])) == count
        byid = {r['conversation_id']: r for r in rows}
        assert all(byid[i]['language'] in {'as', 'bn', 'hi'} for i in cfg['conversation_ids'])
        assert cfg['model_revision'] == 'e9b71b369c048e2c6b634d4c131061c34e441179'

def test_runner_contract():
    tree = ast.parse((HERE / 'run_indic_conformer.py').read_text())
    names = {n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
    assert {'load_adapter', 'load_rows', 'run'} <= names
    source = (HERE / 'run_indic_conformer.py').read_text()
    assert 'transcribe_batch' in source
    assert 'sampling_rate' not in source or '16000' in source

if __name__ == '__main__':
    test_configs(); test_runner_contract(); print('IndicConformer contract checks passed')
