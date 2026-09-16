"""Read-only cache warming for this pilot's existing GPU containers.

Reads only frozen scheduled training exposures and their declared development
inputs. A separate Python process reconstructs the index order; it never calls a
model, alters the trainer RNG, writes remote files, or starts a new container.
"""
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import re
import subprocess
import time

ROOT = Path(__file__).resolve().parents[2]
P = ROOT / 'research/pilot'

REMOTE = r'''
import hashlib,json,sys,time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from research.pilot.training import IndexCycler,canonical_hash
plans=json.loads(sys.argv[1]);wait_seconds=float(sys.argv[2]);start=time.monotonic()
def sha(path):
    digest=hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''):digest.update(chunk)
    return digest.hexdigest()
def main():
    # Model download/loading can precede publication of its immutable manifest.
    # Wait for both files; an early container discovery is not a completed attempt.
    while True:
        base=Path('/outputs/training')
        folders=[] if not base.exists() else [p for p in base.iterdir()
            if p.name in plans and (p/'provenance.json').is_file()
            and (p/'manifest.json').is_file() and not (p/'summary.json').exists()]
        if len(folders)>1:
            raise ValueError(f'Ambiguous active registered runs: {[p.name for p in folders]}')
        if folders:break
        if time.monotonic()-start>=wait_seconds:
            print(json.dumps({'status':'prefetch_not_ready','reason':'No unique active registered manifest before wait deadline'}),flush=True)
            return 75
        time.sleep(3)
    folder=folders[0];plan=plans[folder.name]
    provenance=json.loads((folder/'provenance.json').read_text());cfg=provenance['config']
    rows=json.loads((folder/'manifest.json').read_text())
    for key,value in plan['config'].items():
        if cfg.get(key)!=value:raise ValueError(f'Registered config mismatch: {key}')
    if provenance['training_code_sha256']!=plan['training_code_sha256']:
        raise ValueError('Training implementation differs from the frozen contract')
    if provenance['feature_signature']!=plan['feature_signature']:
        raise ValueError('Feature signature differs from the frozen cache receipt')
    if any(r['split'] not in ('train','dev') for r in rows):raise ValueError('Test access forbidden')
    if len({r['utt_id'] for r in rows})!=len(rows):raise ValueError('Duplicate manifest IDs')
    train=sorted([r for r in rows if r['split']=='train'],key=lambda r:r['utt_id'])
    dev=sorted([r for r in rows if r['split']=='dev'],key=lambda r:r['utt_id'])
    if canonical_hash(train)!=plan['train_manifest_sha256'] or canonical_hash(dev)!=plan['dev_manifest_sha256']:
        raise ValueError('Audio IDs/references differ from the frozen selected train/dev manifests')
    if provenance['initial_state_sha256']!=plan['initial_state_sha256']:
        raise ValueError('Initial state differs from the registered model')
    indices=IndexCycler(len(train),cfg['seed']).take(cfg['num_updates']*cfg['batch_size']*cfg['grad_accum_steps'])
    needed={train[i]['utt_id']:train[i] for i in indices}
    needed.update({r['utt_id']:r for r in dev})
    cache_root=Path(cfg['feature_cache_dir']).resolve()
    if not cache_root.is_relative_to(Path('/data/feature_cache').resolve()):
        raise ValueError('Feature path escapes the existing registered cache')
    audio_root=Path('/data/l2arctic').resolve()
    before={name:sha(folder/name) for name in ('provenance.json','manifest.json')}
    def warm(row):
        audio=(audio_root/row['audio_path']).resolve()
        if not audio.is_relative_to(audio_root):raise ValueError('Audio path escapes the corpus')
        cache=cache_root/(canonical_hash([sha(audio),provenance['feature_signature']])+'.npy')
        with cache.open('rb') as stream:
            for chunk in iter(lambda:stream.read(1024*1024),b''):pass
        return row['utt_id']
    with ThreadPoolExecutor(max_workers=24) as pool:
        ids=list(pool.map(warm,needed.values()))
    if before!={name:sha(folder/name) for name in before}:
        raise ValueError('Manifest/provenance changed during prefetch')
    print(json.dumps({'run':folder.name,'status':'read_only_prefetch_complete','rows':len(ids),
        'seconds':time.monotonic()-start,'test_access':False,'remote_file_writes':False,
        'exposure_ids_sha256':canonical_hash(sorted(ids)),
        'train_manifest_sha256':plan['train_manifest_sha256'],'dev_manifest_sha256':plan['dev_manifest_sha256']}),flush=True)
    return 0
sys.exit(main())
'''


def registered_plans():
    from research.pilot import loop
    from research.pilot.training import canonical_hash, validate_rows
    # Verify every immutable input/code artifact before reconstructing exposures.
    loop.status(P / 'ledger')
    contract = loop.read_json(P / 'ledger/contract.json')
    manifest = [json.loads(line) for line in (P / 'data/manifest.jsonl').read_text().splitlines()]
    dev = [row for row in manifest if row['split'] == 'dev']
    feature_signature = loop.read_json(P / 'feature_cache_receipt.json')['feature_signature']
    plans = {}
    for name, job in contract['jobs'].items():
        if '-seed' not in name:
            continue
        cfg = job['parameters']
        selected = [json.loads(line) for line in (P / 'selections' / (cfg['arm_name'] + '.jsonl')).read_text().splitlines()]
        train, development = validate_rows(selected + dev)
        plans[name] = {'config': cfg, 'train_manifest_sha256': canonical_hash(train),
                       'dev_manifest_sha256': canonical_hash(development),
                       'initial_state_sha256': cfg['expected_initial_state_sha256'],
                       'training_code_sha256': contract['artifacts']['training']['sha256'],
                       'feature_signature': feature_signature}
    if len(plans) != 9:
        raise ValueError('Expected the existing nine registered runs only')
    return plans


def completion(record):
    if record.get('returncode') != 0:
        return False
    for line in record.get('stdout', '').splitlines():
        try:
            payload = json.loads(line)
        except (ValueError, TypeError):
            continue
        if payload.get('status') == 'read_only_prefetch_complete' and payload.get('test_access') is False:
            return True
    return False


def completed_containers(path=None):
    path = path or P / 'prefetch_receipts.log'
    done = set()
    if Path(path).exists():
        for line in Path(path).read_text().splitlines():
            try:
                record = json.loads(line)
            except (ValueError, TypeError):
                continue  # A currently appended final line may be incomplete.
            if completion(record):
                done.add(record['container'])
    return done


def warm(container, plans, deadline):
    remaining = max(1, int(deadline - time.time()))
    # -B prevents incidental __pycache__ writes in the separate helper process.
    result = subprocess.run([str(ROOT / '.venv/bin/modal'), 'container', 'exec', container,
                             '--', 'python', '-B', '-u', '-c', REMOTE,
                             json.dumps(plans, separators=(',', ':')), str(min(180, remaining))],
                            capture_output=True, text=True, timeout=min(900, remaining + 10))
    return dict(container=container, returncode=result.returncode, stdout=result.stdout, stderr=result.stderr)


def main():
    deadline = json.loads((P / 'deadline.json').read_text())['experiment_cutoff_epoch']
    plans = registered_plans()
    done, attempts, futures, pending = completed_containers(), {}, {}, set()
    print(json.dumps({'status': 'prefetch_monitor_started', 'already_completed_containers': sorted(done),
                      'registered_runs': len(plans), 'model_calls': 0}), flush=True)
    with ThreadPoolExecutor(max_workers=3) as executor:
        while time.time() < deadline and len(done) < 9:
            apps = set()
            for log in P.glob('train-seed[012].log'):
                apps.update(re.findall(r'https://modal.com/apps/snigenigmatic/main/(ap-[A-Za-z0-9]+)', log.read_text()))
            try:
                containers = json.loads(subprocess.check_output(
                    [str(ROOT / '.venv/bin/modal'), 'container', 'list', '--json'], text=True, timeout=30))
            except (subprocess.SubprocessError, ValueError) as error:
                print(json.dumps({'status': 'prefetch_discovery_retry', 'error': str(error)}), flush=True)
                containers = []
            for item in containers:
                container = item['container_id']
                if item['app_id'] in apps and container not in done | pending and attempts.get(container, 0) < 3:
                    pending.add(container)
                    attempts[container] = attempts.get(container, 0) + 1
                    futures[executor.submit(warm, container, plans, deadline)] = container
            for future in list(futures):
                if future.done():
                    container = futures.pop(future)
                    pending.remove(container)
                    try:
                        record = future.result()
                    except Exception as error:
                        record = {'container': container, 'error': str(error)}
                    record['attempt'] = attempts[container]
                    print(json.dumps(record), flush=True)
                    if completion(record):
                        done.add(container)
                    elif 'ValueError:' in (record.get('stdout', '') + record.get('stderr', '')):
                        # Identity violations are not transient startup problems.
                        attempts[container] = 3
            time.sleep(min(10, max(0, deadline - time.time())))
        for future, container in futures.items():
            try:
                print(json.dumps(future.result()), flush=True)
            except Exception as error:
                print(json.dumps({'container': container, 'error': str(error)}), flush=True)


if __name__ == '__main__':
    main()
