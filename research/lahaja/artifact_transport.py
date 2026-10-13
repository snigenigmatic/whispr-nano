"""Lossless, hash-checked result transport through ORX's persisted job logs."""
import base64
import hashlib
import json
import zlib
from pathlib import Path


def emit(out):
    files = {p.name: base64.b64encode(p.read_bytes()).decode()
             for p in sorted(Path(out).iterdir()) if p.is_file()}
    raw = json.dumps(files, sort_keys=True).encode()
    payload = base64.b64encode(zlib.compress(raw)).decode()
    chunks = [payload[i:i+4000] for i in range(0, len(payload), 4000)]
    digest = hashlib.sha256(raw).hexdigest()
    for i, chunk in enumerate(chunks):
        print(json.dumps(dict(event='artifact_bundle', sha256=digest,
                              index=i, count=len(chunks), payload=chunk)), flush=True)


def recover(log, out):
    records = {}
    for line in Path(log).read_text().splitlines():
        start = line.find('{"event": "artifact_bundle"')
        if start < 0:
            continue
        row = json.loads(line[start:])
        if row['index'] in records and records[row['index']] != row:
            raise ValueError('Conflicting artifact chunks')
        records[row['index']] = row
    if not records:
        raise ValueError('No artifact bundle found')
    first = records[min(records)]
    if set(records) != set(range(first['count'])):
        raise ValueError('Incomplete artifact bundle')
    if any(r['sha256'] != first['sha256'] or r['count'] != first['count'] for r in records.values()):
        raise ValueError('Mixed artifact bundles')
    raw = zlib.decompress(base64.b64decode(''.join(records[i]['payload'] for i in range(first['count'])), validate=True))
    if hashlib.sha256(raw).hexdigest() != first['sha256']:
        raise ValueError('Artifact bundle hash mismatch')
    files = json.loads(raw)
    if any(Path(name).name != name or name in {'', '.', '..'} for name in files):
        raise ValueError('Invalid artifact filename')
    decoded = {name: base64.b64decode(data, validate=True) for name, data in files.items()}
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    for name, data in decoded.items():
        (out/name).write_bytes(data)
    return dict(sha256=first['sha256'], files=sorted(files), chunks=first['count'])
