"""Stage exact pinned public FLEURS bytes; never regenerate audio files."""
import hashlib
import json
import tarfile
from pathlib import Path

from huggingface_hub import hf_hub_download


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def stage(rows, reference_pack):
    refs = json.loads(Path(reference_pack).read_text())
    groups = {}
    for row in rows:
        assert row['source_repo'] == 'google/fleurs'
        groups.setdefault((row['source_repo'], row['source_revision'], row['source_shard'],
            row['source_archive_sha256']), []).append(row)
        text = refs[row['conversation_id']]
        assert hashlib.sha256(text.encode()).hexdigest() == row['reference_sha256']
        dest = Path(row['reference_path'])
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(text)
    for (repo, revision, shard, expected), group in groups.items():
        archive = Path(hf_hub_download(repo, shard, repo_type='dataset', revision=revision, token=False))
        assert digest(archive) == expected
        selected = {r['source_member']: r for r in group}
        assert len(selected) == len(group)
        seen = set()
        with tarfile.open(archive, 'r:gz') as tar:
            for member in tar:
                if member.name not in selected:
                    continue
                assert member.isfile() and member.name not in seen
                seen.add(member.name)
                row = selected[member.name]
                data = tar.extractfile(member).read()
                assert hashlib.sha256(data).hexdigest() == row['audio_sha256']
                dest = Path(row['audio_path'])
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_bytes(data)
        assert seen == set(selected)
        print(json.dumps(dict(event='fleurs_sources_staged', source_shard=shard,
            archive_sha256=expected, selected_count=len(group))), flush=True)
