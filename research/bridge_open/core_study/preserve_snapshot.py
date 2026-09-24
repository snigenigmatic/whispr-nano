"""Copy only the finite core-study evidence into the local evidence checkout.

This does not publish, push, commit, launch compute, or copy environment files.
Run after validation and compilation. Prior benchmark evidence remains in the
destination's earlier commits.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]


def preserve(destination: Path) -> dict:
    destination = destination.resolve()
    if destination == ROOT or not (destination / ".git").exists():
        raise ValueError("Destination must be a separate existing Git checkout")
    roots = [
        ROOT / "core_audit.tex", ROOT / "core_audit.pdf",
        ROOT / "research/bridge_open/CURRENT_STATUS.md",
        ROOT / "research/bridge_open/core_study",
        ROOT / "research/bridge_open/listening_review_core_en_v1",
        ROOT / "tests/fixtures/resource_suite_correct_mock.py",
        *sorted((ROOT / "tests").glob("test_core_*.py")),
        ROOT / "tests/test_resource_results_provenance.py",
        *sorted((ROOT / "research/bridge_open/artifacts").glob("core_*")),
    ]
    selected = set()
    for path in roots:
        if not path.exists():
            raise FileNotFoundError(path)
        selected.update(path.rglob("*") if path.is_dir() else [path])
    files = sorted(path for path in selected if path.is_file()
                   and "__pycache__" not in path.parts
                   and ".pytest_cache" not in path.parts
                   and path.name != "evidence_manifest.json")
    inventory = []
    for path in files:
        relative = path.relative_to(ROOT)
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        with path.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        with target.open("rb") as stream:
            if hashlib.file_digest(stream, "sha256").hexdigest() != digest:
                raise ValueError(f"Copy hash mismatch: {relative}")
        inventory.append({"path": str(relative), "sha256": digest, "bytes": path.stat().st_size})
    receipt = {
        "schema": "core-study-evidence-inventory/v1", "date": "2026-09-24",
        "scope": "Selected new core-study files; preceding accepted evidence remains in earlier commits",
        "exclusions": "This inventory excludes itself and original dataset audio outside executed source archives",
        "files": inventory,
    }
    relative_manifest = Path("research/bridge_open/artifacts/core_study_v1/evidence_manifest.json")
    for base in (ROOT, destination):
        (base / relative_manifest).write_text(json.dumps(receipt, indent=2) + "\n")
    return {"files": len(files), "bytes": sum(row["bytes"] for row in inventory), "destination": str(destination)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("destination", type=Path)
    print(json.dumps(preserve(parser.parse_args().destination), indent=2))
