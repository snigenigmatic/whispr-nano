"""Extract Svarah metadata without materializing embedded audio bytes.

This module intentionally imports no Modal code and starts no remote jobs.
Call ``extract_metadata()`` in a CPU worker with the existing authorized HF
token and cached dataset. The returned payload contains metadata only.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path


DATASET_ID = "ai4bharat/Svarah"
DATASET_REVISION = "ebbf7777fe771490696a3f7b007097606fa8c924"


def extract_metadata() -> dict:
    """Return pinned dataset metadata, reading the audio path column only."""
    from datasets import load_dataset
    import pyarrow.compute as pc

    dataset = load_dataset(DATASET_ID, revision=DATASET_REVISION, split="test")
    # Read the struct's path field directly from Arrow: iterating audio cells
    # would copy every waveform byte string even with Audio(decode=False).
    paths = pc.struct_field(dataset.data.column("audio_filepath"), "path").to_pylist()
    rows = dataset.remove_columns(["audio_filepath"]).to_list()
    if len(rows) != 6656 or len(paths) != len(rows):
        raise ValueError(f"Unexpected pinned dataset dimensions: {len(rows)} rows")
    records = []
    for index, (row, audio_path) in enumerate(zip(rows, paths)):
        records.append({
            "utt_id": f"svarah_{index:05d}",
            "dataset_index": index,
            "audio_path": audio_path,
            **row,
        })
    return {
        "dataset_id": DATASET_ID,
        "dataset_revision": DATASET_REVISION,
        "split": "test",
        "dataset_fingerprint": dataset._fingerprint,
        "features": dataset.features.to_dict(),
        "extracted_at_utc": datetime.now(timezone.utc).isoformat(),
        "audio_bytes_exported": False,
        "records": records,
    }


def write_payload(payload: dict, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    rows = payload["records"]
    output.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))
    provenance = {key: value for key, value in payload.items() if key != "records"}
    provenance["num_rows"] = len(rows)
    output.with_suffix(".provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("paper/data/svarah_metadata_raw.jsonl"))
    args = parser.parse_args()
    write_payload(extract_metadata(), args.output)
