"""CPU-only extraction of the pinned Svarah metadata through existing access."""
from pathlib import Path
import importlib.util
import json
import time
import modal

ROOT = Path(__file__).resolve().parents[2] if modal.is_local() else Path("/root")
app = modal.App("svarah-paper-metadata")
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("datasets==4.4.1", "pyarrow==25.0.1")
    .env({"HF_HOME": "/cache/huggingface", "HF_HUB_ENABLE_HF_TRANSFER": "0"})
    .add_local_file(ROOT / "paper/metadata/extract_svarah_metadata.py", "/root/extract_svarah_metadata.py")
)

@app.function(
    image=image, cpu=2, memory=8192, timeout=900, scaledown_window=2,
    max_containers=1,
    volumes={
        "/cache/huggingface": modal.Volume.from_name("ast-asr-cache"),
        "/legacy_data": modal.Volume.from_name("ast-asr-data"),
    },
    secrets=[modal.Secret.from_name("huggingface")],
)
def extract():
    from extract_svarah_metadata import extract_metadata
    start = time.monotonic()
    payload = extract_metadata()
    candidates = []
    for root in (Path("/legacy_data"), Path("/cache/huggingface")):
        for name in ("meta_speaker_stats.csv", "metadata.csv", "speaker_metadata.csv"):
            for path in root.rglob(name):
                candidates.append(str(path))
    payload["candidate_speaker_metadata_paths"] = candidates[:100]
    payload["wall_seconds"] = time.monotonic() - start
    return payload

@app.local_entrypoint()
def main():
    payload = extract.remote()
    records = payload.pop("records")
    output = ROOT / "paper/data/svarah_metadata_raw.jsonl"
    output.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in records))
    payload["num_rows"] = len(records)
    output.with_suffix(".provenance.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(f"Exported {len(records)} metadata records; no audio bytes. Runtime {payload['wall_seconds']:.1f}s.")
    print("Candidate original speaker metadata:", payload["candidate_speaker_metadata_paths"])
