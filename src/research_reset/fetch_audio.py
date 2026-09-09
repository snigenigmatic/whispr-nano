"""Download the 120 listening wavs from gated ai4bharat/svarah.

Requires HF_TOKEN or `huggingface-cli login`. Does not fetch the rest of
Svarah. Writes 16 kHz mono WAVs named by utt_id.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from .constants import DATA_DIR
from .io_utils import write_json


def fetch_audio(blind_json: Path, out_dir: Path) -> dict:
    import json

    import numpy as np
    import soundfile as sf
    from datasets import Audio, load_dataset

    items = json.loads(blind_json.read_text(encoding="utf-8"))
    indices = sorted({int(item["svarah_index"]) for item in items})
    ds = load_dataset("ai4bharat/svarah", split="test")
    ds = ds.cast_column("audio_filepath", Audio(sampling_rate=16_000))
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for item in items:
        idx = int(item["svarah_index"])
        row = ds[idx]
        audio = row["audio_filepath"]
        if hasattr(audio, "get_all_samples"):
            samples = audio.get_all_samples()
            arr = samples.data.mean(dim=0).numpy().astype(np.float32)
            sr = int(samples.sample_rate)
        else:
            arr = np.asarray(audio["array"], dtype=np.float32)
            sr = int(audio["sampling_rate"])
        path = out_dir / f"{item['utt_id']}.wav"
        sf.write(path, arr, sr)
        written.append({"utt_id": item["utt_id"], "path": str(path), "sr": sr})
    manifest = {"n": len(written), "out_dir": str(out_dir), "files": written}
    write_json(out_dir / "manifest.json", manifest)
    return manifest


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--blind", type=Path, default=DATA_DIR / "listening_set.blind.json")
    parser.add_argument("--out", type=Path, default=Path("listen/public/audio"))
    args = parser.parse_args(argv)
    if not os.environ.get("HF_TOKEN") and not (Path.home() / ".cache/huggingface/token").exists():
        raise SystemExit(
            "Svarah is gated. Set HF_TOKEN or run huggingface-cli login, then retry."
        )
    info = fetch_audio(args.blind, args.out)
    print(f"wrote {info['n']} wavs to {info['out_dir']}")


if __name__ == "__main__":
    main()
