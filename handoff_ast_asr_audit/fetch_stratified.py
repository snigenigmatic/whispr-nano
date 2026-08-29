"""Fetch a family-stratified sample of Svarah clips as WAV bytes, no GPU needed."""
import modal

app = modal.App("svarah-stratified-fetch")
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("ffmpeg")
    .pip_install("datasets==4.4.1", "torchcodec==0.16.0", "soundfile==0.14.0", "numpy==2.5.2", "torch==2.13.0")
)
hf_secret = modal.Secret.from_name("huggingface")

SVARAH_LANGUAGE_FAMILIES = {
    "Assamese": "Indo-Aryan", "Bengali": "Indo-Aryan", "Bodo": "Sino-Tibetan",
    "Dogri": "Indo-Aryan", "Gujarati": "Indo-Aryan", "Hindi": "Indo-Aryan",
    "Kannada": "Dravidian", "Kashmiri": "Indo-Aryan", "Konkani": "Indo-Aryan",
    "Maithili": "Indo-Aryan", "Malayalam": "Dravidian", "Marathi": "Indo-Aryan",
    "Nepali": "Indo-Aryan", "Odia": "Indo-Aryan", "Punjabi": "Indo-Aryan",
    "Sindhi": "Indo-Aryan", "Tamil": "Dravidian", "Telugu": "Dravidian", "Urdu": "Indo-Aryan",
}


@app.function(image=image, secrets=[hf_secret], timeout=1200)
def fetch_stratified(per_family: int, seed: int) -> list[dict]:
    import io

    import numpy as np
    import soundfile as sf
    from datasets import Audio, load_dataset

    ds = load_dataset("ai4bharat/svarah", split="test")

    by_family: dict[str, list[int]] = {f: [] for f in ("Indo-Aryan", "Dravidian", "Sino-Tibetan")}
    for i, lang in enumerate(ds["primary_language"]):
        family = SVARAH_LANGUAGE_FAMILIES.get(str(lang).strip().title())
        if family:
            by_family[family].append(i)

    rng = np.random.default_rng(seed)
    chosen_idx = []
    for family, indices in by_family.items():
        n = min(per_family, len(indices))
        chosen_idx.extend(int(i) for i in rng.choice(indices, size=n, replace=False))
    chosen_idx.sort()

    ds_audio = ds.select(chosen_idx).cast_column("audio_filepath", Audio(sampling_rate=16000))
    out = []
    for row_idx, orig_idx in zip(range(len(chosen_idx)), chosen_idx):
        row = ds_audio[row_idx]
        audio = row["audio_filepath"]
        if hasattr(audio, "get_all_samples"):
            samples = audio.get_all_samples()
            arr = samples.data.mean(dim=0).numpy().astype(np.float32)
            sr = int(samples.sample_rate)
        else:
            arr = np.asarray(audio["array"], dtype=np.float32)
            sr = int(audio["sampling_rate"])
        buf = io.BytesIO()
        sf.write(buf, arr, sr, format="WAV")
        out.append({
            "index": orig_idx,
            "wav_bytes": buf.getvalue(),
            "primary_language": row["primary_language"],
            "text": row["text"],
            "gender": row.get("gender"),
            "duration_s": row.get("duration"),
        })
    return out


@app.local_entrypoint()
def main(per_family: int = 150, seed: int = 0):
    import json
    from pathlib import Path

    print(f"Fetching {per_family} clips per family (seed={seed})...")
    rows = fetch_stratified.remote(per_family=per_family, seed=seed)
    out_dir = Path("/home/ubuntu/scratch/saaras_audit")
    (out_dir / "wavs").mkdir(parents=True, exist_ok=True)
    manifest = []
    for row in rows:
        wav_path = out_dir / "wavs" / f"svarah_{row['index']:05d}.wav"
        wav_path.write_bytes(row["wav_bytes"])
        manifest.append({
            "index": row["index"],
            "wav_path": str(wav_path),
            "primary_language": row["primary_language"],
            "text": row["text"],
            "gender": row.get("gender"),
            "duration_s": row.get("duration_s"),
        })
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"Wrote {len(manifest)} clips to {out_dir}")
    from collections import Counter
    fam_map = SVARAH_LANGUAGE_FAMILIES
    fams = Counter(fam_map.get(str(m["primary_language"]).strip().title()) for m in manifest)
    print(f"Family counts: {dict(fams)}")
