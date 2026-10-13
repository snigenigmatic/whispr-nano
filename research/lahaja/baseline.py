"""Fixed, speaker-balanced Lahaja pilot for a mentor-facing baseline.

The first released Parquet shard is a feasibility sample, not the full corpus.
No test-derived prompt, training, or reference-aware model selection is used.
"""

from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import io
import json
import os
from pathlib import Path
import time
import unicodedata

import jiwer
import numpy as np
import pyarrow.parquet as pq
import soundfile as sf
import torch
import transformers
from huggingface_hub import hf_hub_download
from transformers import AutoProcessor, WhisperForConditionalGeneration


DATASET_ID = "ai4bharat/Lahaja"
DATASET_REVISION = "d4ffd2ecbdd933e37c917ddcf620eef159ceb3a7"
DEV_FILE = "data/test-00000-of-00004.parquet"
DATA_FILES = tuple(f"data/test-{i:05d}-of-00004.parquet" for i in (1, 2, 3))
MODEL_ID = "openai/whisper-small"
MODEL_REVISION = "973afd24965f72e36ca33b3055d56a652f456b4d"
SEED = 20261011
PER_SPEAKER_SCENARIO = 2
OUTPUT = Path.home() / ".local/share/openresearch/lahaja-small-beam5-cuda-heldout-405-v1"


def _token() -> str | None:
    if os.environ.get("HF_TOKEN"):
        return os.environ["HF_TOKEN"]
    path = Path.home() / ".openresearch/env"
    if path.exists():
        for raw in path.read_text().splitlines():
            line = raw.strip().removeprefix("export ")
            if line.startswith("HF_TOKEN="):
                return line.split("=", 1)[1].strip().strip('"').strip("'")
    return None


def _normalize(value: str) -> str:
    value = unicodedata.normalize("NFC", value).lower()
    return " ".join("".join(
        ch if unicodedata.category(ch)[0] in "LMN" else " " for ch in value
    ).split())


def _pick(rows: list[dict]) -> list[dict]:
    eligible = [r for r in rows if 2.0 <= r["duration"] <= 25.0]
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in eligible:
        groups[(row["sp_id"], row["scenario"])].append(row)
    chosen = []
    for key, group in sorted(groups.items()):
        group.sort(key=lambda r: hashlib.sha256(
            f"{SEED}|{r['data_file']}|{r['row_index']}|{r['fname']}".encode()
        ).hexdigest())
        chosen.extend(group[:PER_SPEAKER_SCENARIO])
    chosen.sort(key=lambda r: (r["data_file"], r["row_index"]))
    return chosen


def _metadata(paths: dict[str, Path], excluded_speakers: set[str]) -> tuple[list[dict], list[dict]]:
    columns = ["text", "verbatim", "normalized", "duration", "scenario",
               "fname", "native_language", "native_state", "native_district",
               "sp_id", "lang", "gender", "age_group"]
    rows = []
    for data_file, path in sorted(paths.items()):
        for idx, row in enumerate(pq.read_table(path, columns=columns).to_pylist()):
            if row["sp_id"] in excluded_speakers:
                continue
            row["data_file"] = data_file
            row["row_index"] = idx
            rows.append(row)
    if not rows or any(not r["sp_id"] or not r["verbatim"] for r in rows):
        raise ValueError("missing Lahaja speaker or verbatim reference")
    if any(r["lang"] != "hi" for r in rows):
        raise ValueError("this pilot expected Hindi-only Lahaja audio")
    for speaker in {r["sp_id"] for r in rows}:
        cohort = [r for r in rows if r["sp_id"] == speaker]
        if len({r["native_state"] for r in cohort}) != 1:
            raise ValueError(f"inconsistent state for speaker {speaker}")
    return rows, _pick(rows)


def _audio(paths: dict[str, Path], selected: set[tuple[str, int]]) -> dict[tuple[str, int], tuple[np.ndarray, int]]:
    got = {}
    for data_file, path in sorted(paths.items()):
        pf = pq.ParquetFile(path)
        offset = 0
        wanted_in_file = {idx for shard, idx in selected if shard == data_file}
        for rg in range(pf.metadata.num_row_groups):
            count = pf.metadata.row_group(rg).num_rows
            wanted = wanted_in_file & set(range(offset, offset + count))
            if wanted:
                values = pf.read_row_group(rg, columns=["audio_filepath"]).to_pylist()
                for idx in sorted(wanted):
                    raw = values[idx - offset]["audio_filepath"]["bytes"]
                    if not raw:
                        raise ValueError(f"missing embedded audio at {data_file} row {idx}")
                    audio, sr = sf.read(io.BytesIO(raw), dtype="float32", always_2d=True)
                    if sr != 16000 or audio.shape[1] != 1:
                        raise ValueError(f"unexpected Lahaja audio format at {data_file} row {idx}: {sr}, {audio.shape}")
                    if not np.isfinite(audio).all():
                        raise ValueError(f"nonfinite audio at {data_file} row {idx}")
                    got[(data_file, idx)] = (audio[:, 0], sr)
            offset += count
    if set(got) != selected:
        raise ValueError("selected audio rows not completely recovered")
    return got


def _pooled(rows: list[dict]) -> dict:
    n = sum(r["ref_words"] for r in rows)
    e = sum(r["word_errors"] for r in rows)
    return {"word_errors": e, "ref_words": n, "wer": e / n if n else None,
            "clips": len(rows), "speakers": len({r["sp_id"] for r in rows})}


def _speaker_interval(rows: list[dict], resamples: int = 4000) -> list[float]:
    speaker_counts = defaultdict(lambda: [0, 0])
    for row in rows:
        bucket = speaker_counts[row["sp_id"]]
        bucket[0] += row["word_errors"]
        bucket[1] += row["ref_words"]
    speakers = sorted(speaker_counts)
    rng = np.random.default_rng(SEED)
    samples = []
    for _ in range(resamples):
        drawn = rng.choice(speakers, size=len(speakers), replace=True)
        errors = sum(speaker_counts[s][0] for s in drawn)
        words = sum(speaker_counts[s][1] for s in drawn)
        samples.append(errors / words)
    return [float(x) for x in np.percentile(samples, [2.5, 97.5])]


def main() -> dict:
    started = time.monotonic()
    token = _token()
    if not token:
        raise RuntimeError("Hugging Face token unavailable")
    dev_path = Path(hf_hub_download(DATASET_ID, DEV_FILE, repo_type="dataset",
                                    revision=DATASET_REVISION, token=token))
    dev_speakers = set(pq.read_table(dev_path, columns=["sp_id"]).column("sp_id").to_pylist())
    paths = {data_file: Path(hf_hub_download(DATASET_ID, data_file, repo_type="dataset",
                                             revision=DATASET_REVISION, token=token))
             for data_file in DATA_FILES}
    all_rows, selected = _metadata(paths, dev_speakers)
    if (len(dev_speakers), len(selected), len({r["sp_id"] for r in selected}),
        len({r["native_state"] for r in selected})) != (26, 405, 106, 17):
        raise ValueError("speaker-disjoint Lahaja selection does not match frozen preflight")
    selection_sha256 = hashlib.sha256(json.dumps(
        [(r["data_file"], r["row_index"], r["sp_id"]) for r in selected],
        separators=(",", ":")
    ).encode()).hexdigest()
    if selection_sha256 != "6134e2a9eabd65f590bf4f7f7829626e520379a66e058d9dd51ce4acf36c4060":
        raise ValueError("held-out selection changed from pinned Turbo reference")
    audios = _audio(paths, {(r["data_file"], r["row_index"]) for r in selected})
    if any(abs(len(audios[(r["data_file"], r["row_index"])][0]) / 16000 - r["duration"]) > 0.12
           for r in selected):
        raise ValueError("audio duration does not match released metadata")
    print(json.dumps({"event": "preflight", "dataset_id": DATASET_ID,
                      "dataset_revision": DATASET_REVISION, "shards": DATA_FILES,
                      "eligible_rows_after_exclusion": len(all_rows), "selected_rows": len(selected),
                      "excluded_first_shard_speakers": len(dev_speakers),
                      "selected_speakers": len({r["sp_id"] for r in selected}),
                      "selected_states": len({r["native_state"] for r in selected}),
                      "selection_sha256": selection_sha256,
                      "selection": "first-shard-speaker-disjoint two-per-speaker/scenario sample, duration 2-25 s",
                      "reference": "verbatim", "model": MODEL_ID,
                      "language": "hi", "task": "transcribe",
                      "model_revision": MODEL_REVISION,
                      "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                      "transformers": transformers.__version__, "torch": torch.__version__},
                     sort_keys=True), flush=True)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required for this cloud replication")
    device = "cuda"
    dtype = torch.float16
    processor = AutoProcessor.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
    model = WhisperForConditionalGeneration.from_pretrained(
        MODEL_ID, revision=MODEL_REVISION, dtype=dtype,
        attn_implementation="sdpa"
    ).to(device).eval()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    predictions_path = OUTPUT / "predictions.jsonl"
    scored = []
    with predictions_path.open("w") as handle, torch.inference_mode():
        for position, row in enumerate(selected, 1):
            audio, sr = audios[(row["data_file"], row["row_index"])]
            inputs = processor(audio, sampling_rate=sr, return_tensors="pt")
            features = inputs.input_features.to(device=device, dtype=dtype)
            tokens = model.generate(features, language="hi", task="transcribe",
                                    num_beams=5, do_sample=False, max_new_tokens=256)
            hyp = processor.batch_decode(tokens, skip_special_tokens=True)[0].strip()
            ref = row["verbatim"]
            counts = jiwer.process_words(_normalize(ref), _normalize(hyp))
            record = {k: row[k] for k in ("data_file", "row_index", "fname", "sp_id", "scenario",
                                             "native_language", "native_state", "native_district",
                                             "duration", "age_group", "gender")}
            record.update({"reference": ref, "prediction": hyp,
                           "generated_tokens": int(tokens.shape[-1]),
                           "ref_words": len(_normalize(ref).split()),
                           "word_errors": counts.substitutions + counts.deletions + counts.insertions,
                           "substitutions": counts.substitutions,
                           "deletions": counts.deletions, "insertions": counts.insertions})
            scored.append(record)
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()
            if position % 10 == 0 or position == len(selected):
                print(json.dumps({"event": "progress", "done": position,
                                  "total": len(selected), "elapsed_s": round(time.monotonic()-started, 1),
                                  "pooled_wer": _pooled(scored)["wer"]}), flush=True)
    overall = _pooled(scored)
    overall["speaker_bootstrap_95_ci_wer"] = _speaker_interval(scored)
    scenarios = {name: _pooled([r for r in scored if r["scenario"] == name])
                 for name in sorted({r["scenario"] for r in scored})}
    states = {name: _pooled([r for r in scored if r["native_state"] == name])
              for name in sorted({r["native_state"] for r in scored})}
    result = {"status": "pilot_complete", "dataset": DATASET_ID,
              "dataset_revision": DATASET_REVISION, "shards": DATA_FILES,
              "excluded_first_shard_speakers": len(dev_speakers),
              "selection_sha256": selection_sha256,
              "sample_policy": "exclude first-shard speakers, then two-per-speaker/scenario in shards 1-3, 2-25 seconds",
              "model": MODEL_ID, "model_revision": MODEL_REVISION, "device": device,
              "dtype": str(dtype), "language": "hi", "task": "transcribe",
              "num_beams": 5, "max_new_tokens": 256,
              "scorer": "NFC lowercase, keep Unicode letters/marks/numbers, pooled word edits",
              "overall": overall, "scenarios": scenarios, "states_descriptive_only": states,
              "decoder_quality": {"generated_token_cap_256_count": sum(
                                      r["generated_tokens"] >= 256 for r in scored),
                                  "hypothesis_ge_3x_reference_words": sum(
                                      len(_normalize(r["prediction"]).split()) >= 3 * r["ref_words"]
                                      for r in scored),
                                  "wer_gt_100_percent": sum(
                                      r["word_errors"] > r["ref_words"] for r in scored)},
              "elapsed_s": time.monotonic() - started,
              "predictions_path": str(predictions_path)}
    (OUTPUT / "summary.json").write_text(json.dumps(result, indent=2, ensure_ascii=False))
    return result
