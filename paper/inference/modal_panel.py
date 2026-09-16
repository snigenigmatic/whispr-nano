"""Pinned, bounded matched-panel inference; no external ast_asr dependency.

Run smoke first, then panel. The host reserves a conservative dollar maximum
before every invocation. Every model persists partial results after each batch.
"""
from __future__ import annotations

import csv
import hashlib
import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parents[2] if modal.is_local() else Path("/root")
app = modal.App("svarah-paper-matched-panel")
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("ffmpeg")
    .pip_install(
        "torch==2.13.0", "transformers==5.16.1", "accelerate==1.11.0",
        "datasets==4.4.1", "torchcodec==0.16.0", "soundfile==0.14.0",
        "librosa==1.0.0", "numpy==2.5.2",
    )
    .env({"HF_HOME": "/cache/huggingface", "HF_HUB_ENABLE_HF_TRANSFER": "0"})
)
cache = modal.Volume.from_name("ast-asr-cache")
outputs = modal.Volume.from_name("svarah-paper-20260909", create_if_missing=True)
DATASET_REVISION = "ebbf7777fe771490696a3f7b007097606fa8c924"
RATE_PER_SECOND = 0.000164 + 2 * 0.0000131 + 16 * 0.00000222
TIMEOUT = 1800
RESERVE_PER_JOB = 1.50


@app.function(
    image=image, gpu="T4", cpu=(2, 2), memory=(16384, 16384), timeout=TIMEOUT,
    startup_timeout=300, retries=0, max_containers=10, scaledown_window=2,
    volumes={"/cache/huggingface": cache, "/outputs": outputs},
    secrets=[modal.Secret.from_name("huggingface")],
)
def transcribe(config: dict) -> dict:
    import importlib.metadata
    import tempfile
    import numpy as np
    import soundfile as sf
    import torch
    from datasets import Audio, load_dataset
    from transformers import AutoModelForMultimodalLM, AutoProcessor
    from transformers import WhisperForConditionalGeneration, WhisperProcessor

    start = time.monotonic()
    model_id, revision = config["model_id"], config["revision"]
    tag = model_id.split("/")[-1]
    is_qwen = model_id.startswith("Qwen/")
    torch.manual_seed(0)
    ds = load_dataset("ai4bharat/svarah", revision=DATASET_REVISION, split="test")
    ds = ds.cast_column("audio_filepath", Audio(sampling_rate=16000))
    if len(ds) != 6656:
        raise ValueError("Pinned dataset row count changed")
    if is_qwen:
        processor = AutoProcessor.from_pretrained(model_id, revision=revision)
        model = AutoModelForMultimodalLM.from_pretrained(model_id, revision=revision, dtype=torch.float32)
    else:
        processor = WhisperProcessor.from_pretrained(model_id, revision=revision)
        model = WhisperForConditionalGeneration.from_pretrained(model_id, revision=revision, dtype=torch.float32)
    model.to("cuda").eval()
    load_seconds = time.monotonic() - start
    generation_config = model.generation_config.to_dict()
    decode_config = {"dtype": "float32", "batch_size": 4, "do_sample": False,
                     "max_new_tokens": 256 if is_qwen else 200,
                     "language": "English" if is_qwen else "en", "task": "transcribe"}
    versions = {p: importlib.metadata.version(p) for p in
                ("torch", "transformers", "datasets", "numpy", "torchcodec", "soundfile", "librosa")}
    manifest = config["manifest"]
    rows = []
    run_dir = Path("/outputs") / config["run_id"] / tag / str(config.get("shard_id", "complete"))
    run_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "model_id": model_id, "model_revision": revision,
        "dataset_revision": DATASET_REVISION, "dataset_fingerprint": ds._fingerprint,
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "gpu": torch.cuda.get_device_name(), "requested_gpu": "T4",
        "decode_config": decode_config, "generation_config": generation_config,
        "versions": versions, "load_seconds": load_seconds,
        "requested_rows": len(manifest), "run_id": config["run_id"],
        "job_id": config.get("job_id", config["run_id"] + "/" + model_id),
        "shard_id": config.get("shard_id"),
    }
    with tempfile.TemporaryDirectory() as tmp:
        for begin in range(0, len(manifest), 4):
            if time.monotonic() - start > TIMEOUT - 120:
                raise TimeoutError("Stopping before hard timeout; partial rows are in the output volume")
            expected_batch = manifest[begin:begin+4]
            source_batch = [ds[int(x["utt_id"].split("_")[-1])] for x in expected_batch]
            arrays = []
            for source, expected in zip(source_batch, expected_batch):
                if source["text"] != expected["reference_text"] or str(source["primary_language"]).strip().title() != expected["primary_language"]:
                    raise ValueError(f"Dataset index provenance mismatch at {expected['utt_id']}")
                audio = source["audio_filepath"]
                if hasattr(audio, "get_all_samples"):
                    samples = audio.get_all_samples()
                    arrays.append(samples.data.mean(dim=0).numpy().astype(np.float32))
                    sr = int(samples.sample_rate)
                else:
                    arrays.append(np.asarray(audio["array"], dtype=np.float32))
                    sr = int(audio["sampling_rate"])
                if sr != 16000:
                    raise ValueError(f"Expected 16kHz audio, got {sr}")
            with torch.no_grad():
                if is_qwen:
                    paths = []
                    for j, arr in enumerate(arrays):
                        path = Path(tmp) / f"clip_{j}.wav"
                        sf.write(str(path), arr, 16000)
                        paths.append(str(path))
                    inputs = processor.apply_transcription_request(audio=paths, language=["English"]*len(paths)).to("cuda", torch.float32)
                    ids = model.generate(**inputs, max_new_tokens=256, do_sample=False)
                    hyps = processor.decode(ids[:, inputs["input_ids"].shape[1]:], return_format="transcription_only")
                else:
                    inputs = processor(arrays, sampling_rate=16000, return_tensors="pt", padding="max_length",
                                       truncation=True, return_attention_mask=True)
                    ids = model.generate(input_features=inputs.input_features.to("cuda", torch.float32),
                                         attention_mask=inputs.attention_mask.to("cuda"), do_sample=False,
                                         num_beams=1, language="en", task="transcribe", max_new_tokens=200)
                    hyps = processor.batch_decode(ids, skip_special_tokens=True)
            if len(hyps) != len(expected_batch) or not all(isinstance(h, str) for h in hyps):
                raise ValueError("Decoder did not return one string per input")
            rows.extend({**expected, "hypothesis_text": hyp.strip()} for expected, hyp in zip(expected_batch, hyps))
            (run_dir / "predictions.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))
            if begin % 100 == 0 or len(rows) == len(manifest):
                outputs.commit()
                print(f"[{tag}] {len(rows)}/{len(manifest)} in {time.monotonic()-start:.1f}s", flush=True)
    metadata["wall_seconds"] = time.monotonic() - start
    metadata["completed_rows"] = len(rows)
    metadata["estimated_resource_cost_usd"] = metadata["wall_seconds"] * RATE_PER_SECOND
    (run_dir / "runtime.json").write_text(json.dumps(metadata, indent=2) + "\n")
    outputs.commit()
    return {"metadata": metadata, "rows": rows}


def reserve_jobs(run_id: str, configs: list[dict]) -> None:
    ledger_path = ROOT / "paper/inference/budget_ledger.json"
    ledger = json.loads(ledger_path.read_text())
    if datetime.now(timezone.utc) >= datetime.fromisoformat(ledger["deadline_utc"].replace("Z", "+00:00")):
        raise RuntimeError("Sprint deadline passed; no new jobs")
    reserved = sum(job["reserved_usd"] for job in ledger["jobs"])
    additional = sum(config.get("reserve_usd", RESERVE_PER_JOB) for config in configs)
    if reserved + additional > ledger["authorized_limit"]:
        raise RuntimeError("Job reservations would exceed the authorized $30 total")
    if any(job["id"].startswith(run_id + "/") for job in ledger["jobs"]):
        raise RuntimeError("Run ID already reserved; choose a unique run ID for an explicit retry")
    for config in configs:
        ledger["jobs"].append({
            "id": config.get("job_id", run_id + "/" + config["model_id"]),
            "purpose": config.get("purpose", "matched-panel transcription"),
            "model_revision": config["revision"], "n_utterances": len(config["manifest"]),
            "reserved_usd": config.get("reserve_usd", RESERVE_PER_JOB), "timeout_seconds": TIMEOUT,
            "startup_timeout_seconds": 300, "cpu_limit_cores": 2, "memory_limit_gib": 16,
            "gpu": "T4", "cpu_cores": 2, "memory_gib": 16, "status": "reserved",
        })
    ledger_path.write_text(json.dumps(ledger, indent=2) + "\n")


def save_result(result: dict, destination: Path) -> None:
    import jiwer
    destination.mkdir(parents=True, exist_ok=True)
    metadata, rows = result["metadata"], result["rows"]
    tag = metadata["model_id"].split("/")[-1]
    columns = ["utt_id", "primary_language", "family", "gender", "duration_s", "reference_text", "reference_normalized",
               "hypothesis_text", "hypothesis_normalized", "wer", "n_sub", "n_del", "n_ins", "ref_len_words"]
    normalize = lambda s: " ".join(re.sub(r"[^\w\s']", " ", s.lower()).split())
    with (destination / f"results_{tag}_svarah_clean.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            ref, hyp = normalize(row["reference_text"]), normalize(row["hypothesis_text"])
            scored = jiwer.process_words(ref, hyp)
            writer.writerow({**row, "reference_normalized": ref, "hypothesis_normalized": hyp,
                             "wer": scored.wer, "n_sub": scored.substitutions, "n_del": scored.deletions,
                             "n_ins": scored.insertions, "ref_len_words": len(ref.split())})
    (destination / f"runtime_{tag}.json").write_text(json.dumps(metadata, indent=2) + "\n")
    ledger_path = ROOT / "paper/inference/budget_ledger.json"
    ledger = json.loads(ledger_path.read_text())
    for job in ledger["jobs"]:
        if job["id"] == metadata.get("job_id", metadata["run_id"] + "/" + metadata["model_id"]):
            job.update(status="completed", wall_seconds=metadata["wall_seconds"],
                       estimated_resource_cost_usd=metadata["estimated_resource_cost_usd"])
    ledger_path.write_text(json.dumps(ledger, indent=2) + "\n")


@app.local_entrypoint()
def main(mode: str = "smoke", run_id: str = "smoke-1", model_id: str = ""):
    if mode not in ("smoke", "panel", "full"):
        raise ValueError("mode must be smoke, panel, or full")
    revisions = json.loads((ROOT / "paper/data/model_revisions.json").read_text())
    with (ROOT / "audit/svarah_fairness_audit/results_saarasv3_svarah_clean.csv").open(newline="") as handle:
        manifest = [{k: row[k] for k in ("utt_id", "primary_language", "family", "gender", "duration_s", "reference_text")}
                    for row in csv.DictReader(handle)]
    if mode == "full":
        historical = {}
        for filename in ("results_whisper-large-v3_svarah_clean.csv", "results_saarasv3_svarah_clean.csv"):
            with (ROOT / "audit/svarah_fairness_audit" / filename).open(newline="") as handle:
                historical.update({row["utt_id"]: row for row in csv.DictReader(handle)})
        with (ROOT / "paper/data/svarah_metadata.csv").open(newline="") as handle:
            manifest = [{key: row[key] for key in ("utt_id", "primary_language", "family", "gender", "duration_s", "reference_text")}
                        for row in csv.DictReader(handle)]
        for row in manifest:
            if row["utt_id"] in historical:
                prior = historical[row["utt_id"]]
                if row["reference_text"] != prior["reference_text"] or row["primary_language"] != prior["primary_language"]:
                    raise ValueError("Full metadata disagrees with a historical reference")
                # Preserve the validated archived duration string, avoiding
                # harmless float-format differences in overlap provenance.
                row["duration_s"] = prior["duration_s"]
        manifest.sort(key=lambda row: row["utt_id"])
        if len(manifest) != 6656 or len({r["utt_id"] for r in manifest}) != 6656:
            raise ValueError("Full benchmark manifest must contain 6,656 unique rows")
        selected = list(revisions)
        frozen = {"dataset_revision": DATASET_REVISION, "n_utterances": 6656,
                  "metadata_sha256": hashlib.sha256((ROOT/"paper/data/svarah_metadata.csv").read_bytes()).hexdigest(),
                  "manifest_sha256": hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest(),
                  "selection": "All pinned test rows; extension chosen after Matched-450 outlier was observed",
                  "manifest": manifest, "model_revisions": revisions}
        (ROOT/"paper/data/full_input_manifest.json").write_text(json.dumps(frozen, indent=2)+"\n")
    elif mode == "smoke":
        manifest = manifest[:12]
        selected = ["openai/whisper-tiny", "Qwen/Qwen3-ASR-0.6B-hf"]
    else:
        selected = list(revisions)
    if model_id:
        if model_id not in revisions:
            raise ValueError("Unknown model ID")
        selected = [model_id]
    if mode == "full":
        configs = [{"model_id": model, "revision": revisions[model]["revision"],
                    "manifest": manifest[start:start+1152], "run_id": run_id,
                    "shard_id": start//1152, "job_id": f"{run_id}/{model}/shard-{start//1152:02d}",
                    "reserve_usd": 0.50, "purpose": "Full 6,656-utterance benchmark, bounded shard"}
                   for start in range(0, len(manifest), 1152) for model in selected]
    else:
        configs = [{"model_id": model, "revision": revisions[model]["revision"], "manifest": manifest, "run_id": run_id}
                   for model in selected]
    reserve_jobs(run_id, configs)
    destination = ROOT / "paper/data" / ("full_prediction_parts" if mode == "full" else "smoke_predictions" if mode == "smoke" else "matched_predictions")
    print(f"Reserved {len(configs)} jobs; maximum additional reservation ${sum(c.get('reserve_usd', RESERVE_PER_JOB) for c in configs):.2f}", flush=True)
    for result in transcribe.map(configs, order_outputs=False):
        part_destination = destination / f"shard-{result['metadata']['shard_id']:02d}" if mode == "full" else destination
        save_result(result, part_destination)
        print(json.dumps({k: result["metadata"][k] for k in ("model_id", "completed_rows", "wall_seconds", "estimated_resource_cost_usd")}))
    if mode == "full":
        output = ROOT/"paper/data/full_predictions"
        output.mkdir(parents=True, exist_ok=True)
        expected = {r["utt_id"] for r in manifest}
        for model in selected:
            tag=model.split("/")[-1]
            combined, runtimes = [], []
            for shard in range(6):
                path=destination/f"shard-{shard:02d}"
                with (path/f"results_{tag}_svarah_clean.csv").open(newline="") as handle:
                    reader=csv.DictReader(handle); columns=reader.fieldnames; combined.extend(reader)
                runtimes.append(json.loads((path/f"runtime_{tag}.json").read_text()))
            if len(combined)!=6656 or {r["utt_id"] for r in combined}!=expected:
                raise ValueError(f"Full shard union incomplete or duplicated for {model}")
            with (output/f"results_{tag}_svarah_clean.csv").open("w",newline="") as handle:
                writer=csv.DictWriter(handle,fieldnames=columns); writer.writeheader()
                writer.writerows(sorted(combined,key=lambda row:row["utt_id"]))
            (output/f"runtime_{tag}.json").write_text(json.dumps({"model_id":model,"model_revision":revisions[model]["revision"],"n_utterances":6656,"shards":runtimes},indent=2)+"\n")
        print("All five complete 6,656-row prediction files assembled.")
