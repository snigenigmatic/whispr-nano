"""Mock-payload and corruption checks for independent Whisper acceptance."""
import gzip
import hashlib
import io
import json
import tarfile
import tempfile
import unittest
from pathlib import Path

from research.bridge_open.core_study import accept_whisper_core as acceptance
from research.bridge_open.scoring import normalize_v1, score


def digest(value):
    return hashlib.sha256(value).hexdigest()


def dump(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def make_run(base):
    run = Path(base) / "run"; run.mkdir()
    reference_raw = json.dumps({"reference_text": "hello"})
    source = {"conversation_id": "mock-hi", "source_repo": "google/fleurs", "language": "hi", "duration_seconds": 1.0,
              "audio_sha256": "a" * 64, "reference_sha256": digest(reference_raw.encode())}
    selected = {"condition_id": "small-fleurs-hi", "source_manifest": "experiments/fleurs_manifest.jsonl",
                "source_manifest_sha256": None, "conversation_ids": ["mock-hi"], "expected_count": 1}
    manifest = (json.dumps(source) + "\n").encode(); selected["source_manifest_sha256"] = digest(manifest)
    suite = {"model": {"model_id": acceptance.MODEL_ID, "model_revision": acceptance.MODEL_REVISION, "native_model": "small"},
             "fixed_short_route": {"batch_size": 8}, "fixed_long_route": {"native_model": "small"},
             "source_manifests": {"fleurs": {"path": selected["source_manifest"], "sha256": selected["source_manifest_sha256"]}},
             "conditions": [{"condition_id": "small-fleurs-hi", "source": "fleurs", "language": "hi", "expected_count": 1,
                             "max_wall_seconds": 1500}]}
    legacy = {"native_transcribe": {"verbose": None}}
    payload = {"experiments/selected_condition.json": json.dumps(selected).encode(),
               "experiments/core_suite.json": json.dumps(suite).encode(), "experiments/config.json": json.dumps(legacy).encode(),
               "experiments/fleurs_manifest.jsonl": manifest, "research/bridge_open/frozen_references.json": json.dumps({"mock-hi": reference_raw}).encode(),
               "experiments/hf_run.py": b"worker", "experiments/diagnostic_v1.py": b"diagnostic", "experiments/hf_entrypoint.sh": b"#!/bin/bash\n"}
    raw = io.BytesIO()
    with tarfile.open(fileobj=raw, mode="w") as archive:
        for name, data in payload.items():
            member = tarfile.TarInfo(name); member.size = len(data); archive.addfile(member, io.BytesIO(data))
    source_tar = raw.getvalue(); archive_blob = gzip.compress(source_tar)
    (run / "source_snapshot.tar.gz").write_bytes(archive_blob)
    dump(run / "orx_source_receipt.json", {"provider_status": "done", "commit": acceptance.TRUSTED_COMMIT,
                                             "portable_archive": "source_snapshot.tar.gz", "portable_archive_sha256": digest(archive_blob), "sourceDigest": digest(source_tar)})
    expected, _ = acceptance._expected_config(payload); dump(run / "config.json", expected)
    cfg_sha = digest((run / "config.json").read_bytes())
    safe_sha, native_sha = "b" * 64, "c" * 64
    dump(run / "hf_receipt.json", {"model_id": acceptance.MODEL_ID, "requested_revision": acceptance.MODEL_REVISION,
                                    "files": {"model.safetensors": safe_sha}})
    dump(run / "native_receipt.json", {"native_model": "small", "official_download_url": "https://example/small.pt",
                                        "official_url_sha256": native_sha, "checkpoint_sha256": native_sha})
    dump(run / "environment.json", {"model": {"id": acceptance.MODEL_ID, "revision": acceptance.MODEL_REVISION,
                                                "hf_safetensors_sha256": {"model.safetensors": safe_sha},
                                                "native_checkpoint_sha256": native_sha, "native_official_download_url": "https://example/small.pt"},
                                        "code": {"worker_sha256": digest(b"worker"), "hf_key_source_sha256": digest(b"diagnostic")},
                                        "dtype": "float32", "config_sha256": cfg_sha,
                                        "validation": {"weight_tensor_count": 1, "feature_count": 1}})
    dump(run / "weight_validation.json", [{"native_key": "n", "hf_key": "h", "learned": True, "max_abs_delta": 0.0}])
    dump(run / "feature_validation.json", [{"conversation_id": "mock-hi", "n_mels": 80, "max_abs_delta": 0.0}])
    dump(run / "source_mapping.json", {"source_manifest_sha256": selected["source_manifest_sha256"], "rows": [{"conversation_id": "mock-hi", "audio_sha256": source["audio_sha256"], "reference_sha256": source["reference_sha256"]}]})
    metrics = score("hello", "hello")
    prediction = {"conversation_id": "mock-hi", "language": "hi", "model": {"hf_model_id": acceptance.MODEL_ID,
                  "hf_model_revision": acceptance.MODEL_REVISION, "native_model": "small"},
                  "evidence": {"input_audio_sha256": source["audio_sha256"], "reference_sha256": source["reference_sha256"],
                               "config_sha256": cfg_sha, "worker_sha256": digest(b"worker")},
                  "full_wav_length_samples": 16000, "full_waveform": {"sample_rate": 16000, "sample_count": 16000, "duration_seconds": 1.0},
                  "route": "hf_batched_shortform", "hf_generation": {"token_ids": [1, 2], "options": {"num_beams": 5, "max_new_tokens": 440}},
                  "eos": {"status": "present", "missing_eos": False, "cap_observed": "unknown"}, "status": "success",
                  "text_raw": "hello", "text_for_scoring": normalize_v1("hello"), "metrics": metrics,
                  "native_short_parity": {"parity_phase": "fixed_three_short_files", "text_raw": "hello", "text_matches_hf": True,
                                          "eos": {"status": "unknown"}}}
    (run / "predictions.jsonl").write_text(json.dumps(prediction) + "\n")
    dump(run / "summary.json", {"status": "completed", "attempted": 1, "prediction_records": 1, "scored": 1,
                                  "runtime_failures": 0, "status_counts": {"success": 1}, "micro_wer": 0.0})
    dump(run / "validation.json", {"expected_count": 1, "prediction_count": 1, "unique_prediction_ids": 1,
                                    "missing_prediction_ids": [], "weight_gate": "passed", "feature_gate": "passed"})
    return run


class WhisperCoreAcceptanceTest(unittest.TestCase):
    def test_accepts_complete_mock_payload(self):
        with tempfile.TemporaryDirectory() as directory:
            result = acceptance.validate(make_run(directory))
            self.assertTrue(result["valid"])

    def test_rejects_metric_corruption(self):
        with tempfile.TemporaryDirectory() as directory:
            run = make_run(directory)
            row = json.loads((run / "predictions.jsonl").read_text()); row["metrics"]["wer"] = 0.5
            (run / "predictions.jsonl").write_text(json.dumps(row) + "\n")
            with self.assertRaises(ValueError):
                acceptance.validate(run)

    def test_rejects_untrusted_source_commit(self):
        with tempfile.TemporaryDirectory() as directory:
            run = make_run(directory)
            receipt = json.loads((run / "orx_source_receipt.json").read_text())
            receipt["commit"] = "0" * 40
            dump(run / "orx_source_receipt.json", receipt)
            with self.assertRaises(ValueError):
                acceptance.validate(run)


if __name__ == "__main__":
    unittest.main()
