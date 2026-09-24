"""Mock-payload and corruption checks for independent Whisper acceptance."""
import gzip
import hashlib
import io
import json
import tarfile
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from research.bridge_open.core_study import accept_whisper_core as acceptance
from research.bridge_open.scoring import normalize_v1, score


def digest(value):
    return hashlib.sha256(value).hexdigest()


def dump(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


SMALL_KEY = ("small-fleurs-hi", acceptance.MODEL_ID, acceptance.MODEL_REVISION, "small")


def mock_small_contract():
    return {"commit": "949a23078e4b15745aa752a373982f74634997e0", "count": 1, "language": "hi", "n_mels": 80}


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
    dump(run / "orx_source_receipt.json", {"provider_status": "done", "commit": acceptance.KNOWN_CONTRACTS[SMALL_KEY]["commit"],
                                             "portable_archive": "source_snapshot.tar.gz", "portable_archive_sha256": digest(archive_blob), "sourceDigest": digest(source_tar)})
    expected, _, _ = acceptance._expected_config(payload); dump(run / "config.json", expected)
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
                                        "validation": {"weight_tensor_count": 1, "feature_count": 1, "n_mels": 80}})
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


def make_mixed_route_run(base):
    """Build a faithful two-row archive whose manifest is long then short.

    The worker writes all batched short predictions first, followed by native
    long-form records.  This fixture makes that ordering observable without
    depending on an imported GPU result.
    """
    run = Path(base) / "run"; run.mkdir()
    reference_raw = {
        "mock-long": json.dumps({"reference_text": "long reference"}),
        "mock-short": json.dumps({"reference_text": "short reference"}),
    }
    sources = [
        {"conversation_id": "mock-long", "source_repo": "google/fleurs", "language": "hi", "duration_seconds": 31.0,
         "audio_sha256": "a" * 64, "reference_sha256": digest(reference_raw["mock-long"].encode())},
        {"conversation_id": "mock-short", "source_repo": "google/fleurs", "language": "hi", "duration_seconds": 1.0,
         "audio_sha256": "b" * 64, "reference_sha256": digest(reference_raw["mock-short"].encode())},
    ]
    selected = {"condition_id": "small-fleurs-hi", "source_manifest": "experiments/fleurs_manifest.jsonl",
                "source_manifest_sha256": None, "conversation_ids": [row["conversation_id"] for row in sources], "expected_count": 2}
    manifest = ("".join(json.dumps(row) + "\n" for row in sources)).encode()
    selected["source_manifest_sha256"] = digest(manifest)
    suite = {"model": {"model_id": acceptance.MODEL_ID, "model_revision": acceptance.MODEL_REVISION, "native_model": "small"},
             "fixed_short_route": {"batch_size": 8}, "fixed_long_route": {"native_model": "small"},
             "source_manifests": {"fleurs": {"path": selected["source_manifest"], "sha256": selected["source_manifest_sha256"]}},
             "conditions": [{"condition_id": "small-fleurs-hi", "source": "fleurs", "language": "hi", "expected_count": 2,
                             "max_wall_seconds": 1500}]}
    legacy = {"native_transcribe": {"verbose": None}}
    payload = {"experiments/selected_condition.json": json.dumps(selected).encode(),
               "experiments/core_suite.json": json.dumps(suite).encode(), "experiments/config.json": json.dumps(legacy).encode(),
               "experiments/fleurs_manifest.jsonl": manifest, "research/bridge_open/frozen_references.json": json.dumps(reference_raw).encode(),
               "experiments/hf_run.py": b"worker", "experiments/diagnostic_v1.py": b"diagnostic", "experiments/hf_entrypoint.sh": b"#!/bin/bash\n"}
    raw = io.BytesIO()
    with tarfile.open(fileobj=raw, mode="w") as archive:
        for name, data in payload.items():
            member = tarfile.TarInfo(name); member.size = len(data); archive.addfile(member, io.BytesIO(data))
    source_tar = raw.getvalue(); archive_blob = gzip.compress(source_tar)
    (run / "source_snapshot.tar.gz").write_bytes(archive_blob)
    contract = {"commit": "949a23078e4b15745aa752a373982f74634997e0", "count": 2, "language": "hi", "n_mels": 80}
    dump(run / "orx_source_receipt.json", {"provider_status": "done", "commit": contract["commit"], "portable_archive": "source_snapshot.tar.gz",
                                             "portable_archive_sha256": digest(archive_blob), "sourceDigest": digest(source_tar)})
    with patch.dict(acceptance.KNOWN_CONTRACTS, {SMALL_KEY: contract}):
        expected, _, _ = acceptance._expected_config(payload)
    dump(run / "config.json", expected)
    config_sha = digest((run / "config.json").read_bytes())
    safe_sha, native_sha = "c" * 64, "d" * 64
    dump(run / "hf_receipt.json", {"model_id": acceptance.MODEL_ID, "requested_revision": acceptance.MODEL_REVISION,
                                    "files": {"model.safetensors": safe_sha}})
    dump(run / "native_receipt.json", {"native_model": "small", "official_download_url": "https://example/small.pt",
                                        "official_url_sha256": native_sha, "checkpoint_sha256": native_sha})
    dump(run / "environment.json", {"model": {"id": acceptance.MODEL_ID, "revision": acceptance.MODEL_REVISION,
                                                "hf_safetensors_sha256": {"model.safetensors": safe_sha},
                                                "native_checkpoint_sha256": native_sha, "native_official_download_url": "https://example/small.pt"},
                                        "code": {"worker_sha256": digest(b"worker"), "hf_key_source_sha256": digest(b"diagnostic")},
                                        "dtype": "float32", "config_sha256": config_sha,
                                        "validation": {"weight_tensor_count": 1, "feature_count": 2, "n_mels": 80}})
    dump(run / "weight_validation.json", [{"native_key": "n", "hf_key": "h", "learned": True, "max_abs_delta": 0.0}])
    dump(run / "feature_validation.json", [{"conversation_id": row["conversation_id"], "n_mels": 80, "max_abs_delta": 0.0} for row in sources])
    dump(run / "source_mapping.json", {"source_manifest_sha256": selected["source_manifest_sha256"],
                                         "rows": [{"conversation_id": row["conversation_id"], "audio_sha256": row["audio_sha256"],
                                                   "reference_sha256": row["reference_sha256"]} for row in sources]})

    def evidence(source):
        return {"input_audio_sha256": source["audio_sha256"], "reference_sha256": source["reference_sha256"],
                "config_sha256": config_sha, "worker_sha256": digest(b"worker")}

    short, long = sources[1], sources[0]
    short_raw, long_raw = "short reference", "long reference"
    short_prediction = {"conversation_id": short["conversation_id"], "language": "hi", "model": {"hf_model_id": acceptance.MODEL_ID,
                        "hf_model_revision": acceptance.MODEL_REVISION, "native_model": "small"}, "evidence": evidence(short),
                        "full_wav_length_samples": 16000, "full_waveform": {"sample_rate": 16000, "sample_count": 16000, "duration_seconds": 1.0},
                        "route": "hf_batched_shortform", "hf_generation": {"token_ids": [1, 2], "options": {"num_beams": 5, "max_new_tokens": 440}},
                        "eos": {"status": "present", "missing_eos": False, "cap_observed": "unknown"}, "status": "success",
                        "text_raw": short_raw, "text_for_scoring": normalize_v1(short_raw), "metrics": score(short_raw, short_raw),
                        "native_short_parity": {"parity_phase": "fixed_three_short_files", "text_raw": short_raw, "text_matches_hf": True,
                                                "eos": {"status": "unknown"}}}
    long_prediction = {"conversation_id": long["conversation_id"], "language": "hi", "model": {"hf_model_id": acceptance.MODEL_ID,
                       "hf_model_revision": acceptance.MODEL_REVISION, "native_model": "small"}, "evidence": evidence(long),
                       "full_wav_length_samples": 496000, "full_waveform": {"sample_rate": 16000, "sample_count": 496000, "duration_seconds": 31.0},
                       "route": "native_longform_full_waveform", "native_longform": {"effective_options": {"language": "hi", "verbose": None}, "decode_attempts": []},
                       "eos": {"status": "unknown"}, "status": "success", "text_raw": long_raw,
                       "text_for_scoring": normalize_v1(long_raw), "metrics": score(long_raw, long_raw)}
    (run / "predictions.jsonl").write_text("\n".join(json.dumps(row) for row in (short_prediction, long_prediction)) + "\n")
    dump(run / "summary.json", {"status": "completed", "attempted": 2, "prediction_records": 2, "scored": 2,
                                  "runtime_failures": 0, "status_counts": {"success": 2}, "micro_wer": 0.0})
    dump(run / "validation.json", {"expected_count": 2, "prediction_count": 2, "unique_prediction_ids": 2,
                                    "missing_prediction_ids": [], "weight_gate": "passed", "feature_gate": "passed"})
    return run, contract


class WhisperCoreAcceptanceTest(unittest.TestCase):
    def test_accepts_complete_mock_payload(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch.dict(acceptance.KNOWN_CONTRACTS, {SMALL_KEY: mock_small_contract()}):
                result = acceptance.validate(make_run(directory))
            self.assertTrue(result["valid"])

    def test_rejects_metric_corruption(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch.dict(acceptance.KNOWN_CONTRACTS, {SMALL_KEY: mock_small_contract()}):
                run = make_run(directory)
                row = json.loads((run / "predictions.jsonl").read_text()); row["metrics"]["wer"] = 0.5
                (run / "predictions.jsonl").write_text(json.dumps(row) + "\n")
                with self.assertRaises(ValueError):
                    acceptance.validate(run)

    def test_accepts_short_then_long_worker_stream_order(self):
        with tempfile.TemporaryDirectory() as directory:
            run, contract = make_mixed_route_run(directory)
            with patch.dict(acceptance.KNOWN_CONTRACTS, {SMALL_KEY: contract}):
                result = acceptance.validate(run)
        self.assertTrue(result["valid"])
        self.assertEqual(result["records"], 2)

    def test_rejects_manifest_order_when_mixed_routes_stream_short_first(self):
        with tempfile.TemporaryDirectory() as directory:
            run, contract = make_mixed_route_run(directory)
            rows = (run / "predictions.jsonl").read_text().splitlines()
            (run / "predictions.jsonl").write_text(rows[1] + "\n" + rows[0] + "\n")
            with patch.dict(acceptance.KNOWN_CONTRACTS, {SMALL_KEY: contract}), self.assertRaises(ValueError):
                acceptance.validate(run)

    def test_rejects_missing_mixed_route_prediction(self):
        with tempfile.TemporaryDirectory() as directory:
            run, contract = make_mixed_route_run(directory)
            rows = (run / "predictions.jsonl").read_text().splitlines()
            (run / "predictions.jsonl").write_text(rows[0] + "\n")
            with patch.dict(acceptance.KNOWN_CONTRACTS, {SMALL_KEY: contract}), self.assertRaises(ValueError):
                acceptance.validate(run)

    def test_rejects_duplicate_mixed_route_prediction(self):
        with tempfile.TemporaryDirectory() as directory:
            run, contract = make_mixed_route_run(directory)
            short = (run / "predictions.jsonl").read_text().splitlines()[0]
            (run / "predictions.jsonl").write_text(short + "\n" + short + "\n")
            with patch.dict(acceptance.KNOWN_CONTRACTS, {SMALL_KEY: contract}), self.assertRaises(ValueError):
                acceptance.validate(run)

    def test_rejects_metric_corruption_in_mixed_route_stream(self):
        with tempfile.TemporaryDirectory() as directory:
            run, contract = make_mixed_route_run(directory)
            rows = [json.loads(line) for line in (run / "predictions.jsonl").read_text().splitlines()]
            rows[1]["metrics"]["wer"] = 1.0
            (run / "predictions.jsonl").write_text("\n".join(json.dumps(row) for row in rows) + "\n")
            with patch.dict(acceptance.KNOWN_CONTRACTS, {SMALL_KEY: contract}), self.assertRaises(ValueError):
                acceptance.validate(run)

    def test_rejects_untrusted_source_commit(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch.dict(acceptance.KNOWN_CONTRACTS, {SMALL_KEY: mock_small_contract()}):
                run = make_run(directory)
                receipt = json.loads((run / "orx_source_receipt.json").read_text())
                receipt["commit"] = "0" * 40
                dump(run / "orx_source_receipt.json", receipt)
                with self.assertRaises(ValueError):
                    acceptance.validate(run)

    def test_known_contracts_cover_exact_model_and_mel_pins(self):
        self.assertEqual(len(acceptance.KNOWN_CONTRACTS), 6)
        expected = {
            "small-fleurs-hi": (418, 80), "small-fleurs-bn": (920, 80),
            "small-fleurs-as": (984, 80), "small-svarah-en-shard02": (518, 80),
            "turbo-svarah-en-shard02": (518, 128), "large-v3-svarah-en-shard02": (518, 128),
        }
        observed = {key[0]: (value["count"], value["n_mels"]) for key, value in acceptance.KNOWN_CONTRACTS.items()}
        self.assertEqual(observed, expected)


if __name__ == "__main__":
    unittest.main()
