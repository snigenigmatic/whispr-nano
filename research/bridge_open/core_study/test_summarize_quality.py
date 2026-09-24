"""Regression tests for independently rescored quality consolidation."""
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from research.bridge_open.core_study import summarize_quality as quality
from research.bridge_open.scoring import normalize_v1, score


def sha(value):
    return hashlib.sha256(value).hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) + "\n")


def make_fixture(base):
    root = Path(base)
    ref = root / "references" / "one.json"
    reference = {"reference_text": "hello world"}
    write_json(ref, reference)
    source = {"conversation_id": "one", "language": "hi", "audio_sha256": "a" * 64,
              "reference_path": "references/one.json", "reference_sha256": sha(ref.read_bytes())}
    manifest = root / quality.FLEURS_MANIFEST
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(json.dumps(source) + "\n")
    artifact = root / "research/bridge_open/artifacts" / "fixture"
    artifact.mkdir(parents=True)
    prediction = {"conversation_id": "one", "language": "hi", "model_id": "fixture/model", "model_revision": "fixture-revision",
                  "audio_sha256": source["audio_sha256"],
                  "reference_sha256": source["reference_sha256"], "status": "success", "route": "fixture_route",
                  "text_raw": "hello", "text_for_scoring": normalize_v1("hello"), "metrics": score("hello world", "hello")}
    (artifact / "predictions.jsonl").write_text(json.dumps(prediction) + "\n")
    write_json(artifact / "config.json", {"model_id": "fixture/model", "model_revision": "fixture-revision"})
    write_json(artifact / "orx_source_receipt.json", {"provider_status": "done", "commit": "fixture-commit"})
    write_json(artifact / "acceptance.json", {"valid": True, "records": 1, "model": "fixture/model",
                                                "revision": "fixture-revision", "receipt": {"commit": "fixture-commit"}})
    spec = {"model": "fixture-model", "track": "fleurs-hi", "records": 1, "artifact": "fixture",
            "prediction": "predictions.jsonl", "acceptance": "acceptance.json", "receipt_kind": "valid",
            "model_id": "fixture/model", "model_revision": "fixture-revision", "commit": "fixture-commit"}
    hashes = {quality.FLEURS_MANIFEST: sha(manifest.read_bytes()), quality.SVARAH_MANIFEST: "unused"}
    return root, artifact, spec, hashes


class QualityConsolidationTest(unittest.TestCase):
    def test_rescores_and_writes_accepted_cell(self):
        with tempfile.TemporaryDirectory() as directory:
            root, _, spec, hashes = make_fixture(directory)
            result = quality.consolidate(root, [spec], hashes)
            cell = result["cells"][0]
            self.assertEqual((result["accepted_cell_count"], result["pending_cell_count"]), (1, 0))
            self.assertEqual(cell["word_errors"], 1)
            self.assertEqual(cell["route_counts"], {"fixture_route": 1})
            self.assertEqual(set(result["source_prediction_hashes"]), {"fixture-model::fleurs-hi"})
            output = root / "out"; quality.write_results(result, output)
            self.assertTrue((output / "quality_results.json").is_file())
            self.assertIn("No aggregate winner", (output / "QUALITY_RESULTS.md").read_text())

    def test_marks_absent_artifact_pending_without_zero_metrics(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            spec = {"model": "fixture-model", "track": "fleurs-hi", "records": 1, "artifact": "absent",
                    "prediction": "predictions.jsonl", "acceptance": "acceptance.json", "receipt_kind": "valid"}
            cell = quality.consolidate(root, [spec])["cells"][0]
            self.assertEqual(cell["status"], "pending")
            self.assertNotIn("micro_wer", cell)

    def test_rejects_duplicate_or_missing_prediction_ids(self):
        with tempfile.TemporaryDirectory() as directory:
            root, artifact, spec, hashes = make_fixture(directory)
            row = (artifact / "predictions.jsonl").read_text()
            (artifact / "predictions.jsonl").write_text(row + row)
            with self.assertRaises(ValueError):
                quality.consolidate(root, [spec], hashes)
        with tempfile.TemporaryDirectory() as directory:
            root, artifact, spec, hashes = make_fixture(directory)
            (artifact / "predictions.jsonl").write_text("")
            with self.assertRaises(ValueError):
                quality.consolidate(root, [spec], hashes)

    def test_rejects_corrupt_saved_metric(self):
        with tempfile.TemporaryDirectory() as directory:
            root, artifact, spec, hashes = make_fixture(directory)
            row = json.loads((artifact / "predictions.jsonl").read_text())
            row["metrics"]["wer"] = 0.0
            (artifact / "predictions.jsonl").write_text(json.dumps(row) + "\n")
            with self.assertRaises(ValueError):
                quality.consolidate(root, [spec], hashes)

    def test_rejects_prediction_evidence_without_independent_acceptance(self):
        with tempfile.TemporaryDirectory() as directory:
            root, artifact, spec, hashes = make_fixture(directory)
            (artifact / "acceptance.json").unlink()
            with self.assertRaises(ValueError):
                quality.consolidate(root, [spec], hashes)

    def test_rejects_runtime_error_even_with_rescored_metrics(self):
        with tempfile.TemporaryDirectory() as directory:
            root, artifact, spec, hashes = make_fixture(directory)
            row = json.loads((artifact / "predictions.jsonl").read_text())
            row["status"] = "runtime_error"
            (artifact / "predictions.jsonl").write_text(json.dumps(row) + "\n")
            with self.assertRaises(ValueError):
                quality.consolidate(root, [spec], hashes)

    def test_rejects_manifest_or_receipt_contract_corruption(self):
        with tempfile.TemporaryDirectory() as directory:
            root, artifact, spec, hashes = make_fixture(directory)
            manifest = root / quality.FLEURS_MANIFEST
            manifest.write_text(manifest.read_text() + "\n")
            with self.assertRaises(ValueError):
                quality.consolidate(root, [spec], hashes)
        with tempfile.TemporaryDirectory() as directory:
            root, artifact, spec, hashes = make_fixture(directory)
            receipt = json.loads((artifact / "acceptance.json").read_text())
            receipt["receipt"]["commit"] = "wrong"
            write_json(artifact / "acceptance.json", receipt)
            with self.assertRaises(ValueError):
                quality.consolidate(root, [spec], hashes)

    def test_distinguishes_punctuation_only_raw_from_normalized_empty_and_unknown_eos(self):
        with tempfile.TemporaryDirectory() as directory:
            root, artifact, spec, hashes = make_fixture(directory)
            row = json.loads((artifact / "predictions.jsonl").read_text())
            row["text_raw"] = "..."
            row["text_for_scoring"] = normalize_v1(row["text_raw"])
            row["metrics"] = score("hello world", row["text_raw"])
            row["eos"] = {"status": "unknown"}
            (artifact / "predictions.jsonl").write_text(json.dumps(row) + "\n")
            cell = quality.consolidate(root, [spec], hashes)["cells"][0]
            self.assertEqual(cell["empty_outputs"], 1)
            self.assertEqual(cell["raw_empty_outputs"], 0)
            self.assertEqual(cell["eos_observable_records"], 0)
            self.assertIsNone(cell["observed_missing_eos"])


if __name__ == "__main__":
    unittest.main()
