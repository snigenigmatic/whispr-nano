"""Mocked CPU integration tests for the real Qwen runner entry point."""
import importlib
import json
import sys
import tempfile
import types
import unittest
import wave
from array import array
from pathlib import Path
from unittest.mock import patch


class _Audio:
    ndim = 1

    def __init__(self, peak):
        self.peak = peak
        self.size = 16000

    def __len__(self):
        return self.size


class _Finite:
    def all(self):
        return True


class _Ids:
    def __init__(self, count): self.shape = (count, 3)


class _Sequences:
    def __init__(self, count): self.count = count
    def detach(self): return self
    def cpu(self): return self
    def tolist(self): return [[10, 11, 12, 42, 151643] for _ in range(self.count)]


class _Random:
    def seed(self, _):
        pass


def _install_stubs(audio, results):
    numpy = types.ModuleType("numpy")
    numpy.float32 = "float32"
    numpy.random = _Random()
    numpy.isfinite = lambda _: _Finite()
    numpy.abs = lambda x: x
    numpy.max = lambda x: x.peak

    soundfile = types.ModuleType("soundfile")
    soundfile.read = lambda *_args, **_kwargs: (audio, 16000)
    soundfile.info = lambda *_args, **_kwargs: types.SimpleNamespace(samplerate=16000, channels=1, frames=16000)

    class _Cuda:
        def is_available(self): return True
        def synchronize(self): pass
        def max_memory_allocated(self): return 0

    class _Torch:
        __version__ = "2.8.0"
        bfloat16 = "torch.bfloat16"
        float16 = "torch.float16"
        float32 = "torch.float32"
        cuda = _Cuda()
        def manual_seed(self, _): pass
        def set_num_threads(self, _): pass

    torch = _Torch()
    transformers = types.ModuleType("transformers")
    transformers.__version__ = "4.57.6"
    hub = types.ModuleType("huggingface_hub")
    hub.snapshot_download = lambda *_args, **_kwargs: str(_install_stubs.snapshot)

    class _Param:
        def numel(self): return 1

    class _Model:
        dtype = "torch.bfloat16"
        training = False
        generation_config = types.SimpleNamespace(do_sample=False, num_beams=1,
            to_dict=lambda: {'do_sample': False, 'num_beams': 1})
        def eval(self): return self
        def parameters(self): return [_Param()]
        def generate(self, input_ids=None, **_kwargs): return _Sequences(input_ids.shape[0])

    class _Result:
        def __init__(self, text): self.text = text; self.language = "Hindi"

    class _Qwen:
        model = _Model()
        processor = types.SimpleNamespace(batch_decode=lambda values, **_kwargs: ["raw-before-parser"] * len(values))
        def __init__(self): self.calls = 0
        def transcribe(self, *_args, **_kwargs):
            self.calls += 1
            self.model.generate(input_ids=_Ids(len(_args[0])))
            return results
        @classmethod
        def from_pretrained(cls, *_args, **_kwargs): return cls()

    qwen_asr = types.ModuleType("qwen_asr")
    qwen_asr.Qwen3ASRModel = _Qwen
    jiwer = types.ModuleType("jiwer")
    def proc(a, b):
        aa, bb = a.split(), b.split()
        return types.SimpleNamespace(wer=0.0 if aa == bb else 1.0, cer=0.0 if aa == bb else 1.0,
            wil=0.0, hits=len(aa) if aa == bb else 0,
            substitutions=0 if aa == bb else (0 if not bb else len(aa)),
            deletions=len(aa) if not bb else 0, insertions=len(bb) if not aa else 0)
    jiwer.process_words = proc; jiwer.process_characters = proc
    modules = {"numpy": numpy, "soundfile": soundfile, "torch": torch,
        "transformers": transformers, "huggingface_hub": hub, "qwen_asr": qwen_asr, "jiwer": jiwer}
    return modules


class NativeRunIntegrationTest(unittest.TestCase):
    def run_case(self, text, audio_peak=0.25, result_count=1, input_count=1):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            audio_path, ref_path, manifest = root / "x.wav", root / "x.ref.json", root / "manifest.jsonl"
            with wave.open(str(audio_path), 'wb') as wav:
                wav.setnchannels(1); wav.setsampwidth(2); wav.setframerate(16000)
                wav.writeframes(array('h', [8192] * 16000).tobytes())
            ref_path.write_text(json.dumps({"reference_text": "namaste duniya"}))
            import hashlib
            digest = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
            row = {"conversation_id": "x", "language": "hi", "reference_qc": "pass",
                "duration_seconds": 1.0,
                "audio_path": str(audio_path), "audio_sha256": digest(audio_path),
                "reference_path": str(ref_path), "reference_sha256": digest(ref_path)}
            rows = [dict(row, conversation_id=chr(ord('x') + i)) for i in range(input_count)]
            manifest.write_text("".join(json.dumps(item) + "\n" for item in rows))
            snapshot = root / "snapshot"; snapshot.mkdir()
            (snapshot / "generation_config.json").write_text(json.dumps({"eos_token_id": [151643, 151645]}))
            (snapshot / "weights.safetensors").write_bytes(b"weights")
            result = types.SimpleNamespace(text=text, language="Hindi")
            results = [result] * result_count
            modules = _install_stubs(_Audio(audio_peak), results)
            _install_stubs.snapshot = snapshot
            with patch.dict(sys.modules, modules), patch("importlib.metadata.version", return_value="4.57.6"), patch("importlib.metadata.files", return_value=[]):
                sys.modules.pop("research.bridge_open.run_qwen", None)
                runner = importlib.import_module("research.bridge_open.run_qwen")
                cfg = {"model_id": "Qwen/test", "model_revision": "rev", "manifest_path": str(manifest),
                    "manifest_sha256": digest(manifest), "conversation_ids": [item["conversation_id"] for item in rows],
                    "source_manifest_sha256": digest(manifest), "supported_languages": ["hi"],
                    "device": "cuda", "device_map": "cuda:0", "torch_dtype": "bfloat16",
                    "seed": 1, "batch_size": input_count, "chunk_seconds": 30, "max_new_tokens": 8, "max_wall_seconds": 30,
                    "num_beams": 1, "return_timestamps": False, "eos_token_ids": [151643, 151645],
                    "inference_api": "Qwen3ASRModel.transcribe", "termination_semantics": "qwen_transcribe_api",
                    "output_dir": str(root / "out"),
                    "executor": "local"}
                config = root / "config.json"; config.write_text(json.dumps(cfg, indent=2) + "\n")
                code = runner.run(config)
                predictions = [json.loads(line) for line in (root / "out/predictions.jsonl").read_text().splitlines()]
                from research.bridge_open.validate_run import validate
                return code, predictions, root / "out", validate(root / "out")

    def test_real_runner_scores_empty_output(self):
        code, predictions, _, report = self.run_case("")
        prediction = predictions[0]
        self.assertEqual(code, 0)
        self.assertTrue(report["valid"], report)
        self.assertEqual(prediction["status"], "empty_output")
        self.assertEqual(prediction["metrics"]["deletions"], 2)
        self.assertEqual(prediction["chunks"][0]["token_evidence"], "captured_from_generate")
        self.assertEqual(prediction["chunks"][0]["generation_trace"][0]["raw_decoded_text"], "raw-before-parser")

    def test_real_runner_maps_positive_two_item_batch(self):
        code, predictions, _, report = self.run_case("namaste duniya", result_count=2, input_count=2)
        prediction = predictions[0]
        self.assertEqual(code, 0)
        self.assertTrue(report["valid"], report)
        self.assertEqual(prediction["inference_group_id"], "group-0000")
        self.assertEqual(prediction["group_size"], 2)
        self.assertEqual(prediction["chunks"][0]["generation_trace"][0]["record_index"], 0)

    def test_real_runner_rejects_result_cardinality(self):
        code, predictions, _, _ = self.run_case("namaste duniya", result_count=2)
        prediction = predictions[0]
        self.assertEqual(code, 1)
        self.assertEqual(prediction["status"], "inference_error")
        self.assertIn("cardinality mismatch", prediction["failure"]["message"])

    def test_real_runner_rejects_silent_audio_before_api(self):
        with self.assertRaisesRegex(ValueError, "silent"):
            self.run_case("namaste duniya", audio_peak=0.0)


if __name__ == "__main__":
    unittest.main()
