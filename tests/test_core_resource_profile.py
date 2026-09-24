import hashlib
import json

import numpy as np
import pytest

from research.bridge_open.core_study import resource_profile as profile


class FakeCuda:
    def __init__(self): self.events = []
    def synchronize(self): self.events.append("sync")
    def memory_allocated(self): self.events.append("allocated"); return 100
    def memory_reserved(self): self.events.append("reserved"); return 200
    def reset_peak_memory_stats(self): self.events.append("reset")
    def max_memory_allocated(self): self.events.append("peak_allocated"); return 300
    def max_memory_reserved(self): self.events.append("peak_reserved"); return 400
    def environment(self):
        return {"gpu_name": "fake-gpu", "device_properties": {"total_memory_bytes": 1},
                "torch_version": "fake-torch", "cuda_version": "fake-cuda"}


class Clock:
    def __init__(self): self.value = 10.0
    def __call__(self): return self.value
    def advance(self, seconds): self.value += seconds


def inputs():
    return tuple(profile.PreloadedAudio(cid, np.array([0.0, 0.25], dtype=np.float32), 16000)
                 for cid in profile.FIXED_PANEL_IDS)


def metadata():
    return ({"model_id": "unit-model", "model_hashes": {"weights": "abc"}, "parameter_count": 1,
             "declared_precision": "float32", "declared_search": {"num_beams": 1},
             "resident_before_measurement": True}, {"batch_size": 1, "frontend": "unit", "decoder": "greedy"})


def test_profile_keeps_failure_and_uses_one_callback_per_phase():
    cuda, clock, calls = FakeCuda(), Clock(), []
    def transcribe(array, sample_rate):
        calls.append(sample_rate); cuda.events.append("call"); clock.advance(0.25)
        if len(calls) == 14: raise RuntimeError("planned callback failure")
        return "result"
    model, pipeline = metadata()
    result = profile.measure_resource_profile(inputs(), transcribe, model, pipeline, cuda=cuda, clock=clock)
    assert len(calls) == 48
    assert result["summary"]["observed_timed_calls"] == 36
    assert result["summary"]["failed_calls_including_warmup"] == 1
    assert result["summary"]["timed_failed_calls"] == 1
    assert result["timed_calls"][1]["failure"]["type"] == "RuntimeError"
    first = result["timed_calls"][0]
    assert first["end_to_end_inference_seconds"] == pytest.approx(0.25)
    assert first["real_time_factor"] == pytest.approx(2000.0)
    assert first["text_sha256"] == hashlib.sha256(b"result").hexdigest()
    assert first["resident_baseline_memory_bytes"] == {"allocated_bytes": 100, "reserved_bytes": 200}
    assert first["peak_cuda_allocated_bytes"] == 300
    assert cuda.events[:2] == ["call", "sync"]
    assert cuda.events[24:31] == ["sync", "allocated", "reserved", "reset", "call", "sync", "peak_allocated"]
    assert cuda.events.count("sync") == 12 + 36 * 2
    assert cuda.events.count("reset") == 36


def test_profile_rejects_missing_or_reordered_panel_row_before_callback():
    model, pipeline = metadata()
    with pytest.raises(ValueError, match="every fixed-panel row"):
        profile.measure_resource_profile(inputs()[:-1], lambda *_: "unexpected", model, pipeline, cuda=FakeCuda())


def test_preload_reads_the_verified_bundle_not_the_source_relative_path():
    loaded = profile.preload_fixed_panel()
    assert tuple(item.conversation_id for item in loaded) == profile.FIXED_PANEL_IDS
    assert {item.sample_rate for item in loaded} == {16000}


def test_profile_rejects_wrong_sample_rate_before_callback():
    model, pipeline = metadata()
    bad = list(inputs()); bad[3] = profile.PreloadedAudio(bad[3].conversation_id, bad[3].array, 8000)
    with pytest.raises(ValueError, match="expected sample rate 16000"):
        profile.measure_resource_profile(tuple(bad), lambda *_: "unexpected", model, pipeline, cuda=FakeCuda())


def test_export_is_create_only_and_has_schema(tmp_path):
    cuda, clock = FakeCuda(), Clock()
    model, pipeline = metadata()
    result = profile.measure_resource_profile(inputs(), lambda *_: "ok", model, pipeline, cuda=cuda, clock=clock)
    path = profile.export_standard_artifact(result, "unit-run", tmp_path)
    payload = json.loads(path.read_text())
    assert payload["schema_version"] == profile.SCHEMA_VERSION
    assert payload["study"]["conversation_ids"] == list(profile.FIXED_PANEL_IDS)
    with pytest.raises(ValueError, match="already exists"):
        profile.export_standard_artifact(result, "unit-run", tmp_path)
