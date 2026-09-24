import hashlib
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

import research.bridge_open.run_whisper_batched as batched
from research.bridge_open.run_whisper_batched import checked_audio, effective_config, group_by_language


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _row(tmp_path, samples, sample_rate=16_000):
    audio_path = tmp_path / "clip.wav"
    reference_path = tmp_path / "reference.json"
    sf.write(audio_path, samples, sample_rate)
    reference_path.write_text('{"reference_text": "must remain unread"}')
    return {"conversation_id": "id", "audio_path": str(audio_path), "audio_sha256": _sha(audio_path),
            "reference_path": str(reference_path), "reference_sha256": _sha(reference_path)}


def test_grouping_is_stable_and_never_mixes_language_hints():
    rows = [{"conversation_id": "b1", "language": "bn"}, {"conversation_id": "h1", "language": "hi"},
            {"conversation_id": "b2", "language": "bn"}]
    assert group_by_language(rows) == [("bn", [rows[0], rows[2]]), ("hi", [rows[1]])]


def test_checked_audio_rejects_overlong_nonfinite_and_silent_inputs_before_model_cost(tmp_path):
    cfg = {"chunk_seconds": 30}
    with pytest.raises(ValueError, match="at most 30"):
        checked_audio(_row(tmp_path, np.ones(480_001, dtype=np.float32)), cfg)
    with pytest.raises(ValueError, match="all zero"):
        checked_audio(_row(tmp_path, np.zeros(100, dtype=np.float32)), cfg)
    row = _row(tmp_path, np.ones(100, dtype=np.float32))
    original_read = batched.sf.read
    batched.sf.read = lambda *args, **kwargs: (np.array([np.nan], dtype=np.float32), 16_000)
    try:
        with pytest.raises(ValueError, match="non-finite"):
            checked_audio(row, cfg)
    finally:
        batched.sf.read = original_read


def test_checked_audio_hashes_but_never_reads_reference_text_before_generation(tmp_path, monkeypatch):
    row = _row(tmp_path, np.ones(100, dtype=np.float32))
    reference = Path(row["reference_path"])
    original = Path.read_text

    def forbid_reference_text(self, *args, **kwargs):
        if self == reference:
            raise AssertionError("reference text opened during preflight")
        return original(self, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", forbid_reference_text)
    audio, sample_rate = checked_audio(row, {"chunk_seconds": 30})
    assert sample_rate == 16_000 and len(audio) == 100


def test_reference_text_is_opened_only_after_raw_batch_prediction_is_installed(tmp_path, monkeypatch):
    row = _row(tmp_path, np.ones(100, dtype=np.float32))
    record = {"chunks": [{"text_raw": "prediction", "eos_present": True}], "duration_seconds": 1.0}
    original = Path.read_text

    def sequenced_read(self, *args, **kwargs):
        if self == Path(row["reference_path"]):
            assert record["chunks"], "reference opened before batch generation"
        return original(self, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", sequenced_read)
    batched._finish_prediction(record, row, batched.time.monotonic())
    assert record["status"] == "success" and "metrics" in record


def test_effective_config_commits_batched_beam5_no_timestamp_defaults():
    cfg = effective_config({})
    assert (cfg["batch_size"], cfg["num_beams"], cfg["return_timestamps"]) == (8, 5, False)
