import hashlib
import json
import sys
import wave
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "research"))

from bridge_open.scoring import normalize_v1, score
from bridge_open.validate_run import validate


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def make_artifact(tmp_path):
    tmp_path.mkdir(parents=True, exist_ok=True)
    out = tmp_path / "run"
    out.mkdir()
    audio = tmp_path / "audio.wav"
    with wave.open(str(audio), 'wb') as wav:
        wav.setnchannels(1); wav.setsampwidth(2); wav.setframerate(16000)
        wav.writeframes(b'\0\0' * 16000)
    reference = tmp_path / "reference.json"
    reference.write_text(json.dumps({"reference_text": "hello world"}) + "\n")
    manifest = tmp_path / "manifest.jsonl"
    row = {
        "conversation_id": "c1", "language": "hi", "reference_qc": "pass",
        "audio_path": str(audio), "audio_sha256": _sha(audio),
        "reference_path": str(reference), "reference_sha256": _sha(reference),
        "duration_seconds": 1.0,
    }
    manifest.write_text(json.dumps(row) + "\n")
    cfg = {
        "model_id": "fixture/model", "model_revision": "rev1",
        "manifest_path": str(manifest), "manifest_sha256": _sha(manifest),
        "conversation_ids": ["c1"], "chunk_seconds": 30,
    }
    (out / "config.json").write_text(json.dumps(cfg, indent=2) + "\n")
    metrics = score("hello world", "hello world")
    prediction = {
        "conversation_id": "c1", "language": "hi", "model_id": "fixture/model",
        "model_revision": "rev1", "audio_sha256": row["audio_sha256"],
        "reference_sha256": row["reference_sha256"], "status": "success",
        "chunks": [{"start_seconds": 0.0, "end_seconds": 1.0,
                     "text_raw": "hello world", "token_ids": [50257],
                     "eos_present": True}],
        "text_raw": "hello world", "text_for_scoring": normalize_v1("hello world"),
        "metrics": metrics,
    }
    (out / "predictions.jsonl").write_text(json.dumps(prediction) + "\n")
    environment = {"config_sha256": _sha(out / "config.json"),
                   "manifest_sha256": cfg["manifest_sha256"]}
    (out / "environment.json").write_text(json.dumps(environment) + "\n")
    summary = {
        "status": "completed", "attempted": 1, "successful": 1, "scored": 1,
        "per_language": {"hi": {
            "conversation_ids": ["c1"], "scored_count": 1, "attempted_count": 1,
            "word_errors": 0, "ref_words": 2, "character_errors": 0, "ref_chars": 10,
            "micro_wer": 0.0, "micro_cer": 0.0,
            "mean_conversation_wer": 0.0, "mean_conversation_wil": 0.0,
        }},
    }
    (out / "summary.json").write_text(json.dumps(summary) + "\n")
    return out


def test_complete_artifact_is_valid(tmp_path):
    assert validate(make_artifact(tmp_path))["valid"]


def test_resampled_wav_duration_is_authoritative_with_one_sample_tolerance(tmp_path):
    out = make_artifact(tmp_path)
    manifest = tmp_path / "manifest.jsonl"
    row = json.loads(manifest.read_text())
    row["duration_seconds"] = 1.0 - 0.5 / 16000
    manifest.write_text(json.dumps(row) + "\n")
    cfg_path = out / "config.json"
    cfg = json.loads(cfg_path.read_text()); cfg["manifest_sha256"] = _sha(manifest)
    cfg_path.write_text(json.dumps(cfg) + "\n")
    env_path = out / "environment.json"
    env = json.loads(env_path.read_text())
    env.update(config_sha256=_sha(cfg_path), manifest_sha256=_sha(manifest))
    env_path.write_text(json.dumps(env) + "\n")
    assert validate(out)["valid"]
    row["duration_seconds"] = 0.99
    manifest.write_text(json.dumps(row) + "\n")
    cfg["manifest_sha256"] = _sha(manifest); cfg_path.write_text(json.dumps(cfg) + "\n")
    env.update(config_sha256=_sha(cfg_path), manifest_sha256=_sha(manifest))
    env_path.write_text(json.dumps(env) + "\n")
    report = validate(out)
    assert not report["valid"]
    assert any("over one sample" in e for e in report["errors"])


def test_tampered_metrics_are_rejected(tmp_path):
    out = make_artifact(tmp_path)
    path = out / "predictions.jsonl"
    item = json.loads(path.read_text())
    item["metrics"]["wer"] = 1.0
    path.write_text(json.dumps(item) + "\n")
    report = validate(out)
    assert not report["valid"]
    assert any("metric wer" in error for error in report["errors"])


def test_missing_conversation_and_partial_chunks_are_rejected(tmp_path):
    out = make_artifact(tmp_path)
    (out / "config.json").write_text(json.dumps({
        "model_id": "fixture/model", "model_revision": "rev1",
        "manifest_path": str(tmp_path / "manifest.jsonl"),
        "manifest_sha256": _sha(tmp_path / "manifest.jsonl"),
        "conversation_ids": ["c1", "missing"], "chunk_seconds": 30,
    }) + "\n")
    report = validate(out)
    assert not report["valid"]
    assert any("missing from manifest" in error for error in report["errors"])

    out = make_artifact(tmp_path / "partial")
    path = out / "predictions.jsonl"
    item = json.loads(path.read_text())
    item["chunks"][0]["end_seconds"] = 0.5
    path.write_text(json.dumps(item) + "\n")
    report = validate(out)
    assert not report["valid"]
    assert any("chunk range" in error for error in report["errors"])


def test_model_specific_eos_is_used_and_false_termination_rejected(tmp_path):
    out = make_artifact(tmp_path)
    path = out / "config.json"
    cfg = json.loads(path.read_text())
    cfg["eos_token_ids"] = [151643, 151645]
    path.write_text(json.dumps(cfg) + "\n")
    env_path = out / "environment.json"
    env = json.loads(env_path.read_text())
    env["config_sha256"] = _sha(path)
    env_path.write_text(json.dumps(env) + "\n")
    pred_path = out / "predictions.jsonl"
    pred = json.loads(pred_path.read_text())
    pred["chunks"][0]["token_ids"] = [42, 151645]
    pred_path.write_text(json.dumps(pred) + "\n")
    assert validate(out)["valid"]
    pred["chunks"][0]["token_ids"] = [42, 50257]
    pred_path.write_text(json.dumps(pred) + "\n")
    assert not validate(out)["valid"]


def test_ctc_collapse_and_no_eos_contract(tmp_path):
    out = make_artifact(tmp_path)
    path = out / "config.json"
    cfg = json.loads(path.read_text()); cfg["termination_semantics"] = "ctc_no_eos"
    path.write_text(json.dumps(cfg) + "\n")
    env_path = out / "environment.json"
    env = json.loads(env_path.read_text()); env["config_sha256"] = _sha(path)
    env_path.write_text(json.dumps(env) + "\n")
    pred_path = out / "predictions.jsonl"
    pred = json.loads(pred_path.read_text()); pred["termination_semantics"] = "ctc_no_eos"
    pred["chunks"][0].update(ctc_raw_token_ids=[0,0,4,4,0,4], ctc_collapsed_token_ids=[0,4,0,4], eos_present=None)
    pred["chunks"][0].pop("token_ids")
    pred_path.write_text(json.dumps(pred) + "\n")
    assert validate(out)["valid"]
    pred["chunks"][0]["ctc_collapsed_token_ids"] = [4,4]
    pred_path.write_text(json.dumps(pred) + "\n")
    assert not validate(out)["valid"]


def test_capped_punctuation_only_output_is_scored_not_rejected(tmp_path):
    out = make_artifact(tmp_path)
    pred_path = out / "predictions.jsonl"
    pred = json.loads(pred_path.read_text())
    raw = "...."
    pred.update(status="generation_limit", text_raw=raw, text_for_scoring="", metrics=score("hello world", raw))
    pred["chunks"][0].update(text_raw=raw, token_ids=[42], generation_sequences=[[42]], eos_present=False)
    pred_path.write_text(json.dumps(pred) + "\n")
    summary_path = out / "summary.json"
    summary = json.loads(summary_path.read_text())
    summary.update(status="completed_with_quality_flags", successful=0, scored=1)
    summary["per_language"]["hi"].update(word_errors=2, character_errors=10, micro_wer=1.0, micro_cer=1.0,
                                             mean_conversation_wer=1.0, mean_conversation_wil=1.0)
    summary_path.write_text(json.dumps(summary) + "\n")
    assert validate(out)["valid"]
    pred["status"] = "success"
    pred_path.write_text(json.dumps(pred) + "\n")
    report = validate(out)
    assert not report["valid"]
    assert any("success has empty normalized text" in error for error in report["errors"])
