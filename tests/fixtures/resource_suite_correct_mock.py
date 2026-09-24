"""Preserved corrected mock coverage from the five-model resource-suite launch.

This remains a fixture rather than an automatically collected main-repository test because
it intentionally imports the frozen, archived Qwen worktree used by the launch.  It is
kept verbatim in substance so that the original CPU-only preflight and adapter contract
can be recovered without modifying that worktree.
"""

import json
from pathlib import Path

import pytest

from research.bridge_open.core_study import run_resource_suite as suite
from research.bridge_open.core_study.resource_profile import FIXED_PANEL_IDS


ROOT = Path("/Users/prateekraj/whispr-fleurs-qwen")


def config():
    return json.loads((ROOT / "research/bridge_open/core_study/resource_suite_config.json").read_text())


def fake_profile(*_args, **_kwargs):
    timed = []
    for pass_index in range(1, 4):
        timed.extend(
            {
                "conversation_id": cid,
                "pass_index": pass_index,
                "status": "success",
                "text": "unit text",
                "text_sha256": "x",
                "failure": None,
            }
            for cid in FIXED_PANEL_IDS
        )
    return {
        "timed_calls": timed,
        "warmup_calls": [{}] * 12,
        "summary": {"timed_failed_calls": 0},
        "model_residency": {"parameter_count": 1},
        "environment": {"gpu_name": "mock"},
    }


class Parameter:
    dtype = suite.torch.float32

    def numel(self):
        return 1


class Model:
    training = False

    def parameters(self):
        return iter([Parameter()])


def test_cpu_preflight_pins_all_five_roster_revisions():
    manifest, rows = suite.validate_config(config())
    assert manifest.name == "manifest.jsonl"
    assert len(rows) == 12


def test_cpu_preflight_rejects_wrong_qwen_language_and_roster_revision():
    cfg = config()
    cfg["models"][3]["language"] = "Hindi"
    with pytest.raises(ValueError, match="explicit English"):
        suite.validate_config(cfg)
    cfg = config()
    cfg["models"][0]["revision"] = "bad"
    with pytest.raises(ValueError, match="differs from roster"):
        suite.validate_config(cfg)


def test_mock_adapters_write_every_model_and_keep_quality_to_pass_three(tmp_path, monkeypatch):
    cfg = config()
    cfg["output_dir"] = str(tmp_path / "result")
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(cfg))
    monkeypatch.setattr(suite.importlib.metadata, "version", lambda name: "mock-version")
    calls = []

    def loader(spec):
        calls.append(spec["key"])
        return (
            lambda _array, _sr: "unit text",
            Model(),
            {"weights": "unit"},
            {"api": "mock", "search": {}},
        )

    monkeypatch.setattr(suite, "_load_whisper", loader)
    monkeypatch.setattr(suite, "_load_qwen", loader)
    monkeypatch.setattr(suite, "measure_resource_profile", fake_profile)
    monkeypatch.setattr(suite.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(suite.torch.cuda, "empty_cache", lambda: None)
    monkeypatch.setattr(suite.torch.cuda, "synchronize", lambda: None)
    out = suite.run(config_path)
    assert calls == [model["key"] for model in cfg["models"]]
    summary = json.loads((out / "overall_summary.json").read_text())
    assert [row["status"] for row in summary["models"]] == ["completed"] * 5
    quality = [
        json.loads(line)
        for line in (out / "qwen_1_7b_quality_predictions.jsonl").read_text().splitlines()
    ]
    assert [row["conversation_id"] for row in quality] == list(FIXED_PANEL_IDS)
    assert {row["timed_pass_index"] for row in quality} == {3}
