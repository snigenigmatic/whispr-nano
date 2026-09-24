"""Keep the published resource summary tied to the accepted raw profiles."""
from __future__ import annotations

import json
import statistics
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT / "research/bridge_open/core_study"
RESULTS = STUDY / "resource_results.json"
MARKDOWN = STUDY / "RESOURCE_RESULTS.md"
OMNI = ROOT / "research/bridge_open/artifacts/core_resource_omni_v1/resource_profile.json"
SUITE = ROOT / "research/bridge_open/artifacts/core_resource_whisper_qwen_v1"


def _published_model(results: dict, key: str) -> dict:
    return results["omni"] if key == "omni" else results["suite"]["models"][key]


def _profile_paths() -> dict[str, Path]:
    return {
        "omni": OMNI,
        **{
            path.name.removesuffix("_resource_profile.json"): path
            for path in SUITE.glob("*_resource_profile.json")
        },
    }


def test_resource_results_are_derived_from_the_six_accepted_profiles():
    results = json.loads(RESULTS.read_text())
    paths = _profile_paths()
    assert results["valid"] is True
    assert set(paths) == {
        "omni",
        "whisper_small",
        "whisper_turbo",
        "whisper_large_v3",
        "qwen_0_6b",
        "qwen_1_7b",
    }

    for key, path in paths.items():
        profile = json.loads(path.read_text())
        calls = profile["timed_calls"]
        published = _published_model(results, key)
        assert profile["study"]["condition"]["batch_size"] == 1
        assert len(profile["warmup_calls"]) == 12
        assert len(calls) == 36
        assert published["failure_calls"] == profile["summary"]["timed_failed_calls"]
        assert published["peak_cuda_allocated_bytes"] == max(
            row["peak_cuda_allocated_bytes"] for row in calls
        )
        assert published["peak_cuda_reserved_bytes"] == max(
            row["peak_cuda_reserved_bytes"] for row in calls
        )

        expected_passes = []
        for pass_index in range(1, 4):
            rows = [row for row in calls if row["pass_index"] == pass_index]
            expected_passes.append(
                sum(row["end_to_end_inference_seconds"] for row in rows)
                / sum(row["duration_seconds"] for row in rows)
            )
        assert [row["corpus_rtf"] for row in published["passes"]] == pytest.approx(
            expected_passes
        )
        assert published["median_pass_wall_seconds"] == pytest.approx(
            statistics.median(row["wall_seconds"] for row in published["passes"])
        )


def test_published_scope_preserves_the_resource_condition_limitations():
    results = json.loads(RESULTS.read_text())
    interpretation = results["interpretation"]
    assert "batch size 8" in interpretation["qwen_primary_batch8"]
    assert "batch-1" in interpretation["qwen_primary_batch8"]
    assert "unknown" in interpretation["billing"].lower()
    assert "no per-call audio hash" in interpretation["warmup_source_binding"].lower()
    assert "no phone" in interpretation["phone_or_edge"].lower()

    text = MARKDOWN.read_text().lower()
    for phrase in ("batch-one", "batch-eight", "billing is unknown", "warmup", "no phone"):
        assert phrase in text
