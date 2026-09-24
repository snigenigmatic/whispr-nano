import hashlib
import json
from pathlib import Path

import pytest

from research.bridge_open.analyze_fleurs import _aggregate, _batch_summary, analyze, cluster_bootstrap_difference
from research.bridge_open.scoring import score


def _sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def _fixture(tmp_path, duplicate=False, omit=False):
    manifest, outputs = [], []
    for index, sentence in enumerate(("s1", "s1", "s2")):
        reference = tmp_path / f"r{index}.json"; reference.write_text(json.dumps({"reference_text": "hello world"}))
        manifest.append({"conversation_id": f"id{index}", "language": "hi", "source_sentence_id": sentence, "inference_route": "batched_shortform" if index < 2 else "longform_audio_clock", "audio_sha256": f"a{index}", "reference_sha256": _sha(reference), "reference_path": str(reference)})
        outputs.append({"conversation_id": f"id{index}", "model_id": "whisper-turbo", "model_revision": "rev", "audio_sha256": f"a{index}", "reference_sha256": _sha(reference), "status": "success", "text_raw": "hello world", "metrics": score("hello world", "hello world")})
    if omit: outputs.pop()
    if duplicate: outputs.append(outputs[0])
    manifest_path = tmp_path / "manifest.jsonl"; manifest_path.write_text("".join(json.dumps(x)+"\n" for x in manifest))
    run = tmp_path / "run"; run.mkdir(); (run / "independent_validation.json").write_text('{"valid": true}')
    (run / "predictions.jsonl").write_text("".join(json.dumps(x)+"\n" for x in outputs))
    registry = tmp_path / "registry.json"; registry.write_text(json.dumps([{ "model_id":"whisper-turbo", "model_revision":"rev", "output_dir":str(run)}]))
    return manifest_path, registry

@pytest.mark.parametrize("kw", [{"duplicate": True}, {"omit": True}])
def test_analysis_rejects_duplicate_or_incomplete_cohort(tmp_path, kw):
    manifest, registry = _fixture(tmp_path, **kw)
    with pytest.raises(ValueError): analyze(manifest, registry, tmp_path / "out")

def test_bootstrap_clusters_all_recordings_of_a_sentence(tmp_path):
    rows = [{"conversation_id": str(i), "source_sentence_id": "same" if i < 2 else "other", "metrics": score("a b", "a b")} for i in range(3)]
    result = cluster_bootstrap_difference(rows, rows)
    assert result["cluster_count"] == 2 and result["difference"] == 0 and result["draws"] == 2000

def test_report_retains_full_routes_and_displays_counts(tmp_path):
    manifest, registry = _fixture(tmp_path)
    report = analyze(manifest, registry, tmp_path / "out")
    model = next(iter(report["models"].values()))
    routes = model["per_language"]["hi"]["by_inference_route"]
    assert routes["batched_shortform"]["scored_count"] == 2
    assert routes["longform_audio_clock"]["scored_count"] == 1
    markdown = (tmp_path / "out" / "RESULTS.md").read_text()
    assert "Route-specific micro metrics" in markdown and "0.00%" in markdown

def test_batch_ids_are_unique_per_source_run_not_globally(tmp_path):
    first, second = tmp_path / "first", tmp_path / "second"
    first.mkdir(); second.mkdir()
    event = {"batch_id": "hi-0", "language": "hi", "audio_seconds": 10.0, "inference_wall_seconds": 2.0}
    for directory in (first, second):
        (directory / "batch_inference.jsonl").write_text(json.dumps(event) + "\n")
    summary = _batch_summary([first, second])
    assert summary["unique_batches"] == 2
    assert summary["audio_seconds"] == 20.0 and summary["inference_wall_seconds"] == 4.0

def test_conflicting_duplicate_batch_event_is_rejected(tmp_path):
    run = tmp_path / "run"; run.mkdir()
    (run / "batch_inference.jsonl").write_text(
        '{"batch_id":"hi-0","audio_seconds":10,"inference_wall_seconds":2}\n'
        '{"batch_id":"hi-0","audio_seconds":9,"inference_wall_seconds":2}\n'
    )
    with pytest.raises(ValueError, match="conflicting duplicate"):
        _batch_summary([run])

def test_normalized_empty_diagnostic_overlaps_capped_status():
    metric = score("hello world", "....")
    aggregate = _aggregate([{"status": "generation_limit", "text_raw": "....", "metrics": metric}])
    assert aggregate["generation_limit_count"] == 1
    assert aggregate["normalized_empty_count"] == 1
    assert aggregate["empty_output_status_count"] == 0
