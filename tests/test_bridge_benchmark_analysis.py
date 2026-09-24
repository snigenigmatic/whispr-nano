import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "research"))
from bridge_open.analyze_benchmark import _aggregate, analyze


def _run(root, name, ids=("a", "b", "c", "d", "e"), model="m1", delta=0, condition=None):
    out = root / name; out.mkdir()
    manifest = root / (name + "-manifest.jsonl")
    manifest.write_text("".join(json.dumps({"conversation_id": i, "released_language": "Hindi", "language": "hi", "reference_qc": "pass"}) + "\n" for i in ids))
    cfg = {"model_id": model, "model_revision": "r1", "condition_id": condition or name, "manifest_path": str(manifest), "manifest_sha256": "manifest-shared", "conversation_ids": list(ids), "device": "cpu", "batch_size": 1, "num_beams": 1, "chunk_seconds": 30, "seed": 12, "metric_protocol": "open-v1"}
    (out / "config.json").write_text(json.dumps(cfg))
    (out / "environment.json").write_text(json.dumps({"dtype": "float32", "code_sha256": {"scoring.py": "score-hash"}}))
    rows = []
    for i in ids:
        wer = 0.2 + delta
        rows.append({"conversation_id": i, "language": "hi", "status": "success", "metrics": {"wer": wer, "cer": wer, "substitutions": 1, "deletions": 0, "insertions": 0, "ref_words": 5, "ref_chars": 5, "character_errors": 1}, "chunks": [{"eos_present": True}], "real_time_factor": 0.1})
    (out / "predictions.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    return out


def test_aggregate_and_shared_bootstrap_fixture(tmp_path):
    result = analyze([_run(tmp_path, "a"), _run(tmp_path, "b", model="m2", delta=-0.1)])
    assert result["protocol_counts"] == {"rows": 5, "languages": 1, "qc_pass_rows": 5, "qc_pass_languages": 1}
    assert result["runs"][0]["summary"]["micro_wer"] == pytest.approx(0.2)
    assert result["comparisons"][0]["common_cohort"]["n_conversations"] == 5
    assert result["comparisons"][0]["common_cohort"]["wer_delta_b_minus_a"] == pytest.approx(-0.1)


def test_duplicate_identity_refuses_and_small_cohort_suppresses_inference(tmp_path):
    first = _run(tmp_path, "a")
    with pytest.raises(ValueError, match="duplicate"):
        analyze([first, first])
    one = _run(tmp_path, "one", ids=("x",), model="m3")
    result = analyze([one, _run(tmp_path, "two", ids=("x",), model="m4")])
    assert "inference_suppressed" in result["comparisons"][0]["common_cohort"]


def test_disjoint_shards_merge_and_recipe_mismatch_refuses(tmp_path):
    a = _run(tmp_path, "a", ids=("a", "b"), condition="fixed")
    b = _run(tmp_path, "b", ids=("c", "d"), condition="fixed")
    result = analyze([a, b])
    assert len(result["runs"]) == 1
    assert result["runs"][0]["summary"]["attempted"] == 4
    (b / "environment.json").write_text(json.dumps({"dtype": "float16"}))
    with pytest.raises(ValueError, match="mismatched recipe"):
        analyze([a, b])
    (b / "environment.json").write_text(json.dumps({"dtype": "float32", "code_sha256": {"scoring.py": "score-hash"}}))
    cfg = json.loads((b / "config.json").read_text()); cfg["metric_protocol"] = "changed"
    (b / "config.json").write_text(json.dumps(cfg))
    with pytest.raises(ValueError, match="mismatched recipe"):
        analyze([a, b])


def test_ctc_null_eos_is_not_generation_capping(tmp_path):
    run = _run(tmp_path, "ctc")
    cfg = json.loads((run / "config.json").read_text()); cfg["termination_semantics"] = "ctc_no_eos"
    (run / "config.json").write_text(json.dumps(cfg))
    rows = [json.loads(line) for line in (run / "predictions.jsonl").read_text().splitlines()]
    for row in rows:
        row["chunks"][0]["eos_present"] = None
    (run / "predictions.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    assert analyze([run])["runs"][0]["summary"]["capped_chunks"] == 0


def test_all_failure_language_retained_and_macro_is_language_micro(tmp_path):
    rows = [
        {"language": "hi", "status": "success", "metrics": {"wer": .2, "cer": .2, "substitutions": 1, "deletions": 0, "insertions": 0, "ref_words": 100, "ref_chars": 100, "character_errors": 1}, "chunks": []},
        {"language": "bn", "status": "inference_error", "metrics": {}, "chunks": []},
        {"language": "bn", "status": "inference_error", "metrics": {}, "chunks": []},
    ]
    for i, row in enumerate(rows): row["conversation_id"] = str(i)
    result = _aggregate(rows)
    assert result["languages"]["bn"]["micro_wer"] is None
    assert result["languages"]["bn"]["failed"] == 2
    assert result["macro_language_wer"] == pytest.approx(.01)


def test_bootstrap_has_no_invalid_pvalues(tmp_path):
    result = analyze([_run(tmp_path, "a"), _run(tmp_path, "b", model="m2", delta=-.1)])
    comparison = result["comparisons"][0]["common_cohort"]
    assert "wer_p_two_sided" not in comparison
    assert "cer_p_two_sided" not in comparison
