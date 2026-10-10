import csv
import json

from scripts.termeval import score_results


def test_scoring_pipeline_writes_valid_artifacts(monkeypatch, tmp_path):
    manifest = {}
    predictions = []
    annotations = {}
    for dataset in ("mucs_hien", "hiacc"):
        for i in range(100):
            utt = f"{dataset}_{i}"
            manifest[utt] = {"dataset": dataset, "utt_id": utt,
                             "speaker": f"{dataset}_s{i % 10}", "reference": "python code",
                             "group": "adult" if i < 50 else "children",
                             "gender": "F" if i % 2 else "M"}
            predictions.append({"dataset": dataset, "utt_id": utt,
                                "system": "whisper-large-v3", "model_revision": score_results.MODEL_REVISION,
                                "config_hash": dataset + "_config", "text": "code" if i % 3 == 0 else "python code"})
            annotations[utt] = {"utt_id": utt, "spans": [{"start": 0, "end": 1, "label": "TERM"}]}
    monkeypatch.setattr(score_results, "OUT", tmp_path)
    monkeypatch.setattr(score_results, "load_inputs", lambda: (manifest, predictions, annotations))
    score_results.main()
    assert json.loads((tmp_path / "score_receipt.json").read_text())["predictions"] == 200
    assert (tmp_path / "tokens_hiacc_whisper-large-v3_A_raw.parquet").exists()
    with (tmp_path / "metrics.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    assert any(r["dataset"] == "mucs_hien" and r["metric"] == "WER" for r in rows)
    assert any(r["dataset"] == "hiacc" and r["metric"] == "TermER" for r in rows)
