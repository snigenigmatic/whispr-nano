"""Consolidate accepted core-study quality cells from frozen prediction evidence.

The consolidator deliberately has no model-ranking logic.  It independently
joins every prediction to the public frozen source manifests and re-scores its
raw transcript before reporting descriptive, per-cell quality statistics.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from research.bridge_open.scoring import normalize_v1, score


ROOT = Path(__file__).resolve().parents[3]
FLEURS_MANIFEST = Path("data/bridge_open/fleurs_test_v1/corpus-v2.jsonl")
SVARAH_MANIFEST = Path("data/bridge_open/svarah_native_shard02_full_v1/bundle/manifest.jsonl")
MANIFEST_SHA256 = {
    FLEURS_MANIFEST: "967d8dca02f9f729c1ed7548dce7283cf9dfc62623815a7c131f71acb13ce2c7",
    SVARAH_MANIFEST: "dbddbaeef3a1f78ff85d0cd741e85d86daf7c868b39ee8d4c48f67a21bfafedd",
}
MODEL_PINS = {
    "whisper-small": ("openai/whisper-small", "973afd24965f72e36ca33b3055d56a652f456b4d"),
    "whisper-turbo": ("openai/whisper-large-v3-turbo", "41f01f3fe87f28c78e2fbf8b568835947dd65ed9"),
    "whisper-large-v3": ("openai/whisper-large-v3", "06f233fe06e710322aca913c1bc4249a0d71fce1"),
    "qwen3-asr-0.6b": ("Qwen/Qwen3-ASR-0.6B", "5eb144179a02acc5e5ba31e748d22b0cf3e303b0"),
    "qwen3-asr-1.7b": ("Qwen/Qwen3-ASR-1.7B", "7278e1e70fe206f11671096ffdd38061171dd6e5"),
    "omniasr-ctc-1b": ("facebook/omniASR-CTC-1B", "8c22e3ffdaa4aab6431b128b84b991a7d9c2515c"),
}
TERMINAL_STATUSES = {"success", "empty_output", "generation_limit", "native_api_completed"}


def _cell(model: str, track: str, records: int, artifact: str, prediction: str = "predictions.jsonl",
          acceptance: str = "local_independent_acceptance.json", receipt_kind: str = "valid") -> dict[str, Any]:
    return {"model": model, "track": track, "records": records, "artifact": artifact,
            "prediction": prediction, "acceptance": acceptance, "receipt_kind": receipt_kind}


# This is the 20-cell supported matrix.  Qwen 0.6B's Bengali/Assamese cells
# are intentionally omitted because the frozen study marks them unsupported.
CELL_SPECS = [
    _cell("whisper-small", "fleurs-hi", 418, "core_whisper_small_hi_v1", acceptance="local_independent_whisper_core_acceptance.json"),
    _cell("whisper-small", "fleurs-bn", 920, "core_whisper_small_bn_v1", acceptance="local_independent_whisper_core_acceptance.json"),
    _cell("whisper-small", "fleurs-as", 984, "core_whisper_small_as_v1", acceptance="local_independent_whisper_core_acceptance.json"),
    _cell("whisper-small", "svarah-recovered-en", 518, "core_whisper_small_svarah_v1", acceptance="local_independent_whisper_core_acceptance.json"),
    _cell("whisper-turbo", "fleurs-hi", 418, "fleurs_whisper_corrected_v3", "turbo_predictions.jsonl", "analysis_receipt.json", "whisper_prior"),
    _cell("whisper-turbo", "fleurs-bn", 920, "fleurs_whisper_corrected_v3", "turbo_predictions.jsonl", "analysis_receipt.json", "whisper_prior"),
    _cell("whisper-turbo", "fleurs-as", 984, "fleurs_whisper_corrected_v3", "turbo_predictions.jsonl", "analysis_receipt.json", "whisper_prior"),
    _cell("whisper-turbo", "svarah-recovered-en", 518, "core_whisper_turbo_svarah_v1", acceptance="local_independent_whisper_core_acceptance.json"),
    _cell("whisper-large-v3", "fleurs-hi", 418, "fleurs_whisper_corrected_v3", "large-v3_predictions.jsonl", "analysis_receipt.json", "whisper_prior"),
    _cell("whisper-large-v3", "fleurs-bn", 920, "fleurs_whisper_corrected_v3", "large-v3_predictions.jsonl", "analysis_receipt.json", "whisper_prior"),
    _cell("whisper-large-v3", "fleurs-as", 984, "fleurs_whisper_corrected_v3", "large-v3_predictions.jsonl", "analysis_receipt.json", "whisper_prior"),
    _cell("whisper-large-v3", "svarah-recovered-en", 518, "core_whisper_large_svarah_v1", acceptance="local_independent_whisper_core_acceptance.json"),
    _cell("qwen3-asr-0.6b", "fleurs-hi", 418, "core_qwen06_hindi_v1"),
    _cell("qwen3-asr-0.6b", "svarah-recovered-en", 518, "core_qwen06_svarah_v1"),
    _cell("qwen3-asr-1.7b", "fleurs-hi", 418, "qwen_native_full_hi_v1", acceptance="independent_validation.json"),
    _cell("qwen3-asr-1.7b", "svarah-recovered-en", 518, "qwen_svarah_full_v1"),
    _cell("omniasr-ctc-1b", "fleurs-hi", 418, "fleurs_omni_comparison_v1", receipt_kind="omni_prior"),
    _cell("omniasr-ctc-1b", "fleurs-bn", 920, "fleurs_omni_comparison_v1", receipt_kind="omni_prior"),
    _cell("omniasr-ctc-1b", "fleurs-as", 984, "fleurs_omni_comparison_v1", receipt_kind="omni_prior"),
    _cell("omniasr-ctc-1b", "svarah-recovered-en", 518, "omni_svarah_full_v1"),
]

# Immutable execution contracts.  The prior prediction hashes are intentionally
# copied from the preserved evidence inventory, rather than accepted merely
# because a file happens to be present today.
CELL_CONTRACTS = {
    ("whisper-small", "fleurs-hi"): {"commit": "949a23078e4b15745aa752a373982f74634997e0"},
    ("whisper-small", "fleurs-bn"): {"commit": "792092deb896ea96f1a576e536255fc91fa45955"},
    ("whisper-small", "fleurs-as"): {"commit": "d1051672a100f4791bdfb9448d90db42bb3e2e8e"},
    ("whisper-small", "svarah-recovered-en"): {"commit": "31904891563d5953195dae3d4e7e5451570f421a"},
    ("whisper-turbo", "fleurs-hi"): {"commit": "4eb3cdafc6eb4d519bd1a8faeaf285cc6b9375a6", "prediction_sha256": "127c4ba4854100b964c40b61701457f4972cc0bc0ca684ce464ec1b6cec5b988"},
    ("whisper-turbo", "fleurs-bn"): {"commit": "4eb3cdafc6eb4d519bd1a8faeaf285cc6b9375a6", "prediction_sha256": "127c4ba4854100b964c40b61701457f4972cc0bc0ca684ce464ec1b6cec5b988"},
    ("whisper-turbo", "fleurs-as"): {"commit": "4eb3cdafc6eb4d519bd1a8faeaf285cc6b9375a6", "prediction_sha256": "127c4ba4854100b964c40b61701457f4972cc0bc0ca684ce464ec1b6cec5b988"},
    ("whisper-turbo", "svarah-recovered-en"): {"commit": "ca7021949517f031dd75e3dfb35cb627cd4a6232"},
    ("whisper-large-v3", "fleurs-hi"): {"commit": "4eb3cdafc6eb4d519bd1a8faeaf285cc6b9375a6", "prediction_sha256": "a23d1fdfd2152e1f520a8527536b71abbc087cd4ee28c4f312423119ae02d82d"},
    ("whisper-large-v3", "fleurs-bn"): {"commit": "4eb3cdafc6eb4d519bd1a8faeaf285cc6b9375a6", "prediction_sha256": "a23d1fdfd2152e1f520a8527536b71abbc087cd4ee28c4f312423119ae02d82d"},
    ("whisper-large-v3", "fleurs-as"): {"commit": "4eb3cdafc6eb4d519bd1a8faeaf285cc6b9375a6", "prediction_sha256": "a23d1fdfd2152e1f520a8527536b71abbc087cd4ee28c4f312423119ae02d82d"},
    ("whisper-large-v3", "svarah-recovered-en"): {"commit": "0cbbecdb2f3a3ef37dc963ef57d3378ad8f73295"},
    ("qwen3-asr-0.6b", "fleurs-hi"): {"commit": "9886373a37a01a41bad3eb15eb546ced3a7391b5"},
    ("qwen3-asr-0.6b", "svarah-recovered-en"): {"commit": "91257030585090522739144688e4c1b99fcf68ef"},
    ("qwen3-asr-1.7b", "fleurs-hi"): {"commit": "6df7cc65424f2108c33a8354d37262e679b9f6b8", "prediction_sha256": "2ac059fd13ccd1ad5c4f92e824022144c91bdf5eca54c751f662b1f50641e5aa"},
    ("qwen3-asr-1.7b", "svarah-recovered-en"): {"commit": "80ec0e80bf994e8528bd0032efe0cf64fb612e2f", "prediction_sha256": "f107e20946efdac2e05b4486f72c6b4c1242ad5ad5ca88caa86986a4ccf9e026"},
    ("omniasr-ctc-1b", "fleurs-hi"): {"commit": "3012771c213b1a9853b0b6214c48577342f1c9f4", "prediction_sha256": "3b00dd8c035aace576c50cd1d921c717b920d2688457a2b748742bbdef17e9e7", "receipt_artifact": "omni_fleurs_full_hi_v1"},
    ("omniasr-ctc-1b", "fleurs-bn"): {"commit": "f9f0a02cbac1c90e5a5228aeeb951039a3e774b2", "prediction_sha256": "3b00dd8c035aace576c50cd1d921c717b920d2688457a2b748742bbdef17e9e7", "receipt_artifact": "omni_fleurs_full_bn_v1"},
    ("omniasr-ctc-1b", "fleurs-as"): {"commit": "91aab01fdb3ecb1258fb9e5e06d14ea4a4a71701", "prediction_sha256": "3b00dd8c035aace576c50cd1d921c717b920d2688457a2b748742bbdef17e9e7", "receipt_artifact": "omni_fleurs_full_as_v1"},
    ("omniasr-ctc-1b", "svarah-recovered-en"): {"commit": "e737d114e374fa303124fed59375673809fd8fbd", "prediction_sha256": "9eae45023c91eaf73dc0d935ae98415cb3dd4c76f213d7ad2a17d02d643ca435"},
}
for _spec in CELL_SPECS:
    _spec.update(MODEL_PINS[_spec["model"]] and {"model_id": MODEL_PINS[_spec["model"]][0], "model_revision": MODEL_PINS[_spec["model"]][1]})
    _spec.update(CELL_CONTRACTS[(_spec["model"], _spec["track"])])


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def digest_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def digest(path: Path) -> str:
    return digest_bytes(path.read_bytes())


def read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise ValueError(f"Malformed JSON: {path}") from exc


def jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for number, line in enumerate(path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Malformed JSONL record {number}: {path}") from exc
        require(isinstance(row, dict), f"JSONL record {number} is not an object: {path}")
        rows.append(row)
    return rows


def _resolve(root: Path, source_row: dict[str, Any], field: str, track: str) -> Path:
    value = source_row.get(field)
    require(isinstance(value, str) and value, f"Source row has no {field}")
    if track == "svarah-recovered-en" and field == "reference_path":
        # The manifest preserves the source-era relative path; the recovered
        # frozen bundle owns the verified reference bytes beside its manifest.
        return root / SVARAH_MANIFEST.parent / "references" / Path(value).name
    path = Path(value)
    return path if path.is_absolute() else root / path


def _manifest_rows(root: Path, track: str, manifest_hashes: dict[Path, str]) -> list[dict[str, Any]]:
    if track.startswith("fleurs-"):
        language = track.removeprefix("fleurs-")
        manifest = FLEURS_MANIFEST
        rows = [row for row in jsonl(root / manifest) if row.get("language") == language]
    elif track == "svarah-recovered-en":
        manifest = SVARAH_MANIFEST
        rows = jsonl(root / manifest)
    else:
        raise ValueError(f"Unknown supported track: {track}")
    ids = [row.get("conversation_id") for row in rows]
    require(digest(root / manifest) == manifest_hashes[manifest], f"Frozen manifest hash differs: {manifest}")
    require(all(isinstance(cid, str) for cid in ids) and len(ids) == len(set(ids)), f"Frozen manifest has duplicate/malformed IDs for {track}")
    return rows


def _receipt_is_accepted(path: Path, kind: str) -> dict[str, Any]:
    require(path.is_file(), f"Independent acceptance receipt is missing: {path}")
    receipt = read_json(path)
    require(isinstance(receipt, dict), f"Independent acceptance receipt is not an object: {path}")
    if kind == "valid":
        require(receipt.get("valid") is True, f"Independent acceptance receipt is not valid: {path}")
    elif kind == "whisper_prior":
        require(isinstance(receipt.get("manifest_sha256"), str) and isinstance(receipt.get("scored_predictions"), int) and receipt["scored_predictions"] > 0,
                f"Whisper prior independent receipt is malformed: {path}")
    elif kind == "omni_prior":
        require(isinstance(receipt.get("records"), int) and receipt["records"] > 0,
                f"Omni prior independent receipt is malformed: {path}")
    else:
        raise ValueError(f"Unknown acceptance receipt kind: {kind}")
    return receipt


def _receipt_count(receipt: dict[str, Any]) -> int | None:
    for key in ("n", "records", "conversation_count"):
        value = receipt.get(key)
        if isinstance(value, int):
            return value
    return None


def _validate_execution_contract(root: Path, spec: dict[str, Any], receipt: dict[str, Any], receipt_artifact: Path) -> None:
    """Bind acceptance evidence to the pinned model, cohort size, and commit."""
    if spec["receipt_kind"] == "whisper_prior":
        correction = receipt.get("correction_run")
        require(isinstance(correction, dict) and correction.get("provider_status") == "done" and correction.get("commit") == spec["commit"],
                f"Whisper prior receipt commit differs for {spec['model']} {spec['track']}")
        return
    config_path = receipt_artifact / "config.json"
    source_receipt_path = receipt_artifact / "orx_source_receipt.json"
    require(config_path.is_file() and source_receipt_path.is_file(),
            f"Execution config/source receipt is missing for {spec['model']} {spec['track']}")
    config, source_receipt = read_json(config_path), read_json(source_receipt_path)
    require(config.get("model_id") == spec["model_id"] and config.get("model_revision") == spec["model_revision"],
            f"Executed model pin differs for {spec['model']} {spec['track']}")
    require(source_receipt.get("provider_status") == "done" and source_receipt.get("commit") == spec["commit"],
            f"Executed source commit differs for {spec['model']} {spec['track']}")
    receipt_count = _receipt_count(receipt)
    require(receipt_count == spec["records"], f"Independent receipt count differs for {spec['model']} {spec['track']}")
    model = receipt.get("model", receipt.get("model_id"))
    if model is not None:
        require(model == spec["model_id"], f"Independent receipt model differs for {spec['model']} {spec['track']}")
    revision = receipt.get("revision", receipt.get("model_revision"))
    if revision is not None:
        require(revision == spec["model_revision"], f"Independent receipt revision differs for {spec['model']} {spec['track']}")
    nested_receipt = receipt.get("receipt", receipt.get("provider_receipt"))
    if nested_receipt is not None:
        require(isinstance(nested_receipt, dict) and nested_receipt.get("commit") == spec["commit"],
                f"Independent receipt execution commit differs for {spec['model']} {spec['track']}")


def _prediction_value(prediction: dict[str, Any], key: str) -> Any:
    if key in prediction:
        return prediction[key]
    evidence = prediction.get("evidence")
    if isinstance(evidence, dict):
        aliases = {"audio_sha256": "input_audio_sha256", "reference_sha256": "reference_sha256"}
        return evidence.get(aliases.get(key, key))
    return None


def _route(prediction: dict[str, Any]) -> str:
    value = prediction.get("route", prediction.get("inference_route", prediction.get("condition_id", "unknown")))
    return value if isinstance(value, str) and value else "unknown"


def _accepted_cell(root: Path, spec: dict[str, Any], manifest_hashes: dict[Path, str]) -> dict[str, Any]:
    artifact = root / "research/bridge_open/artifacts" / spec["artifact"]
    predictions_path = artifact / spec["prediction"]
    receipt_artifact = root / "research/bridge_open/artifacts" / spec.get("receipt_artifact", spec["artifact"])
    receipt_path = receipt_artifact / spec["acceptance"]
    if not predictions_path.exists() and not receipt_path.exists():
        return {"model": spec["model"], "track": spec["track"], "expected_records": spec["records"], "status": "pending",
                "artifact": str(artifact.relative_to(root)), "reason": "prediction evidence and independent acceptance receipt are not yet imported"}
    require(predictions_path.is_file(), f"Prediction evidence is missing for a partially imported cell: {predictions_path}")
    receipt = _receipt_is_accepted(receipt_path, spec["receipt_kind"])
    _validate_execution_contract(root, spec, receipt, receipt_artifact)
    prediction_hash = digest(predictions_path)
    if "prediction_sha256" in spec:
        require(prediction_hash == spec["prediction_sha256"], f"Pinned prior prediction hash differs for {spec['model']} {spec['track']}")
    source_rows = _manifest_rows(root, spec["track"], manifest_hashes)
    require(len(source_rows) == spec["records"], f"Frozen source cohort count differs for {spec['track']}")
    source_by_id = {row["conversation_id"]: row for row in source_rows}
    predictions = [row for row in jsonl(predictions_path) if row.get("language") == source_rows[0].get("language")]
    if spec["receipt_kind"] == "omni_prior":
        full_predictions = receipt_artifact / "predictions.jsonl"
        require(full_predictions.is_file(), f"Full-cohort Omni prediction evidence is missing: {full_predictions}")
        full_rows = [row for row in jsonl(full_predictions) if row.get("language") == source_rows[0].get("language")]
        comparison_rows = [{key: value for key, value in row.items() if key not in {"source_sentence_id", "script_mismatch", "unknown_script"}}
                           for row in predictions]
        require(full_rows == comparison_rows,
                f"Pinned Omni comparison predictions do not exactly match the full-cohort accepted run for {spec['track']}")
    ids = [row.get("conversation_id") for row in predictions]
    require(len(predictions) == spec["records"], f"Prediction count differs for {spec['model']} {spec['track']}")
    require(all(isinstance(cid, str) for cid in ids) and len(ids) == len(set(ids)),
            f"Prediction IDs are duplicate or malformed for {spec['model']} {spec['track']}")
    require(set(ids) == set(source_by_id), f"Prediction cohort does not exactly join frozen source for {spec['model']} {spec['track']}")
    totals: Counter[str] = Counter()
    statuses: Counter[str] = Counter()
    routes: Counter[str] = Counter()
    utterance_wil = 0.0
    empty_outputs = 0
    raw_empty_outputs = 0
    eos_observable_records = 0
    observed_missing_eos = 0
    wer_ge_1 = 0
    for prediction in predictions:
        source = source_by_id[prediction["conversation_id"]]
        model_id = prediction.get("model_id")
        model_revision = prediction.get("model_revision")
        if model_id is None and isinstance(prediction.get("model"), dict):
            model_id = prediction["model"].get("hf_model_id")
            model_revision = prediction["model"].get("hf_model_revision")
        require(model_id == spec["model_id"] and model_revision == spec["model_revision"],
                f"Prediction model pin differs for {prediction['conversation_id']}")
        require(prediction.get("status") in TERMINAL_STATUSES,
                f"Prediction is runtime-error or nonterminal for {prediction['conversation_id']}")
        require(_prediction_value(prediction, "audio_sha256") == source.get("audio_sha256"),
                f"Prediction audio hash differs for {prediction['conversation_id']}")
        require(_prediction_value(prediction, "reference_sha256") == source.get("reference_sha256"),
                f"Prediction reference hash differs for {prediction['conversation_id']}")
        reference_path = _resolve(root, source, "reference_path", spec["track"])
        require(reference_path.is_file() and digest(reference_path) == source.get("reference_sha256"),
                f"Frozen reference bytes/hash differ for {prediction['conversation_id']}")
        reference = read_json(reference_path).get("reference_text")
        raw = prediction.get("text_raw")
        require(isinstance(reference, str) and isinstance(raw, str), f"Prediction/reference text is malformed for {prediction['conversation_id']}")
        # Historical accepted artifacts differ in whether this convenience field
        # preserved punctuation.  Quality evidence is always recomputed from raw
        # text below, using the one frozen scorer, rather than trusting it.
        recomputed = score(reference, raw)
        require(prediction.get("metrics") == recomputed, f"Prediction metrics differ from independent rescore for {prediction['conversation_id']}")
        totals.update({key: recomputed[key] for key in ("substitutions", "deletions", "insertions", "ref_words", "ref_chars", "character_errors")})
        utterance_wil += recomputed["wil"]
        empty_outputs += int(not normalize_v1(raw))
        raw_empty_outputs += int(not raw)
        eos = prediction.get("eos")
        if isinstance(eos, dict) and isinstance(eos.get("missing_eos"), bool):
            eos_observable_records += 1
            observed_missing_eos += int(eos["missing_eos"])
        wer_ge_1 += int(recomputed["wer"] >= 1.0)
        statuses[str(prediction.get("status", "unknown"))] += 1
        routes[_route(prediction)] += 1
    require(totals["ref_words"] > 0 and totals["ref_chars"] > 0, f"Zero scoring denominator for {spec['model']} {spec['track']}")
    word_errors = totals["substitutions"] + totals["deletions"] + totals["insertions"]
    result = {"model": spec["model"], "track": spec["track"], "expected_records": spec["records"], "status": "accepted",
              "artifact": str(artifact.relative_to(root)), "acceptance_receipt": str(receipt_path.relative_to(root)),
              "acceptance_receipt_sha256": digest(receipt_path), "source_predictions": str(predictions_path.relative_to(root)),
              "source_predictions_sha256": prediction_hash, "records": len(predictions),
              "word_errors": word_errors, "substitutions": totals["substitutions"], "deletions": totals["deletions"], "insertions": totals["insertions"],
              "reference_words": totals["ref_words"], "character_errors": totals["character_errors"], "reference_chars": totals["ref_chars"],
              "micro_wer": word_errors / totals["ref_words"], "micro_cer": totals["character_errors"] / totals["ref_chars"],
              "mean_utterance_wil": utterance_wil / len(predictions), "empty_outputs": empty_outputs,
              "raw_empty_outputs": raw_empty_outputs, "eos_observable_records": eos_observable_records,
              "observed_missing_eos": observed_missing_eos if eos_observable_records else None, "utterance_wer_ge_1": wer_ge_1,
              "status_counts": dict(sorted(statuses.items())), "route_counts": dict(sorted(routes.items())),
              "manifest": str((FLEURS_MANIFEST if spec["track"].startswith("fleurs-") else SVARAH_MANIFEST)),
              "manifest_sha256": manifest_hashes[FLEURS_MANIFEST if spec["track"].startswith("fleurs-") else SVARAH_MANIFEST]}
    return result


def consolidate(root: Path = ROOT, specs: Iterable[dict[str, Any]] = CELL_SPECS,
                manifest_hashes: dict[Path, str] = MANIFEST_SHA256) -> dict[str, Any]:
    cells = [_accepted_cell(root, spec, manifest_hashes) for spec in specs]
    source_prediction_hashes = {
        f"{cell['model']}::{cell['track']}": cell["source_predictions_sha256"]
        for cell in cells if cell["status"] == "accepted"
    }
    return {"schema": "core-asr-quality-results/v1", "supported_cell_count": len(cells),
            "accepted_cell_count": sum(cell["status"] == "accepted" for cell in cells),
            "pending_cell_count": sum(cell["status"] == "pending" for cell in cells),
            "source_prediction_hashes": source_prediction_hashes,
            "aggregation": "No aggregate winner or cross-cell ranking is computed; results are descriptive per fixed cell.",
            "cells": cells}


def markdown(result: dict[str, Any]) -> str:
    lines = ["# Core ASR quality results", "", "Each accepted cell is independently rejoined to frozen source/reference bytes and rescored. Pending cells have no quality value.", "",
             "| Model | Track | Status | N | Micro WER (%) | Micro CER (%) | Mean utterance WIL (%) | Empty (normalized) | WER ≥100% |", "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for cell in result["cells"]:
        if cell["status"] == "pending":
            lines.append(f"| {cell['model']} | {cell['track']} | pending | {cell['expected_records']} | — | — | — | — | — |")
            continue
        lines.append("| {model} | {track} | accepted | {records} | {micro_wer:.2%} | {micro_cer:.2%} | {mean_utterance_wil:.2%} | {empty_outputs} | {utterance_wer_ge_1} |".format(**cell))
    lines += ["", "No aggregate winner is reported: cells use different models, languages, and fixed source cohorts.", ""]
    return "\n".join(lines)


def write_results(result: dict[str, Any], output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    (output / "quality_results.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    (output / "QUALITY_RESULTS.md").write_text(markdown(result))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    result = consolidate(args.root)
    write_results(result, args.output)
    print(json.dumps({key: result[key] for key in ("supported_cell_count", "accepted_cell_count", "pending_cell_count")}, indent=2))


if __name__ == "__main__":
    main()
