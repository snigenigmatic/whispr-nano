"""Independent acceptance of transported fixed-condition Whisper artifacts.

This validator trusts neither the worker summary nor files in the current
checkout.  It reconstructs the expected condition from the hash-checked ORX
source archive, verifies the exact execution commit, then re-scores every
prediction from frozen reference bytes in that archive.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import tarfile
from pathlib import Path
from typing import Any

from research.bridge_open.scoring import normalize_v1, score

TRUSTED_COMMIT = "949a23078e4b15745aa752a373982f74634997e0"
MODEL_ID = "openai/whisper-small"
MODEL_REVISION = "973afd24965f72e36ca33b3055d56a652f456b4d"
SAMPLE_RATE = 16_000


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha(path: str | Path) -> str:
    return sha_bytes(Path(path).read_bytes())


def load(path: str | Path) -> Any:
    return json.loads(Path(path).read_text())


def lines(path: str | Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def _archive(run: Path) -> tuple[dict[str, Any], dict[str, bytes]]:
    receipt = load(run / "orx_source_receipt.json")
    require(receipt.get("provider_status") == "done", "ORX provider did not finish")
    require(receipt.get("commit") == TRUSTED_COMMIT, "ORX source commit is not the approved launch commit")
    name = receipt.get("portable_archive")
    require(isinstance(name, str) and Path(name).name == name, "Portable archive must be a basename")
    archive = run / name
    require(archive.is_file(), "Portable source archive is absent")
    blob = archive.read_bytes()
    require(sha_bytes(blob) == receipt.get("portable_archive_sha256"), "Portable archive digest differs from receipt")
    raw = gzip.decompress(blob)
    require(sha_bytes(raw) == receipt.get("sourceDigest"), "Source tar digest differs from receipt")
    payload: dict[str, bytes] = {}
    with tarfile.open(archive, "r:gz") as tf:
        for member in tf:
            name = member.name.removeprefix("./")
            path = Path(name)
            require(name and not path.is_absolute() and ".." not in path.parts, "Source archive path is unsafe")
            if not member.isfile():
                continue
            require(name not in payload, f"Duplicate source archive member: {name}")
            stream = tf.extractfile(member)
            require(stream is not None, f"Cannot read source archive member: {name}")
            payload[name] = stream.read()
    require(payload, "Source archive is empty")
    return receipt, payload


def _payload_json(payload: dict[str, bytes], name: str) -> Any:
    require(name in payload, f"Source archive misses {name}")
    try:
        return json.loads(payload[name])
    except json.JSONDecodeError as exc:
        raise ValueError(f"Source archive {name} is not JSON") from exc


def _expected_config(payload: dict[str, bytes]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    selected = _payload_json(payload, "experiments/selected_condition.json")
    suite = _payload_json(payload, "experiments/core_suite.json")
    legacy = _payload_json(payload, "experiments/config.json")
    matching = [item for item in suite["conditions"] if item.get("condition_id") == selected.get("condition_id")]
    require(len(matching) == 1, "Selected condition is not uniquely defined in archived suite")
    condition = matching[0]
    require(selected.get("expected_count") == condition.get("expected_count"), "Selected expected count differs from archived suite")
    source_spec = suite.get("source_manifests", {}).get(condition.get("source"))
    require(isinstance(source_spec, dict), "Archived suite has no source specification for condition")
    require(selected.get("source_manifest") == source_spec.get("path") and
            selected.get("source_manifest_sha256") == source_spec.get("sha256"), "Selected manifest differs from archived suite")
    manifest_name = selected["source_manifest"]
    require(manifest_name in payload and sha_bytes(payload[manifest_name]) == selected["source_manifest_sha256"],
            "Archived source manifest digest differs from selected condition")
    manifest = [json.loads(line) for line in payload[manifest_name].decode().splitlines() if line.strip()]
    by_id = {row.get("conversation_id"): row for row in manifest}
    require(len(by_id) == len(manifest), "Archived source manifest has duplicate conversation IDs")
    ids = selected.get("conversation_ids")
    require(isinstance(ids, list) and len(ids) == condition["expected_count"] == len(set(ids)), "Selected IDs are not an exact cohort")
    require(all(cid in by_id for cid in ids), "Selected cohort contains IDs absent from archived manifest")
    cohort = [by_id[cid] for cid in ids]
    require(all(row.get("language") == condition["language"] for row in cohort), "Archived cohort has a wrong language")
    model = suite.get("model", {})
    expected = dict(selected, language=condition["language"], model_id=model.get("model_id"),
                    model_revision=model.get("model_revision"), native_model=model.get("native_model"),
                    short_route=suite.get("fixed_short_route"), long_route=suite.get("fixed_long_route"),
                    max_wall_seconds=condition.get("max_wall_seconds"), native_transcribe=legacy.get("native_transcribe"),
                    core_suite_sha256=sha_bytes(payload["experiments/core_suite.json"]))
    require(expected["model_id"] == MODEL_ID and expected["model_revision"] == MODEL_REVISION and expected["native_model"] == "small",
            "Archived suite is not the pinned Whisper Small contract")
    return expected, cohort


def _reference(payload: dict[str, bytes], source: dict[str, Any]) -> str:
    if source.get("source_repo") == "google/fleurs":
        refs = _payload_json(payload, "research/bridge_open/frozen_references.json")
        raw = refs.get(source["conversation_id"])
        require(isinstance(raw, str), "Frozen FLEURS reference is absent")
        parsed = json.loads(raw)
        require(isinstance(parsed, dict) and isinstance(parsed.get("reference_text"), str), "Frozen FLEURS reference is malformed")
        text = parsed["reference_text"]
        require(sha_bytes(raw.encode()) == source["reference_sha256"], "Frozen FLEURS reference digest differs from manifest")
        return text
    name = source.get("reference_path")
    require(isinstance(name, str) and name in payload, "Archived reference bytes are absent")
    raw = payload[name]
    require(sha_bytes(raw) == source["reference_sha256"], "Archived reference digest differs from manifest")
    parsed = json.loads(raw)
    require(isinstance(parsed, dict) and isinstance(parsed.get("reference_text"), str), "Archived reference is malformed")
    return parsed["reference_text"]


def _same_float(actual: Any, expected: float, label: str) -> None:
    require(isinstance(actual, (int, float)) and math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-12), label)


def _check_receipts(run: Path, config: dict[str, Any], environment: dict[str, Any], payload: dict[str, bytes]) -> None:
    model = environment.get("model")
    require(isinstance(model, dict) and environment.get("dtype") == "float32" and model.get("id") == MODEL_ID and model.get("revision") == MODEL_REVISION,
            "Environment model pin differs from config")
    receipt = load(run / "hf_receipt.json")
    require(receipt.get("model_id") == MODEL_ID and receipt.get("requested_revision") == MODEL_REVISION, "HF receipt model pin differs")
    files = receipt.get("files")
    require(isinstance(files, dict) and files, "HF receipt lacks downloaded-file digests")
    safe = {name: digest for name, digest in files.items() if name.endswith(".safetensors")}
    require(safe and model.get("hf_safetensors_sha256") == safe, "HF safetensor receipt differs from environment")
    native = load(run / "native_receipt.json")
    require(native.get("native_model") == "small" and native.get("checkpoint_sha256") == native.get("official_url_sha256"),
            "Native checkpoint does not match its official URL pin")
    require(model.get("native_checkpoint_sha256") == native.get("checkpoint_sha256") and
            model.get("native_official_download_url") == native.get("official_download_url"),
            "Native receipt differs from environment")
    code = environment.get("code", {})
    require("experiments/hf_run.py" in payload and "experiments/diagnostic_v1.py" in payload and "experiments/hf_entrypoint.sh" in payload,
            "Archived worker, diagnostic, or fixed entrypoint is absent")
    require(code.get("worker_sha256") == sha_bytes(payload["experiments/hf_run.py"]), "Archived worker hash differs")
    require(code.get("hf_key_source_sha256") == sha_bytes(payload["experiments/diagnostic_v1.py"]), "Archived hf_key source hash differs")
    require(sha(run / "config.json") == environment.get("config_sha256", sha(run / "config.json")), "Environment config digest differs")


def _check_proofs(run: Path, cohort: list[dict[str, Any]], environment: dict[str, Any]) -> None:
    weights = load(run / "weight_validation.json")
    require(isinstance(weights, list) and len(weights) == environment.get("validation", {}).get("weight_tensor_count"), "Weight proof count differs")
    native_keys: set[str] = set(); hf_keys: set[str] = set()
    for row in weights:
        native_key, hf_key = row.get("native_key"), row.get("hf_key")
        require(isinstance(native_key, str) and native_key not in native_keys and isinstance(hf_key, str) and hf_key not in hf_keys,
                "Weight proof has duplicate or malformed tensor keys")
        native_keys.add(native_key); hf_keys.add(hf_key)
        delta = row.get("max_abs_delta")
        require(delta == 0.0, "Learned tensor is not byte-equal") if row.get("learned") is True else require(
            isinstance(delta, (int, float)) and delta <= 1e-5, "Non-learned tensor delta exceeds tolerance")
    features = load(run / "feature_validation.json")
    require(isinstance(features, list) and len(features) == len(cohort) == environment.get("validation", {}).get("feature_count"),
            "Feature proof count differs from exact cohort")
    require({row.get("conversation_id") for row in features} == {row["conversation_id"] for row in cohort}, "Feature proof IDs differ from cohort")
    for row in features:
        require(row.get("n_mels") == 80 and isinstance(row.get("max_abs_delta"), (int, float)) and row["max_abs_delta"] < 1e-4,
                "Feature proof fails the 80-bin parity gate")


def _check_prediction(prediction: dict[str, Any], source: dict[str, Any], config: dict[str, Any], worker_hash: str,
                      config_hash: str, payload: dict[str, bytes], pilot_ids: set[str]) -> None:
    cid = source["conversation_id"]
    require(prediction.get("conversation_id") == cid and prediction.get("language") == source["language"], "Prediction ID/language differs")
    require(prediction.get("model") == {"hf_model_id": MODEL_ID, "hf_model_revision": MODEL_REVISION, "native_model": "small"}, "Prediction model pin differs")
    evidence = prediction.get("evidence", {})
    require(evidence.get("input_audio_sha256") == source["audio_sha256"] and evidence.get("reference_sha256") == source["reference_sha256"] and
            evidence.get("config_sha256") == config_hash and evidence.get("worker_sha256") == worker_hash, "Prediction evidence hashes differ")
    full = prediction.get("full_waveform", {})
    expected_samples = round(float(source["duration_seconds"]) * SAMPLE_RATE)
    require(prediction.get("full_wav_length_samples") == full.get("sample_count") and full.get("sample_rate") == SAMPLE_RATE and
            abs(int(full.get("sample_count", -1)) - expected_samples) <= 1 and
            abs(float(full.get("duration_seconds", -1)) - float(source["duration_seconds"])) < .02, "Prediction full-waveform evidence differs")
    is_short = float(source["duration_seconds"]) <= 30
    if is_short:
        require(prediction.get("route") == "hf_batched_shortform", "Short audio did not use the fixed HF route")
        generation = prediction.get("hf_generation", {})
        require(isinstance(generation.get("token_ids"), list) and generation.get("options", {}).get("num_beams") == 5 and
                generation["options"].get("max_new_tokens") == 440, "HF generation evidence differs")
        eos = prediction.get("eos", {})
        missing = eos.get("missing_eos")
        require(eos.get("status") in {"present", "absent"} and isinstance(missing, bool) and eos.get("cap_observed") == "unknown", "HF EOS/cap evidence differs")
        require((prediction.get("status") == "generation_limit") == missing or prediction.get("status") == "empty_output", "HF status disagrees with missing EOS")
        parity = prediction.get("native_short_parity", {})
        if cid in pilot_ids:
            require(parity.get("parity_phase") == "fixed_three_short_files", "Native parity pilot IDs differ")
            if parity.get("status") != "error":
                require(parity.get("eos", {}).get("status") == "unknown", "Native short EOS must be explicit unknown")
        else:
            require(parity.get("status") == "not_requested", "Non-pilot native parity must be explicitly not requested")
    else:
        require(prediction.get("route") == "native_longform_full_waveform", "Long audio was not routed as full native waveform")
        native = prediction.get("native_longform", {})
        require(native.get("effective_options") == {"language": source["language"], **config["native_transcribe"]}, "Native long options differ")
        require(isinstance(native.get("decode_attempts"), list) and prediction.get("eos", {}).get("status") == "unknown", "Native long trace/EOS evidence differs")
    raw = prediction.get("text_raw")
    require(isinstance(raw, str) and prediction.get("text_for_scoring") == normalize_v1(raw), "Prediction text normalization differs")
    expected_metrics = score(_reference(payload, source), raw)
    require(prediction.get("metrics") == expected_metrics, "Prediction metrics differ from independent rescoring")
    require(prediction.get("status") in {"success", "empty_output", "generation_limit"}, "Prediction has a runtime failure/nonterminal status")


def validate(run: str | Path) -> dict[str, Any]:
    run = Path(run)
    required = ("config.json", "environment.json", "hf_receipt.json", "native_receipt.json", "weight_validation.json",
                "feature_validation.json", "source_mapping.json", "predictions.jsonl", "summary.json", "validation.json", "orx_source_receipt.json")
    require(all((run / name).is_file() for name in required), "Run is missing required acceptance artifacts")
    receipt, payload = _archive(run)
    expected_config, cohort = _expected_config(payload)
    config = load(run / "config.json")
    require(config == expected_config, "Run config differs from archived frozen condition")
    environment, summary, validation = load(run / "environment.json"), load(run / "summary.json"), load(run / "validation.json")
    _check_receipts(run, config, environment, payload)
    _check_proofs(run, cohort, environment)
    mapping = load(run / "source_mapping.json")
    require(mapping.get("source_manifest_sha256") == config["source_manifest_sha256"] and isinstance(mapping.get("rows"), list), "Source mapping is malformed")
    mapped = {row.get("conversation_id"): row for row in mapping["rows"]}
    require(set(mapped) == {row["conversation_id"] for row in cohort}, "Source mapping IDs differ from cohort")
    for source in cohort:
        row = mapped[source["conversation_id"]]
        require(row.get("audio_sha256") == source["audio_sha256"] and row.get("reference_sha256") == source["reference_sha256"], "Source mapping hashes differ")
    predictions = lines(run / "predictions.jsonl")
    ids = [row.get("conversation_id") for row in predictions]
    expected_ids = [row["conversation_id"] for row in cohort]
    require(ids == expected_ids and len(ids) == len(set(ids)), "Prediction IDs/order do not exactly match cohort")
    config_hash, worker_hash = sha(run / "config.json"), sha_bytes(payload["experiments/hf_run.py"])
    short_ids = [row["conversation_id"] for row in cohort if float(row["duration_seconds"]) <= 30]
    pilot_ids = set(short_ids[:3])
    for prediction, source in zip(predictions, cohort, strict=True):
        _check_prediction(prediction, source, config, worker_hash, config_hash, payload, pilot_ids)
    metrics = [row["metrics"] for row in predictions]
    errors = sum(sum(metric[key] for key in ("substitutions", "deletions", "insertions")) for metric in metrics)
    words = sum(metric["ref_words"] for metric in metrics)
    require(words > 0, "Reference denominator is zero")
    status_counts = {status: sum(row["status"] == status for row in predictions) for status in {row["status"] for row in predictions}}
    require(summary.get("status") == "completed" and summary.get("attempted") == len(cohort) and
            summary.get("prediction_records") == len(cohort) and summary.get("scored") == len(cohort) and
            summary.get("runtime_failures") == 0 and summary.get("status_counts") == status_counts,
            "Summary status or denominators do not represent accepted predictions")
    _same_float(summary.get("micro_wer"), errors / words, "Summary micro WER differs")
    require(validation.get("expected_count") == len(cohort) and validation.get("prediction_count") == len(cohort) and
            validation.get("unique_prediction_ids") == len(cohort) and validation.get("missing_prediction_ids") == [] and
            validation.get("weight_gate") == "passed" and validation.get("feature_gate") == "passed", "Validation receipt differs")
    result = {"valid": True, "trusted_commit": TRUSTED_COMMIT, "condition_id": config["condition_id"], "records": len(cohort),
              "word_errors": errors, "reference_words": words, "micro_wer": errors / words, "provider_receipt": receipt,
              "scope": "Fixed public cohort acceptance; no held-out or population-generalization claim."}
    (run / "local_independent_whisper_core_acceptance.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    print(json.dumps(validate(args.run), indent=2))
