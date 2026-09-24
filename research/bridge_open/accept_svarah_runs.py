"""Accept native Svarah runtime checks against exact cached-source evidence."""
import argparse
import gzip
import hashlib
import json
import math
import tarfile
from pathlib import Path

from research.bridge_open.scoring import score, normalize_v1
from research.bridge_open.validate_svarah_inputs import validate as validate_inputs

ROOT = Path(__file__).resolve().parents[2]
ART = ROOT / "research/bridge_open/artifacts"
BUNDLE = ROOT / "data/bridge_open/svarah_native_shard02_v2/bundle"

# Frozen from the accepted 12-item English Svarah execution. This intentionally
# differs from the Hindi FLEURS runner because this is the English adapter.
QWEN_SVARAH_ACCEPTANCE_CODE = {
    "run_qwen.py": "b7e2c3843fcab145bd500c7e3be529314fdf7f5a075207c683ec86a03b6c4315",
    "scoring.py": "a63c0fdce68e33810bf233c813a5ebad0f3692e6718aad2828a802b2e776ff3c",
    "qwen_trace.py": "6331d3c4636351769a4db35812fc6cac5e4186ec14bfe532bdceb220d60a4a3f",
}


def sha(blob):
    return hashlib.sha256(blob).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def lines(path):
    return [json.loads(s) for s in Path(path).read_text().splitlines()]


def require(condition, message):
    """Reject invalid evidence even when Python is run with -O."""
    if not condition:
        raise ValueError(message)


def require_equal(actual, expected, message):
    require(actual == expected, message)


def require_optional_equal(mapping, key, actual, label):
    if key in mapping and mapping[key] is not None:
        require_equal(mapping[key], actual, f"{label}.{key} disagrees with recomputed predictions")


def require_optional_float(mapping, key, actual, label):
    if key in mapping and mapping[key] is not None:
        require(math.isclose(mapping[key], actual, rel_tol=1e-12, abs_tol=1e-12),
                f"{label}.{key} disagrees with recomputed predictions")


def portable_archive_path(run, receipt):
    name = receipt.get("portable_archive")
    require(isinstance(name, str) and name, "Receipt has no portable archive basename")
    relative = Path(name)
    require(not relative.is_absolute() and relative.name == name and len(relative.parts) == 1,
            "Receipt portable_archive must be a basename under the run directory")
    archive = run / relative
    require(archive.is_file() and archive.resolve().parent == run.resolve(),
            "Receipt portable archive is not a regular file directly under the run directory")
    return archive


def require_summary_aggregates(summary, predictions, expected_count):
    errors = sum(sum(row["metrics"][key] for key in ("substitutions", "deletions", "insertions"))
                 for row in predictions)
    words = sum(row["metrics"]["ref_words"] for row in predictions)
    char_errors = sum(row["metrics"]["character_errors"] for row in predictions)
    chars = sum(row["metrics"]["ref_chars"] for row in predictions)
    require(words > 0 and chars > 0, "Recomputed references are empty")
    statuses = [row["status"] for row in predictions]
    successful = sum(status == "success" for status in statuses)
    empty = sum(status == "empty_output" for status in statuses)
    mean_wer = sum(row["metrics"]["wer"] for row in predictions) / expected_count
    mean_wil = sum(row["metrics"]["wil"] for row in predictions) / expected_count
    require_optional_equal(summary, "attempted", expected_count, "summary")
    require_optional_equal(summary, "scored", expected_count, "summary")
    require_optional_equal(summary, "successful", successful, "summary")
    require_optional_equal(summary, "empty_output_count", empty, "summary")
    if "conversation_ids" in summary:
        require_equal(summary["conversation_ids"], [row["conversation_id"] for row in predictions],
                      "summary.conversation_ids does not match prediction order")
    per_language = summary.get("per_language", {}).get("en")
    if per_language is not None:
        require(isinstance(per_language, dict), "summary.per_language.en is not an object")
        for key, actual in {
            "attempted_count": expected_count,
            "scored_count": expected_count,
            "clean_completion_count": successful,
            "word_errors": errors,
            "reference_words": words,
            "character_errors": char_errors,
            "ref_chars": chars,
        }.items():
            require_optional_equal(per_language, key, actual, "summary.per_language.en")
        for key, actual in {
            "micro_wer": errors / words,
            "micro_cer": char_errors / chars,
            "mean_conversation_wer": mean_wer,
            "mean_conversation_wil": mean_wil,
        }.items():
            require_optional_float(per_language, key, actual, "summary.per_language.en")
        if "conversation_ids" in per_language:
            require_equal(per_language["conversation_ids"], [row["conversation_id"] for row in predictions],
                          "summary.per_language.en.conversation_ids does not match prediction order")
    return errors, words


def validate(run, bundle=BUNDLE, expected_count=12):
    run = Path(run)
    bundle = Path(bundle)
    source_check = validate_inputs(bundle, expected_count)
    rows = {row["conversation_id"]: row for row in lines(bundle / "manifest.jsonl")}
    cfg, env, summary = [read(run / (name + ".json")) for name in ("config", "environment", "summary")]
    receipt = read(run / "orx_source_receipt.json")
    require(receipt.get("provider_status") == "done", "Provider receipt is not complete")
    require(read(run / "remote_validation.json").get("valid") is True, "Remote validation is not valid")
    archive = portable_archive_path(run, receipt)
    blob = archive.read_bytes()
    require_equal(sha(blob), receipt.get("portable_archive_sha256"), "Portable archive SHA-256 mismatch")
    require_equal(sha(gzip.decompress(blob)), receipt.get("sourceDigest"), "Portable archive source digest mismatch")

    omni = cfg.get("model_id") == "facebook/omniASR-CTC-1B"
    expected_id, expected_rev = (("facebook/omniASR-CTC-1B", "8c22e3ffdaa4aab6431b128b84b991a7d9c2515c")
                                 if omni else ("Qwen/Qwen3-ASR-1.7B", "7278e1e70fe206f11671096ffdd38061171dd6e5"))
    require_equal(cfg.get("model_id"), expected_id, "Unexpected model ID")
    require_equal(cfg.get("model_revision"), expected_rev, "Unexpected model revision")
    require(isinstance(env.get("code_sha256"), dict), "Environment lacks code SHA-256 mapping")
    if not omni:
        require_equal(env["code_sha256"], QWEN_SVARAH_ACCEPTANCE_CODE,
                      "Qwen execution code does not match frozen Svarah acceptance hashes")

    with tarfile.open(archive, "r:gz") as tar:
        # A single sequential pass avoids re-inflating an 80 MB gzip archive
        # hundreds of times when alternating audio and reference lookups.
        payloads = {}
        for member in tar:
            if not member.isfile():
                continue
            name = member.name.removeprefix("./")
            require(name not in payloads, f"Duplicate portable archive member {name}")
            payload = tar.extractfile(member)
            require(payload is not None, f"Unable to read portable archive member {name}")
            payloads[name] = payload.read()

        def content(name):
            require(name in payloads, f"Portable archive lacks required member {name}")
            return payloads[name]

        committed = json.loads(content("research/bridge_open/run_config.json"))
        current_cfg = {key: value for key, value in cfg.items() if key not in ("output_dir", "manifest_path")}
        committed_cfg = {key: value for key, value in committed.items() if key not in ("output_dir", "manifest_path")}
        require_equal(current_cfg, committed_cfg, "Run config differs from committed portable config")
        hf_receipt = read(run / "hf_receipt.json")
        for name, digest in hf_receipt.get("source_sha256", {}).items():
            require_equal(sha(content(name)), digest, f"Worker receipt source hash mismatch for {name}")
        checked_blob = content(cfg["input_acceptance_receipt"])
        require_equal(sha(checked_blob), cfg["input_acceptance_receipt_sha256"],
                      "Archived independent-input receipt hash mismatch")
        require_equal(json.loads(checked_blob), source_check,
                      "Archived input validation is not the independent check for this cohort")
        if "input_receipt_path" in cfg:
            prepared_blob = content(cfg["input_receipt_path"])
            require_equal(sha(prepared_blob), cfg["input_receipt_sha256"],
                          "Archived preparation receipt hash mismatch")
            prepared = json.loads(prepared_blob)
            require_equal(prepared.get("cohort_count"), expected_count,
                          "Archived preparation receipt selected the wrong cohort count")
            require_equal(prepared.get("cohort_manifest_sha256"), source_check["manifest_sha256"],
                          "Archived preparation receipt selected the wrong manifest")
        require_equal(content("research/bridge_open/svarah_inputs/manifest.jsonl"),
                      (bundle / "manifest.jsonl").read_bytes(), "Portable manifest differs from validated bundle")
        prefix = "research/bridge_open/omni_extension/" if omni else "research/bridge_open/"
        for name, digest in env["code_sha256"].items():
            require_equal(sha(content(prefix + name)), digest, f"Portable code hash mismatch for {name}")
        for row in rows.values():
            for pathkey, hashkey in (("audio_path", "audio_sha256"), ("reference_path", "reference_sha256")):
                require_equal(sha(content(row[pathkey])), row[hashkey],
                              f"Portable {pathkey} hash mismatch for {row['conversation_id']}")

    require_equal(cfg.get("manifest_sha256"), source_check["manifest_sha256"], "Config manifest hash mismatch")
    require_equal(env.get("manifest_sha256"), source_check["manifest_sha256"], "Environment manifest hash mismatch")
    require_equal(env.get("config_sha256"), sha((run / "config.json").read_bytes()), "Environment config hash mismatch")
    require_equal(cfg.get("supported_languages"), ["en"], "Unexpected supported languages")
    parent = read(ART / ("omni_fleurs_acceptance_v1" if omni else "qwen_native_full_hi_v1") / "environment.json")
    if omni:
        require(cfg.get("batch_size") == 1 and cfg.get("dtype") == "bfloat16" and cfg.get("full_waveform") is True,
                "Omni native decoding contract changed")
        require_equal(env.get("asset_sha256"), parent.get("asset_sha256"), "Omni asset hashes differ from accepted baseline")
        require_equal(env["code_sha256"], parent.get("code_sha256"), "Omni code hashes differ from accepted baseline")
        runtime = env.get("runtime_evidence", {})
        require(runtime.get("model_training") is False, "Omni model was not confirmed inference-only")
        require_equal(runtime.get("parameter_dtypes"), ["torch.bfloat16"], "Unexpected Omni parameter dtype")
        require_equal(runtime.get("pipeline_source_sha256"), parent.get("runtime_evidence", {}).get("pipeline_source_sha256"),
                      "Omni pipeline source hash differs from accepted baseline")
    else:
        require_equal(env.get("weight_sha256"), parent.get("weight_sha256"), "Qwen weight hashes differ from accepted baseline")
        require_equal(env.get("auxiliary_file_sha256"), parent.get("auxiliary_file_sha256"),
                      "Qwen auxiliary file hashes differ from accepted baseline")
        require_equal(env.get("actual_model_dtype"), "torch.bfloat16", "Unexpected Qwen model dtype")
        require(env.get("model_eval") is True, "Qwen model_eval evidence is absent")
        require(cfg.get("transcribe_language") == "English" and cfg.get("num_beams") == 1 and cfg.get("batch_size") == 8,
                "Qwen English decoding contract differs from acceptance configuration")
        package = env.get("qwen_asr_package_file_sha256", {})
        for name, digest in parent.get("qwen_asr_package_file_sha256", {}).items():
            if name.endswith(".py"):
                require_equal(package.get(name), digest, f"Qwen package Python hash differs for {name}")

    predictions = lines(run / "predictions.jsonl")
    require_equal(len(predictions), expected_count, "Prediction count does not match expected cohort")
    require_equal(len({row["conversation_id"] for row in predictions}), expected_count, "Predictions contain duplicate IDs")
    require_equal({row["conversation_id"] for row in predictions}, set(rows), "Prediction IDs differ from bundle")
    require_equal(set(rows), set(cfg.get("conversation_ids", [])), "Config IDs differ from bundle")
    for row in predictions:
        src = rows[row["conversation_id"]]
        require(row.get("status") in ("success", "empty_output") and row.get("language") == "en",
                f"Invalid status or language for {row['conversation_id']}")
        require_equal(row.get("model_id"), cfg["model_id"], f"Prediction model ID mismatch for {row['conversation_id']}")
        require_equal(row.get("model_revision"), cfg["model_revision"],
                      f"Prediction model revision mismatch for {row['conversation_id']}")
        require_equal(row.get("audio_sha256"), src["audio_sha256"], f"Prediction audio hash mismatch for {row['conversation_id']}")
        require_equal(row.get("reference_sha256"), src["reference_sha256"],
                      f"Prediction reference hash mismatch for {row['conversation_id']}")
        ref = read(bundle / "references" / Path(src["reference_path"]).name)["reference_text"]
        require_equal(row.get("metrics"), score(ref, row["text_raw"]), f"Prediction metrics mismatch for {row['conversation_id']}")
        require_equal(row.get("text_for_scoring"), normalize_v1(row["text_raw"]),
                      f"Prediction normalized text mismatch for {row['conversation_id']}")
        require_equal(len(row.get("chunks", [])), 1, f"Prediction chunk count mismatch for {row['conversation_id']}")
        chunk = row["chunks"][0]
        require(chunk.get("start_seconds") == 0 and abs(chunk.get("end_seconds") - src["duration_seconds"]) < 1e-6,
                f"Prediction chunk range mismatch for {row['conversation_id']}")
        if omni:
            require(row.get("api_token_agreement") is True and row.get("termination_semantics") == "ctc_no_eos",
                    f"Omni token/API evidence mismatch for {row['conversation_id']}")
            raw = chunk.get("ctc_raw_token_ids")
            require(bool(raw) and all(type(token) is int and token >= 0 for token in raw),
                    f"Invalid CTC raw token evidence for {row['conversation_id']}")
            collapsed = [token for i, token in enumerate(raw) if not i or raw[i - 1] != token]
            require_equal(chunk.get("ctc_collapsed_token_ids"), collapsed,
                          f"CTC collapse evidence mismatch for {row['conversation_id']}")
            require(chunk.get("eos_present") is None, f"CTC EOS evidence must be null for {row['conversation_id']}")
        else:
            trace = chunk.get("generation_trace")
            require(bool(trace), f"Missing Qwen generation trace for {row['conversation_id']}")
            for event in trace:
                tokens = event.get("new_token_sequences")
                require(bool(tokens) and event.get("observation_status") == "captured",
                        f"Incomplete Qwen generation trace for {row['conversation_id']}")
                require(any(token in cfg["eos_token_ids"] for token in tokens) and event.get("eos_present") is True and
                        event.get("cap_observed") is False, f"Qwen EOS/cap evidence mismatch for {row['conversation_id']}")

    errors, words = require_summary_aggregates(summary, predictions, expected_count)
    if omni:
        require_optional_equal(summary, "runtime_failure_count", 0, "summary")
    result = dict(
        valid=True,
        model=cfg["model_id"],
        n=expected_count,
        word_errors=errors,
        reference_words=words,
        micro_wer=errors / words,
        source_cohort=f"{expected_count} items from one recovered 518-record shard; not full Svarah",
        api_agreements=expected_count if omni else None,
        eos_completed=None if omni else expected_count,
        interpretation=("Runtime/source-acceptance result only; no model ranking or generalization claim" if expected_count == 12 else
                        "Descriptive comparison on all 518 available shard02 items; includes preexposed data; not full Svarah or independent confirmation"),
        receipt=receipt,
    )
    (run / "local_independent_acceptance.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("run")
    parser.add_argument("--bundle", type=Path, default=BUNDLE)
    parser.add_argument("--expected-count", type=int, default=12)
    args = parser.parse_args()
    print(json.dumps(validate(args.run, args.bundle, args.expected_count), indent=2))
