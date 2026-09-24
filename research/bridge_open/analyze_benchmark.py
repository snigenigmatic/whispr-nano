"""CPU-only aggregate analysis for frozen open-ASR run artifacts."""
from __future__ import annotations

import argparse
import csv
import json
import random
from itertools import combinations
from pathlib import Path
from typing import Any


SCORED = {"success", "empty_output", "generation_limit"}
BOOTSTRAP_REPS = 2000
BOOTSTRAP_SEED = 20260922


def _read(path: Path) -> Any:
    return json.loads(path.read_text())


def _prediction_rows(directory: Path) -> tuple[dict, dict, list[dict]]:
    cfg = _read(directory / "config.json")
    env = _read(directory / "environment.json") if (directory / "environment.json").exists() else {}
    if env.get("device") is not None and cfg.get("device") is not None and env["device"] != cfg["device"]:
        raise ValueError(f"environment device disagrees with config in {directory}")
    if env.get("dtype") is not None and cfg.get("dtype") is not None and env["dtype"] != cfg["dtype"]:
        raise ValueError(f"environment dtype disagrees with config in {directory}")
    predictions = [json.loads(line) for line in (directory / "predictions.jsonl").read_text().splitlines()]
    if set(cfg.get("conversation_ids", [])) != {r.get("conversation_id") for r in predictions}:
        raise ValueError(f"config IDs do not exactly match predictions in {directory}")
    if len(predictions) != len({r.get("conversation_id") for r in predictions}):
        raise ValueError(f"duplicate prediction IDs in {directory}")
    return cfg, env, predictions


def _aggregate(rows: list[dict], termination_semantics: str = "eos") -> dict[str, Any]:
    scored = [r for r in rows if r["status"] in SCORED and isinstance(r.get("metrics"), dict)]
    all_languages = sorted({r.get("language") for r in rows})
    by_language: dict[str, list[dict]] = {language: [] for language in all_languages}
    for row in scored: by_language.setdefault(row["language"], []).append(row)
    languages = {}
    for language in all_languages:
        group = by_language[language]
        word_errors = sum(m["substitutions"] + m["deletions"] + m["insertions"] for r in group for m in [r["metrics"]])
        ref_words = sum(r["metrics"]["ref_words"] for r in group)
        char_errors = sum(r["metrics"]["character_errors"] for r in group)
        ref_chars = sum(r["metrics"]["ref_chars"] for r in group)
        languages[language] = {
            "attempted": sum(r["language"] == language for r in rows),
            "scored": len(group),
            "failed": sum(r["language"] == language and r["status"] not in SCORED for r in rows),
            "capped_conversations": sum(r["status"] == "generation_limit" for r in group),
            "word_errors": word_errors, "ref_words": ref_words,
            "character_errors": char_errors, "ref_chars": ref_chars,
            "micro_wer": word_errors / ref_words if ref_words else None,
            "micro_cer": char_errors / ref_chars if ref_chars else None,
            "mean_conversation_wer": sum(r["metrics"]["wer"] for r in group) / len(group) if group else None,
        }
    macro = [v["micro_wer"] for v in languages.values() if v["micro_wer"] is not None]
    word_errors = sum(r["metrics"]["substitutions"] + r["metrics"]["deletions"] + r["metrics"]["insertions"] for r in scored)
    ref_words = sum(r["metrics"]["ref_words"] for r in scored)
    char_errors = sum(r["metrics"]["character_errors"] for r in scored)
    ref_chars = sum(r["metrics"]["ref_chars"] for r in scored)
    return {"attempted": len(rows), "scored": len(scored), "failed": len(rows) - len(scored),
            "capped_chunks": 0 if termination_semantics == "ctc_no_eos" else sum(c.get("eos_present") is False for r in rows for c in r.get("chunks", [])),
            "total_chunks": sum(len(r.get("chunks", [])) for r in rows),
            "word_errors": word_errors, "ref_words": ref_words, "character_errors": char_errors,
            "ref_chars": ref_chars, "micro_wer": word_errors / ref_words if ref_words else None,
            "micro_cer": char_errors / ref_chars if ref_chars else None,
            "macro_language_wer": sum(macro) / len(macro) if macro else None,
            "macro_conversation_wer": sum(v["mean_conversation_wer"] for v in languages.values() if v["mean_conversation_wer"] is not None) / sum(v["mean_conversation_wer"] is not None for v in languages.values()) if any(v["mean_conversation_wer"] is not None for v in languages.values()) else None,
            "languages": languages}


def _bootstrap(a: dict[str, dict], b: dict[str, dict]) -> dict[str, Any]:
    common = sorted(set(a) & set(b))
    strata: dict[str, list[str]] = {}
    for cid in common:
        strata.setdefault(a[cid]["language"], []).append(cid)
    observed_wer = sum(b[c]["metrics"]["wer"] - a[c]["metrics"]["wer"] for c in common) / len(common) if common else None
    observed_cer = sum(b[c]["metrics"]["cer"] - a[c]["metrics"]["cer"] for c in common) / len(common) if common else None
    result = {"n_conversations": len(common), "conversation_ids": common,
              "wer_delta_b_minus_a": observed_wer, "cer_delta_b_minus_a": observed_cer,
              "bootstrap_reps": BOOTSTRAP_REPS, "seed": BOOTSTRAP_SEED}
    if len(common) < 5:
        result["inference_suppressed"] = "fewer than 5 shared conversations"
        return result
    rng = random.Random(BOOTSTRAP_SEED)
    wer_samples, cer_samples = [], []
    for _ in range(BOOTSTRAP_REPS):
        sample = [rng.choice(ids) for ids in strata.values() for _ in ids]
        wer_samples.append(sum(b[c]["metrics"]["wer"] - a[c]["metrics"]["wer"] for c in sample) / len(sample))
        cer_samples.append(sum(b[c]["metrics"]["cer"] - a[c]["metrics"]["cer"] for c in sample) / len(sample))
    for key, samples, observed in (("wer", wer_samples, observed_wer), ("cer", cer_samples, observed_cer)):
        ordered = sorted(samples)
        result[f"{key}_ci95"] = [ordered[int(.025 * BOOTSTRAP_REPS)], ordered[int(.975 * BOOTSTRAP_REPS) - 1]]
    return result


def analyze(run_dirs: list[str | Path]) -> dict[str, Any]:
    fragments, seen, manifest_hashes = [], set(), set()
    manifest_counts = None
    for raw_dir in run_dirs:
        directory = Path(raw_dir)
        cfg, env, predictions = _prediction_rows(directory)
        key = (cfg.get("model_id"), cfg.get("model_revision"), cfg.get("condition_id"))
        manifest_hashes.add(cfg.get("manifest_sha256"))
        recipe = {"model_id": cfg.get("model_id"), "model_revision": cfg.get("model_revision"), "condition_id": cfg.get("condition_id"), "device": cfg.get("device"), "dtype": env.get("dtype", cfg.get("dtype")), "batch_size": cfg.get("batch_size"), "num_beams": cfg.get("num_beams"), "return_timestamps": cfg.get("return_timestamps"), "max_new_tokens": cfg.get("max_new_tokens"), "termination_semantics": cfg.get("termination_semantics", "eos"), "chunk_seconds": cfg.get("chunk_seconds"), "seed": cfg.get("seed"), "metric_protocol": cfg.get("metric_protocol"), "scoring_code_sha256": env.get("code_sha256", {}).get("scoring.py")}
        for row in predictions:
            identity = (*key, row.get("conversation_id"))
            if identity in seen:
                raise ValueError(f"duplicate model/condition/conversation: {identity}")
            seen.add(identity)
        if manifest_counts is None:
            manifest = [json.loads(line) for line in Path(cfg["manifest_path"]).read_text().splitlines()]
            manifest_counts = {"rows": len(manifest), "languages": len({r["released_language"] for r in manifest}),
                               "qc_pass_rows": sum(r.get("reference_qc") == "pass" for r in manifest),
                               "qc_pass_languages": len({r["language"] for r in manifest if r.get("reference_qc") == "pass"})}
        fragments.append({"name": directory.name, "key": key, "recipe": recipe, "predictions": predictions, "config": cfg})
    if len(manifest_hashes) > 1:
        raise ValueError("manifest hashes differ across runs")
    groups = {}
    for fragment in fragments:
        key = fragment["key"]
        if key in groups:
            if groups[key]["recipe"] != fragment["recipe"]:
                raise ValueError(f"mismatched recipe within run key {key}")
            groups[key]["predictions"].extend(fragment["predictions"])
            groups[key]["names"].append(fragment["name"])
        else:
            groups[key] = {**fragment, "names": [fragment["name"]]}
    runs = []
    for group in groups.values():
        predictions = group["predictions"]
        ids = [r.get("conversation_id") for r in predictions]
        if len(ids) != len(set(ids)): raise ValueError(f"duplicate IDs across shards for {group['key']}")
        sem = group["recipe"]["termination_semantics"]
        runs.append({"name": "+".join(group["names"]), "model_id": group["key"][0], "model_revision": group["key"][1], "condition_id": group["key"][2], "recipe": group["recipe"], "summary": _aggregate(predictions, sem),
                     "conversations": [{"conversation_id": r.get("conversation_id"), "language": r.get("language"),
                                         "status": r.get("status"), "metrics": r.get("metrics", {}), "wer": r.get("metrics", {}).get("wer"),
                                         "cer": r.get("metrics", {}).get("cer"), "sdi": sum(r.get("metrics", {}).get(k, 0) for k in ("substitutions", "deletions", "insertions")),
                                         "substitutions": r.get("metrics", {}).get("substitutions"), "deletions": r.get("metrics", {}).get("deletions"), "insertions": r.get("metrics", {}).get("insertions"),
                                         "capped_chunks": 0 if sem == "ctc_no_eos" else sum(c.get("eos_present") is False for c in r.get("chunks", [])), "rtf": r.get("real_time_factor"),
                                         "audio_sha256": r.get("audio_sha256"), "reference_sha256": r.get("reference_sha256"),
                                         "peak_cuda_allocated_bytes": r.get("peak_cuda_allocated_bytes")} for r in predictions]})
    comparisons = []
    for left, right in combinations(runs, 2):
        if left["model_id"] == right["model_id"] and left["model_revision"] == right["model_revision"] and left["condition_id"] == right["condition_id"]:
            continue
        la_all = {r["conversation_id"]: r for r in left["conversations"]}
        rb_all = {r["conversation_id"]: r for r in right["conversations"]}
        for cid in set(la_all) & set(rb_all):
            if (la_all[cid].get("audio_sha256"), la_all[cid].get("reference_sha256")) != (rb_all[cid].get("audio_sha256"), rb_all[cid].get("reference_sha256")):
                raise ValueError(f"hash mismatch for shared conversation {cid}")
        la = {r["conversation_id"]: r for r in left["conversations"] if r["wer"] is not None}
        rb = {r["conversation_id"]: r for r in right["conversations"] if r["wer"] is not None}
        comparisons.append({"a": left["name"], "b": right["name"], "common_cohort": _bootstrap(la, rb)})
    return {"contract": "open_benchmark_analysis_v1", "scope": "exploratory sampled corpus; not population evidence",
            "bootstrap": {"reps": BOOTSTRAP_REPS, "seed": BOOTSTRAP_SEED, "stratified_by": "language", "intervals": "descriptive percentile CI only; no p-values", "suppression_rule": "suppress CI below 5 conversations"},
            "protocol_counts": manifest_counts, "runs": runs, "comparisons": comparisons}


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dirs", nargs="+")
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-csv")
    args = parser.parse_args(argv)
    result = analyze(args.run_dirs)
    Path(args.output_json).write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    if args.output_csv:
        rows = [(run["name"], c) for run in result["runs"] for c in run["conversations"]]
        with Path(args.output_csv).open("w", newline="") as stream:
            fields = ["run", "conversation_id", "language", "status", "wer", "cer", "sdi", "substitutions", "deletions", "insertions", "capped_chunks", "rtf", "peak_cuda_allocated_bytes"]
            writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader()
            for run, row in rows:
                writer.writerow({"run": run, **{field: row.get(field) for field in fields if field != "run"}})
    print(json.dumps({"runs": len(result["runs"]), "comparisons": len(result["comparisons"]), "output_json": args.output_json}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
