"""Deterministic, development-only audit of completed selection-training runs.

Example:
  python -m research.pilot.evaluate_runs --manifest research/pilot/data/manifest.jsonl \
    --selections research/pilot/selections --schedule schedule.json \
    --run-dir results/full_seed0 --run-dir results/filtered_seed0 \
    --output results/evidence.json --packet results/evidence.md

Schedule: {"runs": [{"arm_name": "full_gold", "seed": 0}, ...],
           "fixed_config": {"num_updates": 200, "batch_size": 4, ...}}
Unspecified fixed_config settings use TrainConfig defaults. Cache/normalizer
paths and expected-hash guard fields are checked separately from scientific
settings. Missing scheduled runs produce incomplete evidence, not success.
This module makes no statistical promotion or publishability decision.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
import hashlib
import inspect
import itertools
import json
import math
from pathlib import Path
import random
import re
import unicodedata
from typing import Any

from research.pilot.frozen_english_normalizer import EnglishTextNormalizer
from research.pilot.training import TrainConfig

ROOT = Path(__file__).resolve().parents[2]
VARIABLE_CONFIG = {"arm_name", "seed", "feature_cache_dir", "normalizer_path",
                   "expected_initial_state_sha256", "expected_dev_manifest_sha256"}
IDENTITY_FIELDS = ("utt_id", "speaker_id", "native_language", "prompt_id", "reference_text", "audio_path", "split")
COUNTS = ("errors", "substitutions", "deletions", "insertions", "reference_words")
PRIMARY_ARMS = ("filtered10_gold", "random_matched_gold")


class EvidenceError(ValueError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise EvidenceError(message)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def reject_constant(value):
    raise EvidenceError(f"Non-finite JSON constant: {value}")


def load_json(path: Path):
    return json.loads(path.read_text(), parse_constant=reject_constant)


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line, parse_constant=reject_constant)
            for line in path.read_text().splitlines() if line.strip()]


def minimal(text: str) -> str:
    """Frozen training convention, distinct from the historical paper Minimal."""
    text = unicodedata.normalize("NFKC", text).lower()
    return " ".join("".join(" " if unicodedata.category(c).startswith("P") else c for c in text).split())


def edit_counts(reference: str, hypothesis: str) -> dict[str, int]:
    """Independent full-matrix traceback; S, then D, then I breaks cost ties."""
    a, b = reference.split(), hypothesis.split()
    require(bool(a), "Empty normalized reference")
    matrix = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]
    for i in range(len(a) + 1):
        matrix[i][0] = i
    for j in range(len(b) + 1):
        matrix[0][j] = j
    for i, word_a in enumerate(a, 1):
        for j, word_b in enumerate(b, 1):
            matrix[i][j] = matrix[i-1][j-1] if word_a == word_b else 1 + min(
                matrix[i-1][j-1], matrix[i-1][j], matrix[i][j-1])
    i, j, s, d, ins = len(a), len(b), 0, 0, 0
    while i or j:
        if i and j and a[i-1] == b[j-1]:
            i, j = i-1, j-1
        elif i and j and matrix[i][j] == matrix[i-1][j-1] + 1:
            s, i, j = s+1, i-1, j-1
        elif i and matrix[i][j] == matrix[i-1][j] + 1:
            d, i = d+1, i-1
        else:
            ins, j = ins+1, j-1
    return dict(errors=s+d+ins, substitutions=s, deletions=d, insertions=ins, reference_words=len(a))


def compare(actual, expected, context: str) -> None:
    if isinstance(expected, dict):
        require(isinstance(actual, dict) and set(actual) == set(expected), f"{context}: object fields differ")
        for key, value in expected.items():
            compare(actual[key], value, f"{context}.{key}")
    elif isinstance(expected, list):
        require(isinstance(actual, list) and len(actual) == len(expected), f"{context}: list lengths differ")
        for index, (left, right) in enumerate(zip(actual, expected)):
            compare(left, right, f"{context}[{index}]")
    elif isinstance(expected, float):
        require(isinstance(actual, (int, float)) and not isinstance(actual, bool)
                and math.isfinite(actual) and math.isclose(actual, expected, rel_tol=1e-11, abs_tol=1e-12),
                f"{context}: {actual!r} != {expected!r}")
    else:
        require(type(actual) is type(expected) and actual == expected, f"{context}: {actual!r} != {expected!r}")


def index_rows(rows: list[dict], context: str) -> dict[str, dict]:
    indexed = {}
    for row in rows:
        uid = row.get("utt_id")
        require(isinstance(uid, str) and uid and uid not in indexed, f"{context}: missing/duplicate utterance ID {uid}")
        indexed[uid] = row
    return indexed


def sorted_rows(rows: list[dict]) -> list[dict]:
    return sorted((dict(row, duration_s=float(row["duration_s"])) for row in rows), key=lambda row: row["utt_id"])


def identity_check(row: dict, expected: dict, context: str) -> None:
    for field in IDENTITY_FIELDS:
        compare(row.get(field), expected[field], f"{context}.{field}")
    require(math.isfinite(float(row["duration_s"])) and abs(float(row["duration_s"]) - float(expected["duration_s"])) <= 0.01,
            f"{context}: duration changed by more than10ms")


def aggregate(rows: list[dict]) -> dict:
    metrics = {}
    for convention in ("minimal", "english"):
        buckets = {"overall": rows}
        for row in rows:
            for name in ("language:" + row["native_language"], "split:" + row["split"]):
                buckets.setdefault(name, []).append(row)
        groups = {}
        for name, bucket in sorted(buckets.items()):
            errors = sum(row[convention]["errors"] for row in bucket)
            words = sum(row[convention]["reference_words"] for row in bucket)
            require(words > 0, "Zero reference denominator")
            groups[name] = dict(wer=errors/words, errors=errors, reference_words=words,
                                utterances=len(bucket), speakers=len({row["speaker_id"] for row in bucket}))
        languages = [group["wer"] for name, group in groups.items() if name.startswith("language:")]
        metrics[convention] = dict(micro_wer=groups["overall"]["wer"], macro_language_wer=sum(languages)/len(languages),
                                   worst_language_wer=max(languages), groups=groups)
    return metrics


def audit_predictions(path: Path, dev: dict[str, dict], update: int, english) -> tuple[dict, str]:
    rows = load_jsonl(path)
    indexed = index_rows(rows, str(path))
    require(set(indexed) == set(dev), f"{path}: prediction IDs must cover frozen dev exactly; missing={len(set(dev)-set(indexed))}, extra={len(set(indexed)-set(dev))}")
    scored = []
    for uid, row in indexed.items():
        expected = dev[uid]
        for field in ("speaker_id", "native_language", "prompt_id", "reference_text", "split"):
            compare(row.get(field), expected[field], f"{path}:{uid}.{field}")
        compare(row.get("update"), update, f"{path}:{uid}.update")
        require(isinstance(row.get("prediction"), str), f"{path}:{uid}: missing hypothesis")
        recomputed = dict(row)
        for name, normalizer in (("minimal", minimal), ("english", english)):
            recomputed[name] = edit_counts(normalizer(expected["reference_text"]), normalizer(row["prediction"]))
            compare(row.get(name), recomputed[name], f"{path}:{uid}.{name}")
        scored.append(recomputed)
    return aggregate(scored), canonical_hash(sorted((uid, row["prediction"]) for uid, row in indexed.items()))


def validate_schedule(schedule: dict) -> tuple[list[tuple[str, int]], dict]:
    require(isinstance(schedule.get("runs"), list) and schedule["runs"], "Schedule requires expected runs")
    expected = []
    for run in schedule["runs"]:
        arm, seed = run.get("arm_name"), run.get("seed")
        require(isinstance(arm, str) and re.fullmatch(r"[A-Za-z0-9_-]+", arm) is not None and "smoke" not in arm.lower(),
                "Smoke runs are separate operational checks and cannot enter scientific evidence")
        require(type(seed) is int and seed >= 0, "Invalid scheduled seed")
        expected.append((arm, seed))
    require(len(set(expected)) == len(expected), "Duplicate scheduled arm/seed")
    arms = {arm for arm, _ in expected}
    require({"full_gold", *PRIMARY_ARMS}.issubset(arms), "Schedule must include full, filtered and matched-random controls")
    for seed in {seed for _, seed in expected}:
        require({arm for arm, value in expected if value == seed} == arms, "Every registered seed requires the same arms")
    require(isinstance(schedule.get("fixed_config"), dict), "Schedule requires fixed_config")
    require(not (set(schedule["fixed_config"]) & VARIABLE_CONFIG), "fixed_config cannot contain arm, seed or run-path fields")
    cfg = TrainConfig(**schedule["fixed_config"])
    cfg.validate()
    return expected, {key: value for key, value in asdict(cfg).items() if key not in VARIABLE_CONFIG}


def expected_exposures(train: list[dict], cfg: dict) -> list[list[str]]:
    total = cfg["num_updates"] * cfg["batch_size"] * cfg["grad_accum_steps"]
    rng, stream = random.Random(cfg["seed"]), []
    while len(stream) < total:
        indices = list(range(len(train)))
        rng.shuffle(indices)
        stream.extend(train[index]["utt_id"] for index in reversed(indices))
    width = cfg["batch_size"] * cfg["grad_accum_steps"]
    return [stream[start:start+width] for start in range(0, total, width)]


def audit_run(directory: Path, main: dict[str, dict], selected: list[dict], dev_rows: list[dict],
              expected_config: dict, english, code_hash: str, normalizer_hash: str, normalizer_code_hash: str) -> dict:
    summary = load_json(directory / "summary.json")
    require(summary.get("status") == "completed", f"{directory}: run not completed")
    cfg = summary["config"]
    TrainConfig(**cfg).validate()
    compare({key: value for key, value in cfg.items() if key not in VARIABLE_CONFIG}, expected_config, f"{directory}.config")
    train = sorted_rows(selected)
    expected_manifest = train + dev_rows
    actual_manifest = load_json(directory / "manifest.json")
    index_rows(actual_manifest, f"{directory}.manifest")
    require(all(row["split"] in ("train", "dev") for row in actual_manifest), f"{directory}: nonprimary/test row in run manifest")
    compare(sorted_rows([r for r in actual_manifest if r["split"] == "train"]) + sorted_rows([r for r in actual_manifest if r["split"] == "dev"]),
            expected_manifest, f"{directory}.manifest")
    expected_dev_hash = canonical_hash(dev_rows)
    for key, expected in (("train_manifest_sha256", canonical_hash(train)), ("dev_manifest_sha256", expected_dev_hash),
                          ("training_code_sha256", code_hash), ("normalizer_sha256", normalizer_hash),
                          ("normalizer_implementation_sha256", normalizer_code_hash)):
        compare(summary.get(key), expected, f"{directory}.{key}")
    if cfg.get("expected_dev_manifest_sha256"):
        compare(cfg["expected_dev_manifest_sha256"], expected_dev_hash, f"{directory}.config.dev_guard")
    for field in ("initial_state_sha256", "initial_encoder_sha256", "final_encoder_sha256"):
        require(re.fullmatch(r"[0-9a-f]{64}", summary.get(field, "")) is not None, f"{directory}: invalid {field}")
    if cfg.get("expected_initial_state_sha256"):
        compare(summary["initial_state_sha256"], cfg["expected_initial_state_sha256"], f"{directory}.initial_guard")
    require(summary["initial_encoder_sha256"] != summary["final_encoder_sha256"], f"{directory}: encoder unchanged")
    for key in ("encoder_parameter_l2_change", "encoder_relative_l2_change"):
        require(math.isfinite(summary.get(key, 0)) and summary[key] > 0, f"{directory}: encoder update not established")
    require(summary.get("encoder_trainable_parameters", 0) > 0 and summary.get("trainable_parameters", 0) > summary["encoder_trainable_parameters"],
            f"{directory}: encoder and decoder must both be trainable")
    generation = dict(language="en", task="transcribe", return_timestamps=False, max_new_tokens=224,
                      do_sample=False, num_beams=1, use_cache=True)
    compare(summary.get("generation_kwargs"), generation, f"{directory}.decoder")
    provenance = load_json(directory / "provenance.json")
    for key, value in provenance.items():
        compare(summary.get(key), value, f"{directory}.provenance.{key}")
    require(set(provenance) >= {"config", "training_code_sha256", "normalizer_implementation_sha256"}, f"{directory}: incomplete provenance")
    history = load_jsonl(directory / "history.jsonl")
    compare(summary.get("history"), history, f"{directory}.history")
    require(all(row.get("event") in ("train_update", "dev_eval") for row in history), f"{directory}: unknown history event")
    updates = [row for row in history if row["event"] == "train_update"]
    require([row["update"] for row in updates] == list(range(1, cfg["num_updates"]+1)), f"{directory}: missing/duplicate/out-of-order updates")
    exposures, expected_batches = Counter(), expected_exposures(train, cfg)
    audio = summary["actual_audio"]
    total_audio, total_tokens = 0.0, 0
    for row, expected_ids in zip(updates, expected_batches):
        compare(row.get("utterance_ids"), expected_ids, f"{directory}.update{row['update']}.exposures")
        exposures.update(expected_ids)
        compare(row.get("unique_seen"), len(exposures), f"{directory}.update.unique_seen")
        for field in ("loss", "pre_clip_grad_norm"):
            require(isinstance(row.get(field), (int, float)) and math.isfinite(row[field]) and row[field] >= 0, f"{directory}: invalid {field}")
        compare(row.get("learning_rate"), cfg["learning_rate"], f"{directory}.learning_rate")
        require(type(row.get("supervised_tokens")) is int and row["supervised_tokens"] > 0, f"{directory}: invalid token exposure")
        total_tokens += row["supervised_tokens"]
        seconds = sum(audio[uid]["duration_s"] for uid in expected_ids)
        compare(row.get("audio_seconds"), seconds, f"{directory}.update.audio_seconds")
        total_audio += seconds
    require(set(audio) == set(exposures) | {row["utt_id"] for row in dev_rows}, f"{directory}: unexpected/missing decoded-audio IDs")
    for uid, metadata in audio.items():
        require(uid in main and re.fullmatch(r"[0-9a-f]{64}", metadata.get("audio_sha256", "")) is not None, f"{directory}: invalid audio identity")
        require(metadata.get("original_sample_rate") == 44100 and metadata.get("channels") == 1, f"{directory}: unexpected source audio format")
        require(math.isfinite(metadata["duration_s"]) and abs(metadata["duration_s"] - main[uid]["duration_s"]) <= 0.01,
                f"{directory}: decoded duration disagrees with main manifest")
    totals = dict(completed_updates=cfg["num_updates"], train_pool_utterances=len(train), dev_utterances=len(dev_rows),
                  unique_training_utterances_seen=len(exposures), total_training_exposures=sum(exposures.values()),
                  total_supervised_tokens=total_tokens, total_training_audio_seconds=total_audio,
                  exposures_by_utterance=dict(sorted(exposures.items())))
    for key, value in totals.items():
        compare(summary.get(key), value, f"{directory}.{key}")
    expected_evals = [("baseline", 0)] + [(f"update_{n:06d}", n) for n in range(1, cfg["num_updates"])
                      if cfg["eval_every"] and n % cfg["eval_every"] == 0] + [("final", cfg["num_updates"])]
    evaluations = [row for row in history if row["event"] == "dev_eval"]
    require([(row.get("label"), row.get("update")) for row in evaluations] == expected_evals, f"{directory}: unexpected evaluation/query schedule")
    expected_files = {f"{label}_dev_predictions.jsonl" for label, _ in expected_evals}
    require({path.name for path in directory.glob('*_dev_predictions.jsonl')} == expected_files, f"{directory}: unexpected prediction artifact")
    dev = {row["utt_id"]: row for row in dev_rows}
    metrics, hypothesis_hashes = {}, {}
    for event in evaluations:
        label, update = event["label"], event["update"]
        filename = f"{label}_dev_predictions.jsonl"
        compare(event.get("prediction_file"), filename, f"{directory}.prediction_file")
        recomputed, predictions_hash = audit_predictions(directory / filename, dev, update, english)
        compare(event["metrics"], recomputed, f"{directory}.{label}.history_metrics")
        compare(load_json(directory / f"{label}_dev_metrics.json"), recomputed, f"{directory}.{label}.metrics_file")
        if label in ("baseline", "final"):
            compare(summary[f"{label}_dev_metrics"], recomputed, f"{directory}.{label}.summary_metrics")
        metrics[label], hypothesis_hashes[label] = recomputed, predictions_hash
    compare(summary.get("prediction_files"), {"baseline":"baseline_dev_predictions.jsonl", "final":"final_dev_predictions.jsonl"}, f"{directory}.prediction_files")
    files = ["summary.json", "provenance.json", "manifest.json", "history.jsonl"] + sorted(expected_files)
    return dict(arm_name=cfg["arm_name"], seed=cfg["seed"], directory=str(directory.resolve()),
                initial_state_sha256=summary["initial_state_sha256"], initial_encoder_sha256=summary["initial_encoder_sha256"],
                baseline_hypotheses_sha256=hypothesis_hashes["baseline"], dev_manifest_sha256=expected_dev_hash,
                runtime=summary["runtime"], feature_signature=summary["feature_signature"],
                decoder_prefix_ids=summary["decoder_prefix_ids"], generation_config=summary["generation_config"],
                metrics=metrics, exposure=totals, dev_evaluation_queries=len(evaluations),
                encoder_relative_l2_change=summary["encoder_relative_l2_change"],
                audio_fingerprints={uid: value["audio_sha256"] for uid, value in audio.items()},
                artifact_sha256={name: sha(directory/name) for name in files})


def evaluate(run_dirs: list[Path], manifest_path: Path, selections_dir: Path, schedule: dict,
             *, expected_dev_count: int = 974, normalizer_path: Path | None = None,
             training_code_path: Path | None = None) -> dict:
    expected, fixed_config = validate_schedule(schedule)
    normalizer_path = normalizer_path or ROOT / "paper/data/whisper_normalizer.json"
    training_code_path = training_code_path or ROOT / "research/pilot/training.py"
    english = EnglishTextNormalizer(load_json(normalizer_path))
    for relative, expected_hash in schedule.get("frozen_inputs", {}).items():
        source = (ROOT / relative).resolve()
        require(source.is_relative_to(ROOT) and source.is_file(), f"Missing/invalid frozen input {relative}")
        compare(sha(source), expected_hash, f"Frozen input {relative}")
    supplied_sources = {"research/pilot/data/manifest.jsonl": manifest_path,
                        "research/pilot/training.py": training_code_path,
                        "paper/data/whisper_normalizer.json": normalizer_path}
    for name, supplied in supplied_sources.items():
        if name in schedule.get("frozen_inputs", {}):
            compare(sha(supplied), schedule["frozen_inputs"][name], f"Supplied frozen input {name}")
    main_rows = load_jsonl(manifest_path)
    main = index_rows(main_rows, "main manifest")
    require({row["split"] for row in main_rows} == {"train", "dev", "test"}, "Main manifest must contain primary train/dev/test")
    dev_rows = sorted_rows([row for row in main_rows if row["split"] == "dev"])
    require(len(dev_rows) == expected_dev_count, f"Expected exact frozen dev count {expected_dev_count}, got {len(dev_rows)}")
    overlap = {}
    for field in ("speaker_id", "prompt_id", "minimal_reference"):
        sets = {split: {minimal(row["reference_text"]) if field == "minimal_reference" else row[field]
                        for row in main_rows if row["split"] == split} for split in ("train", "dev", "test")}
        overlap[field] = {f"{a}:{b}": len(sets[a] & sets[b]) for a, b in itertools.combinations(sets, 2)}
        require(not any(overlap[field].values()), f"Main manifest {field} leakage")
    selected = {}
    for arm in sorted({arm for arm, _ in expected}):
        rows = load_jsonl(selections_dir / f"{arm}.jsonl")
        index_rows(rows, f"selection:{arm}")
        require(bool(rows), f"Empty selected arm {arm}")
        for row in rows:
            require(row["utt_id"] in main and main[row["utt_id"]]["split"] == "train", f"{arm}: nontraining selection")
            identity_check(row, main[row["utt_id"]], f"{arm}:{row['utt_id']}")
        selected[arm] = rows
    require({row["utt_id"] for row in selected["full_gold"]} == {row["utt_id"] for row in main_rows if row["split"] == "train"},
            "full_gold must include the entire frozen training pool")
    def stratum_counts(rows):
        def word_bin(row):
            count = len(english(row["reference_text"]).split())
            return 0 if count < 5 else 1 if count < 10 else 2 if count < 20 else 3
        return Counter((row["speaker_id"], word_bin(row)) for row in rows)
    require(stratum_counts(selected[PRIMARY_ARMS[0]]) == stratum_counts(selected[PRIMARY_ARMS[1]]),
            "Filtered/random controls differ in speaker by reference-length counts")
    seen, audited, incomplete = set(), [], []
    for directory in map(Path, run_dirs):
        if not (directory / "summary.json").exists():
            incomplete.append(str(directory)); continue
        summary = load_json(directory / "summary.json")
        key = (summary["config"]["arm_name"], summary["config"]["seed"])
        require(key in expected and key not in seen, f"Unexpected/duplicate arm and seed {key}; smoke cannot count")
        seen.add(key)
        result = audit_run(directory, main, selected[key[0]], dev_rows, fixed_config, english,
                           sha(training_code_path), sha(normalizer_path), sha(Path(inspect.getfile(EnglishTextNormalizer))))
        if schedule.get("expected_initial_state_sha256"):
            compare(result["initial_state_sha256"], schedule["expected_initial_state_sha256"], "Registered initial state")
        if audited:
            for field in ("initial_state_sha256", "initial_encoder_sha256", "baseline_hypotheses_sha256", "dev_manifest_sha256", "runtime",
                          "feature_signature", "decoder_prefix_ids", "generation_config"):
                compare(result[field], audited[0][field], f"Cross-run {field}")
            for previous in audited:
                common = set(previous["audio_fingerprints"]) & set(result["audio_fingerprints"])
                for uid in common:
                    compare(result["audio_fingerprints"][uid], previous["audio_fingerprints"][uid], f"Cross-run audio:{uid}")
        audited.append(result)
    audited.sort(key=lambda row: (row["arm_name"], row["seed"]))
    indexed = {(row["arm_name"], row["seed"]): row for row in audited}
    contrasts = []
    for seed in sorted({seed for _, seed in expected}):
        a, b = ((arm, seed) for arm in PRIMARY_ARMS)
        if a not in indexed or b not in indexed:
            continue
        for convention in ("english", "minimal"):
            left, right = indexed[a]["metrics"]["final"][convention], indexed[b]["metrics"]["final"][convention]
            contrasts.append(dict(seed=seed, normalization=convention,
                filtered_minus_random_macro_wer_pp=100*(left["macro_language_wer"]-right["macro_language_wer"]),
                filtered_minus_random_micro_wer_pp=100*(left["micro_wer"]-right["micro_wer"]),
                by_native_language_pp={key.removeprefix("language:"): 100*(value["wer"]-right["groups"][key]["wer"])
                                      for key, value in left["groups"].items() if key.startswith("language:")}))
    train_overlap = {f"{a}:{b}": len({row["utt_id"] for row in selected[a]} & {row["utt_id"] for row in selected[b]})
                     for a, b in itertools.combinations(sorted(selected), 2)}
    missing = [dict(arm_name=arm, seed=seed) for arm, seed in expected if (arm, seed) not in seen]
    return dict(status="incomplete" if missing or incomplete else "valid_complete", scientific_promotion=False,
                scope="Development-only pilot evidence; no final-test predictions read or scored",
                primary_contrast="filtered10_gold minus random_matched_gold, final English L1-macro WER; positive means filtered worse",
                expected_runs=[dict(arm_name=arm, seed=seed) for arm, seed in expected], missing_runs=missing,
                incomplete_directories=incomplete, runs=audited, paired_contrasts=contrasts,
                source_sha256=dict(main_manifest=sha(manifest_path), schedule=canonical_hash(schedule),
                    evaluator=sha(Path(__file__)), training_code=sha(training_code_path), normalizer_map=sha(normalizer_path),
                    normalizer_implementation=sha(Path(inspect.getfile(EnglishTextNormalizer)))),
                selection_sha256={arm: sha(selections_dir/f"{arm}.jsonl") for arm in selected},
                primary_overlap_counts=overlap, selected_train_overlap_counts=train_overlap,
                dev_utterances=len(dev_rows), dev_speakers=len({row["speaker_id"] for row in dev_rows}),
                baseline_hypothesis_variants=len({run["baseline_hypotheses_sha256"] for run in audited}),
                limitations=["Recorded weight fingerprints and encoder changes are checked; no checkpoint tensors were loaded by this auditor.",
                    "Development evaluations are exploratory, including all logged intermediate queries.",
                    "Provided transcripts are not newly audio-adjudicated; one development speaker per L1 limits population inference."])


def evidence_markdown(report: dict) -> str:
    lines = ["# Controlled ASR pilot: evidence packet", "", f"Evidence status: **{report['status']}**. No scientific promotion decision is made.", "",
             report["scope"], "", report["primary_contrast"], "", "## All observed runs", "",
             "| Arm | Seed | Normalizer | Baseline micro WER % | Final micro WER % | Final L1 macro WER % | Worst L1 WER % |", "|---|---:|---|---:|---:|---:|---:|"]
    for run in report["runs"]:
        for normalization in ("english", "minimal"):
            before, after = run["metrics"]["baseline"][normalization], run["metrics"]["final"][normalization]
            lines.append(f"| {run['arm_name']} | {run['seed']} | {normalization} | {100*before['micro_wer']:.3f} | {100*after['micro_wer']:.3f} | {100*after['macro_language_wer']:.3f} | {100*after['worst_language_wer']:.3f} |")
    lines += ["", "## Every L1 and seed", "", "| Arm | Seed | Normalizer | L1 | Final WER % | Clips | Speakers |", "|---|---:|---|---|---:|---:|---:|"]
    for run in report["runs"]:
        for normalization in ("english", "minimal"):
            for name, group in run["metrics"]["final"][normalization]["groups"].items():
                if name.startswith("language:"):
                    lines.append(f"| {run['arm_name']} | {run['seed']} | {normalization} | {name[9:]} | {100*group['wer']:.3f} | {group['utterances']} | {group['speakers']} |")
    lines += ["", "## Paired mechanism contrast", "", "| Seed | Normalizer | Filtered minus random L1 macro WER (pp) |", "|---:|---|---:|"]
    for contrast in report["paired_contrasts"]:
        lines.append(f"| {contrast['seed']} | {contrast['normalization']} | {contrast['filtered_minus_random_macro_wer_pp']:+.3f} |")
    lines += ["", "## Exposure and overlap", "", "| Arm | Seed | Updates | Unique seen | Total examples presented | Audio hours presented | Dev queries |", "|---|---:|---:|---:|---:|---:|---:|"]
    for run in report["runs"]:
        exposure = run["exposure"]
        lines.append(f"| {run['arm_name']} | {run['seed']} | {exposure['completed_updates']} | {exposure['unique_training_utterances_seen']} | {exposure['total_training_exposures']} | {exposure['total_training_audio_seconds']/3600:.4f} | {run['dev_evaluation_queries']} |")
    lines += ["", "Primary split overlap counts: `" + json.dumps(report["primary_overlap_counts"], sort_keys=True) + "`.", "",
              "Training-set overlaps: `" + json.dumps(report["selected_train_overlap_counts"], sort_keys=True) + "`.", "",
              "Missing scheduled runs: `" + json.dumps(report["missing_runs"]) + "`.", "",
              "Incomplete directories: `" + json.dumps(report["incomplete_directories"]) + "`.", "", "## Limits", ""]
    lines += ["- " + note for note in report["limitations"]]
    lines += ["", "The accompanying JSON records exact source, evaluator and run-artifact hashes. A valid packet is an integrity result, not evidence that the hypothesis is true or the work is publishable.", ""]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--selections", type=Path, required=True)
    parser.add_argument("--schedule", type=Path, required=True)
    parser.add_argument("--run-dir", action="append", type=Path, default=[])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--packet", type=Path, required=True)
    args = parser.parse_args()
    try:
        report = evaluate(args.run_dir, args.manifest, args.selections, load_json(args.schedule))
    except (EvidenceError, KeyError, TypeError, ValueError, OSError) as error:
        parser.exit(2, f"Evidence validation failed: {error}\n")
    for path, content in ((args.output, json.dumps(report, indent=2, sort_keys=True, allow_nan=False)+"\n"),
                          (args.packet, evidence_markdown(report))):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
    print(json.dumps({key: report[key] for key in ("status", "missing_runs", "dev_utterances", "scientific_promotion")}))
    return 0 if report["status"] == "valid_complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
