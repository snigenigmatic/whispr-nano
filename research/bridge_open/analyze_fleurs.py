"""Fail-closed offline comparison for completed FLEURS test artifacts."""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from research.bridge_open.scoring import normalize_v1, score


TERMINAL = {"success", "empty_output", "generation_limit"}


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _valid_validation(path: Path) -> bool:
    report = json.loads(path.read_text())
    return report.get("valid") is True or report.get("validation") == "pass"


def _model_label(model_id: str) -> str:
    value = model_id.lower()
    for name in ("turbo", "large", "small"):
        if name in value:
            return name.title()
    return model_id


def _metric_counts(metric: dict[str, Any]) -> tuple[int, int, int, int]:
    return (metric["substitutions"] + metric["deletions"] + metric["insertions"], metric["ref_words"],
            metric["character_errors"], metric["ref_chars"])


def cluster_bootstrap_difference(
    left: list[dict[str, Any]], right: list[dict[str, Any]], *, seed: int = 12, draws: int = 2000,
) -> dict[str, Any]:
    """Bootstrap paired WER differences by sentence, never individual recordings."""
    by_sentence: dict[str, list[tuple[dict[str, Any], dict[str, Any]]]] = defaultdict(list)
    right_by_id = {row["conversation_id"]: row for row in right}
    for row in left:
        other = right_by_id.get(row["conversation_id"])
        if other is None:
            raise ValueError("paired models do not contain the same utterance IDs")
        by_sentence[row["source_sentence_id"]].append((row, other))
    clusters = list(by_sentence.values())
    if not clusters:
        raise ValueError("no sentence clusters")
    rng = np.random.default_rng(seed)

    def difference(sample: list[list[tuple[dict[str, Any], dict[str, Any]]]]) -> float:
        le = lr = re = rr = 0
        for cluster in sample:
            for a, b in cluster:
                ae, ar, _, _ = _metric_counts(a["metrics"])
                be, br, _, _ = _metric_counts(b["metrics"])
                le += ae; lr += ar; re += be; rr += br
        return le / lr - re / rr

    observed = difference(clusters)
    samples = np.array([difference([clusters[i] for i in rng.integers(0, len(clusters), len(clusters))]) for _ in range(draws)])
    return {"metric": "micro_wer", "difference": observed, "ci_95": [float(np.quantile(samples, .025)), float(np.quantile(samples, .975))],
            "draws": draws, "seed": seed, "cluster_count": len(clusters),
            "interpretation": "sentence-cluster bootstrap only; not speaker or population fairness inference"}


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    errors = words = char_errors = chars = 0
    for row in rows:
        e, w, ce, c = _metric_counts(row["metrics"])
        errors += e; words += w; char_errors += ce; chars += c
    return {"utterances": len(rows), "scored_count": len(rows), "word_errors": errors, "ref_words": words, "character_errors": char_errors,
            "ref_chars": chars, "micro_wer": errors / words, "micro_cer": char_errors / chars,
            "success_count": sum(row["status"] == "success" for row in rows),
            "empty_output_status_count": sum(row["status"] == "empty_output" for row in rows),
            # This diagnosis intentionally overlaps capped outputs: punctuation
            # loops can normalize to empty while remaining scored cap evidence.
            "normalized_empty_count": sum(not normalize_v1(row.get("text_raw", "")) for row in rows),
            "generation_limit_count": sum(row["status"] == "generation_limit" for row in rows)}


def _batch_summary(run_dirs: list[Path]) -> dict[str, Any]:
    unique: dict[tuple[Any, ...], dict[str, Any]] = {}
    for output_dir in run_dirs:
        event_path = output_dir / "batch_inference.jsonl"
        if not event_path.exists():
            continue
        for event in _jsonl(event_path):
            batch_id = event.get("batch_id")
            if not isinstance(batch_id, str) or not batch_id:
                raise ValueError(f"{event_path}: batch event lacks a batch_id")
            # batch_id identity is only meaningful within its source run.  A
            # shard may legitimately use the same name as another run.
            key = (str(output_dir.resolve()), batch_id)
            if key in unique:
                if unique[key] != event:
                    raise ValueError(f"{event_path}: conflicting duplicate batch event {batch_id}")
                continue
            unique[key] = event
    usable = [event for event in unique.values() if isinstance(event.get("inference_wall_seconds"), (int, float)) and event["inference_wall_seconds"] > 0 and isinstance(event.get("audio_seconds"), (int, float))]
    audio = sum(event["audio_seconds"] for event in usable)
    wall = sum(event["inference_wall_seconds"] for event in usable)
    return {"unique_batches": len(unique), "timed_batches": len(usable), "audio_seconds": audio, "inference_wall_seconds": wall,
            "audio_seconds_per_wall_second": audio / wall if wall else None,
            "note": "Unique batch throughput only; this is not per-request latency."}


def load_model(manifest: list[dict[str, Any]], model_id: str, revision: str, run_dirs: list[Path]) -> list[dict[str, Any]]:
    expected = {row["conversation_id"]: row for row in manifest}
    found: dict[str, dict[str, Any]] = {}
    for output_dir in run_dirs:
        validation = output_dir / "independent_validation.json"
        predictions = output_dir / "predictions.jsonl"
        if not validation.is_file() or not _valid_validation(validation):
            raise ValueError(f"{output_dir}: missing or failing independent_validation.json")
        if not predictions.is_file():
            raise ValueError(f"{output_dir}: missing predictions.jsonl")
        for prediction in _jsonl(predictions):
            cid = prediction.get("conversation_id")
            if cid in found:
                raise ValueError(f"{model_id}: duplicate prediction for {cid}")
            if cid not in expected:
                raise ValueError(f"{model_id}: unexpected prediction ID {cid}")
            source = expected[cid]
            if prediction.get("model_id") != model_id or prediction.get("model_revision") != revision:
                raise ValueError(f"{cid}: pinned model identity mismatch")
            if prediction.get("audio_sha256") != source.get("audio_sha256") or prediction.get("reference_sha256") != source.get("reference_sha256"):
                raise ValueError(f"{cid}: source hash mismatch")
            if prediction.get("status") not in TERMINAL:
                raise ValueError(f"{cid}: incomplete status {prediction.get('status')!r}")
            reference = json.loads(Path(source["reference_path"]).read_text())["reference_text"]
            recomputed = score(reference, prediction.get("text_raw", ""))
            if prediction.get("metrics") != recomputed:
                raise ValueError(f"{cid}: metrics do not match source-reference recomputation")
            route = source.get("inference_route")
            if not isinstance(route, str) or not route:
                raise ValueError(f"{cid}: manifest lacks inference_route")
            found[cid] = {**prediction, "source_sentence_id": str(source["source_sentence_id"]),
                          "language": source["language"], "inference_route": route}
    if set(found) != set(expected):
        missing = sorted(set(expected) - set(found))
        raise ValueError(f"{model_id}: incomplete cohort, missing {missing}")
    return [found[row["conversation_id"]] for row in manifest]


def analyze(manifest_path: str | Path, registry_path: str | Path, output_dir: str | Path) -> dict[str, Any]:
    manifest = _jsonl(Path(manifest_path))
    if len({row["conversation_id"] for row in manifest}) != len(manifest):
        raise ValueError("manifest contains duplicate IDs")
    registry = json.loads(Path(registry_path).read_text())
    grouped: dict[tuple[str, str], list[Path]] = defaultdict(list)
    for item in registry:
        grouped[(item["model_id"], item["model_revision"])].append(Path(item["output_dir"]))
    if not grouped:
        raise ValueError("registry is empty")
    models = {}
    records = {}
    for (model_id, revision), dirs in grouped.items():
        rows = load_model(manifest, model_id, revision, dirs)
        label = _model_label(model_id)
        key = f"{label} ({model_id}@{revision})"
        records[key] = rows
        languages = {}
        for language in sorted({row["language"] for row in rows}):
            language_rows = [row for row in rows if row["language"] == language]
            languages[language] = {"combined": _aggregate(language_rows), "by_inference_route": {
                route: _aggregate([row for row in language_rows if row["inference_route"] == route])
                for route in sorted({row["inference_route"] for row in language_rows})}}
        models[key] = {"model_id": model_id, "model_revision": revision, "output_dirs": [str(directory) for directory in dirs], "per_language": languages, "throughput": _batch_summary(dirs)}
    pairs = []
    turbo = [key for key in models if key.startswith("Turbo (")]
    for other_kind in ("Large", "Small"):
        for left in turbo:
            for right in [key for key in models if key.startswith(other_kind + " (")]:
                for language in sorted({row["language"] for row in records[left]}):
                    pairs.append({"left": left, "right": right, "language": language,
                        **cluster_bootstrap_difference([row for row in records[left] if row["language"] == language], [row for row in records[right] if row["language"] == language])})
    report = {"scope": "Descriptive public FLEURS benchmark; not population, speaker, or fairness evidence.", "manifest": str(manifest_path),
              "models": models, "paired_differences": pairs, "bootstrap": {"draws": 2000, "seed": 12, "unit": "source_sentence_id cluster within language"}}
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    (output / "comparison.json").write_text(json.dumps(report, indent=2) + "\n")
    with (output / "per_utterance.tsv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["model", "conversation_id", "language", "inference_route", "source_sentence_id", "status", "wer", "cer"] , delimiter="\t", lineterminator="\n")
        writer.writeheader()
        for model, rows in records.items():
            for row in rows:
                writer.writerow({"model": model, "conversation_id": row["conversation_id"], "language": row["language"], "inference_route": row["inference_route"], "source_sentence_id": row["source_sentence_id"], "status": row["status"], "wer": row["metrics"]["wer"], "cer": row["metrics"]["cer"]})
    lines = ["# FLEURS comparison", "", report["scope"], "", "## Per-language combined micro metrics", "", "| Model | Language | WER | CER | Scored | Capped | Normalized-empty* | Empty-output status |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for model, value in models.items():
        for language, report_language in value["per_language"].items():
            metric = report_language["combined"]
            lines.append(f"| {model} | {language} | {metric['micro_wer']:.2%} | {metric['micro_cer']:.2%} | {metric['scored_count']} | {metric['generation_limit_count']} | {metric['normalized_empty_count']} | {metric['empty_output_status_count']} |")
    lines += ["", "*Normalized-empty counts overlap capped outputs; they diagnose an empty scoring string after fixed normalization, not an exclusive terminal status.", "", "## Route-specific micro metrics", "", "| Model | Language | Route | WER | CER | Scored | Capped | Normalized-empty* | Empty-output status |", "|---|---:|---|---:|---:|---:|---:|---:|---:|"]
    for model, value in models.items():
        for language, report_language in value["per_language"].items():
            for route, metric in report_language["by_inference_route"].items():
                lines.append(f"| {model} | {language} | {route} | {metric['micro_wer']:.2%} | {metric['micro_cer']:.2%} | {metric['scored_count']} | {metric['generation_limit_count']} | {metric['normalized_empty_count']} | {metric['empty_output_status_count']} |")
    lines += ["", "## Paired WER differences", "", "Difference is left minus right in percentage points. Intervals use 2,000 seed-12 source-sentence cluster bootstrap draws within language.", "", "| Left | Right | Language | Difference | 95% interval | Sentence clusters |", "|---|---|---:|---:|---:|---:|"]
    for pair in pairs:
        lines.append(f"| {pair['left']} | {pair['right']} | {pair['language']} | {pair['difference'] * 100:+.2f} pp | [{pair['ci_95'][0] * 100:+.2f}, {pair['ci_95'][1] * 100:+.2f}] pp | {pair['cluster_count']} |")
    lines += ["", "## Batch throughput", "", "Batch throughput is aggregate audio seconds divided by aggregate timed batch wall seconds. It is not per-request latency.", "", "| Model | Unique batches | Timed batches | Audio seconds | Batch wall seconds | Throughput |", "|---|---:|---:|---:|---:|---:|"]
    for model, value in models.items():
        throughput = value["throughput"]
        rate = "n/a" if throughput["audio_seconds_per_wall_second"] is None else f"{throughput['audio_seconds_per_wall_second']:.2f} audio s/wall s"
        lines.append(f"| {model} | {throughput['unique_batches']} | {throughput['timed_batches']} | {throughput['audio_seconds']:.2f} | {throughput['inference_wall_seconds']:.2f} | {rate} |")
    lines += ["", "These descriptive public-benchmark summaries retain empty and generation-limited outputs. They do not support speaker, population, or fairness claims.", ""]
    (output / "RESULTS.md").write_text("\n".join(lines))
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--registry", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args(argv)
    analyze(args.manifest, args.registry, args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
