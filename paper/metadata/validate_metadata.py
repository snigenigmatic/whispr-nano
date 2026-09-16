"""Verify pinned row identities and produce metadata for the ASR audit.

The output recording tags are inferred from explicit ``_fN_chunk_N`` file
names. They are never described as speaker identifiers. This script does not
infer speaker identities from demographics, text, or audio.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

from extract_svarah_metadata import DATASET_REVISION


ROOT = Path(__file__).resolve().parents[2]
TAG_RE = re.compile(r"_(f\d+)_chunk_\d+(?:\.[^.]+)?$")
PROFILE_FIELDS = [
    "gender", "age-group", "primary_language", "native_place_state",
    "native_place_district", "highest_qualification", "job_category", "occupation_domain",
]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def summary(rows: list[dict]) -> dict:
    tags = Counter(row["recording_cluster_id"] for row in rows)
    return {
        "n_utterances": len(rows),
        "n_recording_clusters": len(tags),
        "n_numeric_prefix_clusters": len({row["coarse_recording_cluster_id"] for row in rows}),
        "n_demographic_profiles_unverified": len({row["demographic_profile_id"] for row in rows}),
        "n_languages": len({row["primary_language"] for row in rows}),
        "n_families": len({row["family"] for row in rows}),
        "n_singleton_recording_clusters": sum(n == 1 for n in tags.values()),
        "maximum_utterances_per_recording_cluster": max(tags.values(), default=0),
        "duration_seconds": sum(float(row["duration_s"]) for row in rows),
        "true_speaker_count": None,
    }


def validate(raw_path: Path, manifest_path: Path, output_dir: Path) -> dict:
    raw = read_jsonl(raw_path)
    provenance_path = raw_path.with_suffix(".provenance.json")
    provenance = json.loads(provenance_path.read_text())
    if provenance["dataset_revision"] != DATASET_REVISION:
        raise ValueError("Dataset revision differs from the frozen revision")
    if provenance.get("audio_bytes_exported") is not False:
        raise ValueError("Metadata extraction did not confirm exclusion of audio bytes")
    if len(raw) != 6656:
        raise ValueError(f"Expected 6656 records, received {len(raw)}")
    by_id = {row["utt_id"]: row for row in raw}
    if len(by_id) != len(raw):
        raise ValueError("Duplicate metadata utterance IDs")
    for index, row in enumerate(raw):
        if row["utt_id"] != f"svarah_{index:05d}" or row["dataset_index"] != index:
            raise ValueError(f"Unexpected metadata row order at index {index}")

    manifest = json.loads(manifest_path.read_text())
    checked_ids = set()
    family_by_language = {}
    source_checks = []
    maximum_duration_difference_s = 0.0
    for filename, expected in manifest["source_files"].items():
        source = ROOT / "audit" / "svarah_fairness_audit" / filename
        actual_hash = sha256(source)
        if actual_hash != expected["sha256"]:
            raise ValueError(f"Frozen CSV hash differs: {filename}")
        with source.open(newline="") as handle:
            source_rows = list(csv.DictReader(handle))
        if len(source_rows) != expected["n_rows"]:
            raise ValueError(f"Source row count differs: {filename}")
        if len({row["utt_id"] for row in source_rows}) != len(source_rows):
            raise ValueError(f"Duplicate source IDs: {filename}")
        for source_row in source_rows:
            uid = source_row["utt_id"]
            row = by_id[uid]
            for source_key, metadata_key in [
                ("reference_text", "text"),
                ("primary_language", "primary_language"),
                ("gender", "gender"),
            ]:
                # Original inference scripts strip surrounding transcript
                # whitespace. Internal whitespace and all text must agree.
                if source_row[source_key].strip() != str(row[metadata_key]).strip():
                    raise ValueError(f"Pinned row mismatch: {filename}, {uid}, {source_key}")
            language = source_row["primary_language"].strip()
            family = source_row["family"]
            if family_by_language.setdefault(language, family) != family:
                raise ValueError(f"Inconsistent family assignment for {language}")
            delta = abs(float(source_row["duration_s"]) - float(row["duration"]))
            maximum_duration_difference_s = max(maximum_duration_difference_s, delta)
            if delta > 0.02:
                raise ValueError(f"Pinned audio duration mismatch: {filename}, {uid}, {delta}s")
            checked_ids.add(uid)
        source_checks.append({"file": filename, "sha256": actual_hash, "n_verified": len(source_rows)})

    if len(checked_ids) != 2794:
        raise ValueError(f"Expected 2794 unique previously evaluated IDs, got {len(checked_ids)}")
    original_checks = []
    for name, kind in [
        ("original_meta_speaker_stats.csv", "csv"),
        ("original_profile_meta_speaker_stats.csv", "csv"),
        ("original_svarah_manifest.jsonl", "jsonl"),
    ]:
        path = output_dir / name
        if not path.exists():
            continue
        if kind == "csv":
            with path.open(newline="") as handle:
                original_rows = list(csv.DictReader(handle))
        else:
            original_rows = read_jsonl(path)
        original_by_basename = {Path(row["audio_filepath"]).name: row for row in original_rows}
        if len(original_by_basename) != 6656:
            raise ValueError(f"Unexpected original artifact row identities: {name}")
        for row in raw:
            original = original_by_basename[Path(row["audio_path"]).name]
            for key in set(original) - {"audio_filepath", "duration"}:
                if str(row[key]) != str(original[key]):
                    raise ValueError(f"Original artifact metadata mismatch: {name}, {row['utt_id']}, {key}")
            if float(original["duration"]) != float(row["duration"]):
                raise ValueError(f"Original artifact duration mismatch: {name}, {row['utt_id']}")
        original_checks.append({
            "file": name, "sha256": sha256(path), "n_verified_by_basename": 6656,
            "fields": list(original_rows[0]), "speaker_id_present": "speaker_id" in original_rows[0],
        })

    profile_values = sorted({tuple(str(row[field]) for field in PROFILE_FIELDS) for row in raw})
    profile_ids = {value: f"profile_{index:03d}" for index, value in enumerate(profile_values)}
    rows = []
    metadata_by_tag = defaultdict(lambda: defaultdict(set))
    metadata_by_prefix = defaultdict(lambda: defaultdict(set))
    prefixes_by_tag = defaultdict(set)
    tags_by_prefix = defaultdict(set)
    for row in raw:
        basename = Path(row["audio_path"]).name
        match = TAG_RE.search(basename)
        if match is None:
            raise ValueError(f"Audio path does not match the expected chunk-tag convention: {row['utt_id']}")
        tag = match.group(1)
        prefix = basename.split("_")[0]
        if not prefix.isdigit():
            raise ValueError(f"Non-numeric source filename prefix: {row['utt_id']}")
        prefixes_by_tag[tag].add(prefix)
        tags_by_prefix[prefix].add(tag)
        for field in PROFILE_FIELDS:
            metadata_by_tag[tag][field].add(str(row.get(field)))
            metadata_by_prefix[prefix][field].add(str(row.get(field)))
        language = row["primary_language"].strip()
        if language not in family_by_language:
            raise ValueError(f"Language has no previously declared family mapping: {language}")
        rows.append({
            "utt_id": row["utt_id"],
            "dataset_index": row["dataset_index"],
            "audio_basename": basename,
            "recording_cluster_id": tag,
            "coarse_recording_cluster_id": f"prefix:{prefix}",
            "demographic_profile_id": profile_ids[tuple(str(row[field]) for field in PROFILE_FIELDS)],
            "speaker_id": "",
            "primary_language": language,
            "family": family_by_language[language],
            "gender": row["gender"],
            "duration_s": row["duration"],
            "reference_text": row["text"],
            "existing_row_verified": row["utt_id"] in checked_ids,
        })
    inconsistency_counts = {
        field: sum(len(values[field]) > 1 for values in metadata_by_tag.values())
        for field in PROFILE_FIELDS
    }
    if any(inconsistency_counts.values()):
        raise ValueError(f"Filename clusters have conflicting demographic metadata: {inconsistency_counts}")
    prefix_inconsistency_counts = {
        field: sum(len(values[field]) > 1 for values in metadata_by_prefix.values())
        for field in PROFILE_FIELDS
    }
    if any(prefix_inconsistency_counts.values()) or any(len(v) != 1 for v in prefixes_by_tag.values()):
        raise ValueError("Numeric-prefix clusters do not consistently nest the f-tags")
    cohort_ids = {
        "saaras450": set(manifest["saaras_cohort_ids"]),
        "open2500": set(manifest["open_cohort_ids"]),
        "union2794": checked_ids,
        "all6656": set(by_id),
    }
    cohort_rows = {name: [row for row in rows if row["utt_id"] in ids] for name, ids in cohort_ids.items()}
    cohort_summaries = {name: summary(items) for name, items in cohort_rows.items()}
    group_rows = []
    for cohort, items in cohort_rows.items():
        for dimension in ["family", "primary_language", "gender"]:
            groups = defaultdict(list)
            for row in items:
                groups[row[dimension]].append(row)
            for group, group_items in sorted(groups.items()):
                group_rows.append({"cohort": cohort, "dimension": dimension, "group": group, **summary(group_items)})

    report = {
        "dataset_revision": DATASET_REVISION,
        "raw_metadata_sha256": sha256(raw_path),
        "n_unique_existing_ids_verified": len(checked_ids),
        "identity_checks": ["dataset row index", "reference text after boundary whitespace strip", "primary language", "gender", "duration within 0.02 seconds"],
        "maximum_source_duration_difference_seconds": maximum_duration_difference_s,
        "source_checks": source_checks,
        "original_artifact_checks": original_checks,
        "public_speaker_id_available": False,
        "recording_tag_regex": TAG_RE.pattern,
        "recording_tag_interpretation": "Filename-derived recording tags; not speaker identifiers. Their semantics follow the chunk filename convention and remain an inference, not an author-provided recording-to-speaker mapping.",
        "recording_tag_demographic_inconsistency_counts": inconsistency_counts,
        "numeric_prefix_demographic_inconsistency_counts": prefix_inconsistency_counts,
        "f_tags_per_numeric_prefix_distribution": dict(sorted(Counter(len(v) for v in tags_by_prefix.values()).items())),
        "numeric_prefix_interpretation": "A coarser filename grouping: every f-tag nests in one numeric prefix. No author-provided semantics or speaker mapping was recovered.",
        "demographic_profile_fields": PROFILE_FIELDS,
        "demographic_profile_interpretation": "Exact equality of eight categorical metadata fields. These 115 profiles are NOT verified speakers; distinct speakers may share a profile, and inconsistent metadata could split a speaker. Use only as a labeled exploratory blocking sensitivity.",
        "demographic_profile_missing_value_counts": {
            field: sum(row[field] is None or str(row[field]).strip().lower() in {"", "nan", "none", "null", "na", "n/a"} for row in raw)
            for field in PROFILE_FIELDS
        },
        "family_mapping": family_by_language,
        "cohorts": cohort_summaries,
        "limitations": [
            "The paper reports 117 speakers but the current Hub schema exposes no speaker identifiers.",
            "A speaker may contribute several filename-derived recording clusters; cluster resampling does not resolve all within-speaker dependence.",
            "Filename-tag semantics have not been independently verified against original recording manifests or audio continuity.",
            "The Sino-Tibetan group contains only Bodo; it cannot identify a general family-level effect.",
        ],
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    for filename, items in [("svarah_metadata.csv", rows), ("cohort_metadata_counts.csv", group_rows)]:
        with (output_dir / filename).open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(items[0]))
            writer.writeheader()
            writer.writerows(items)
    (output_dir / "metadata_validation.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, default=ROOT / "paper/data/svarah_metadata_raw.jsonl")
    parser.add_argument("--manifest", type=Path, default=ROOT / "paper/data/input_manifest.json")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "paper/data")
    args = parser.parse_args()
    result = validate(args.raw, args.manifest, args.output_dir)
    print(json.dumps({"n_unique_existing_ids_verified": result["n_unique_existing_ids_verified"], "cohorts": result["cohorts"]}, indent=2))
