#!/usr/bin/env python3
"""Acquire L2-ARCTIC metadata by bounded ZIP ranges and freeze leakage-safe splits.

No model inference and no bulk audio download. Run `python data_setup.py` to
fetch metadata, or `python data_setup.py --offline` to rebuild from cached ranges.
The mirror's LFS hashes identify whole archives; range reads do not verify those
whole-file hashes. Verify them when downloading archives for training.
"""
from __future__ import annotations

import argparse
import collections
import concurrent.futures
import hashlib
import io
import json
import re
import struct
import urllib.request
import zipfile
from pathlib import Path

DATA = Path(__file__).resolve().parent / "data"
REPO = "chikingsley/l2-arctic-release-v5.0"
REVISION = "cc02fd37197966e5de11ad737f7affd577c61992"
BASE = f"https://huggingface.co/datasets/{REPO}/resolve/{REVISION}"
SEED = 20260909
OFFICIAL_HOME = "https://psi.engr.tamu.edu/l2-arctic-corpus/"
OFFICIAL_DOCS = "https://psi.engr.tamu.edu/l2-arctic-corpus-docs/"
# L1/gender mapping is from the corpus owners' homepage. "Mandarin" is its
# homepage label; README.original.md labels those same four speakers "Chinese".
SPEAKERS = {
    "Arabic": {"ABA": "M", "SKA": "F", "YBAA": "M", "ZHAA": "F"},
    "Mandarin": {"BWC": "M", "LXC": "F", "NCC": "F", "TXHC": "M"},
    "Hindi": {"ASI": "M", "RRBI": "M", "SVBI": "F", "TNI": "F"},
    "Korean": {"HJK": "F", "HKK": "M", "YDCK": "F", "YKWK": "M"},
    "Spanish": {"EBVS": "M", "ERMS": "M", "MBMPS": "F", "NJS": "F"},
    "Vietnamese": {"HQTV": "M", "PNV": "F", "THV": "F", "TLV": "M"},
}
COUNTS = {"ABA":1129,"SKA":974,"YBAA":1130,"ZHAA":1132,"BWC":1130,
          "LXC":1131,"NCC":1131,"TXHC":1132,"ASI":1131,"RRBI":1130,
          "SVBI":1132,"TNI":1131,"HJK":1131,"HKK":1131,"YDCK":1131,
          "YKWK":1131,"EBVS":1007,"ERMS":1132,"MBMPS":1132,"NJS":1131,
          "HQTV":1132,"PNV":1132,"THV":1132,"TLV":1132}


def sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def normalize_minimal(text: str) -> str:
    """Same minimal normalizer as paper/analysis/analyze.py."""
    return " ".join(re.sub(r"[^\w\s']", " ", text.lower()).split())


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def cached_url(url: str, path: Path, offline: bool, max_bytes: int = 2_000_000) -> bytes:
    if path.exists():
        return path.read_bytes()
    if offline:
        raise FileNotFoundError(f"Offline source missing: {path}")
    with urllib.request.urlopen(url, timeout=90) as response:
        body = response.read(max_bytes + 1)
    if len(body) > max_bytes:
        raise ValueError(f"Metadata download exceeds cap: {url}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(body)
    return body


class RangeFile(io.RawIOBase):
    """Small, cached HTTP ranges; refuses a server silently sending whole ZIP."""
    def __init__(self, url: str, size: int, cache: Path, offline: bool):
        self.url, self.size, self.cache, self.offline = url, size, cache, offline
        self.position = 0
        self.ranges: list[tuple[int, bytes]] = []
        self.cache.mkdir(parents=True, exist_ok=True)
        for path in sorted(self.cache.glob("*.bin")):
            start, end = map(int, path.stem.split("-"))
            body = path.read_bytes()
            if len(body) != end - start + 1:
                raise ValueError(f"Truncated cached range: {path}")
            self.ranges.append((start, body))

    def seekable(self):
        return True

    def tell(self):
        return self.position

    def seek(self, offset, whence=0):
        self.position = offset if whence == 0 else self.position + offset if whence == 1 else self.size + offset
        if not 0 <= self.position <= self.size:
            raise ValueError("ZIP seek outside archive")
        return self.position

    def fetch(self, start: int, end: int) -> bytes:
        if end < start:
            return b""
        for old_start, body in self.ranges:
            if old_start <= start and end < old_start + len(body):
                return body[start - old_start:end - old_start + 1]
        if self.offline:
            raise FileNotFoundError(f"Offline range missing: {self.cache.name} {start}-{end}")
        if end - start + 1 > 8_000_000:
            raise ValueError("Refusing an unexpectedly large metadata range")
        request = urllib.request.Request(self.url, headers={"Range": f"bytes={start}-{end}"})
        with urllib.request.urlopen(request, timeout=90) as response:
            if response.status != 206:
                raise ValueError(f"Expected HTTP206, got {response.status}")
            if response.headers.get("Content-Range") != f"bytes {start}-{end}/{self.size}":
                raise ValueError(f"Unexpected Content-Range: {response.headers.get('Content-Range')}")
            body = response.read(end - start + 2)
        if len(body) != end - start + 1:
            raise ValueError("Incomplete or excessive HTTP range")
        (self.cache / f"{start}-{end}.bin").write_bytes(body)
        self.ranges.append((start, body))
        return body

    def read(self, size=-1):
        size = self.size - self.position if size < 0 else min(size, self.size - self.position)
        result = self.fetch(self.position, self.position + size - 1)
        self.position += len(result)
        return result


def acquire_speaker(speaker: dict, offline: bool) -> tuple[list[dict], dict]:
    code = speaker["speaker_id"]
    source = RangeFile(speaker["archive_url"], speaker["archive_bytes"], DATA / "range_cache" / code, offline)
    with zipfile.ZipFile(source) as archive:
        infos = archive.infolist()
        wavs = sorted((info for info in infos if "/wav/" in info.filename and info.filename.endswith(".wav")), key=lambda info: info.filename)
        if len(wavs) != COUNTS[code]:
            raise ValueError(f"Official count mismatch for {code}: {len(wavs)}")
        # These archives place all metadata before the audio. One contiguous
        # range covers annotations, forced alignments and orthographic text.
        first_audio = min(info.header_offset for info in wavs)
        metadata = [info for info in infos if not info.is_dir() and "/wav/" not in info.filename]
        if any(info.header_offset + info.compress_size + len(info.filename.encode()) + 30 > first_audio for info in metadata):
            raise ValueError(f"Unexpected ZIP layout for {code}; stop rather than download audio")
        source.fetch(0, first_audio - 1)
        names = set(archive.namelist())
        rows = []
        max_duration_difference = 0.0
        for info in wavs:
            prompt_id = Path(info.filename).stem
            transcript_path = f"{code}/transcript/{prompt_id}.txt"
            grid_path = f"{code}/textgrid/{prompt_id}.TextGrid"
            text_bytes = archive.read(transcript_path)  # ZipFile checks each CRC.
            text = text_bytes.decode("utf-8-sig").strip()
            grid_bytes = archive.read(grid_path)
            match = re.search(rb"^xmax\s*=\s*([0-9.eE+\-]+)", grid_bytes, re.M)
            if not text or not match:
                raise ValueError(f"Missing reference or alignment duration: {code}/{prompt_id}")
            duration = float(match.group(1))
            # WAV size-derived duration is a diagnostic. Do not assert that an
            # uninspected RIFF header is always44 bytes or silently use a proxy
            # to authorize equal-duration training. Actual audio must be checked.
            size_duration = (info.file_size - 44) / (44100 * 2)
            max_duration_difference = max(max_duration_difference, abs(duration - size_duration))
            rows.append({
                "utt_id": f"{code}_{prompt_id}", "speaker_id": code,
                "native_language": speaker["native_language"], "gender": speaker["gender"],
                "prompt_id": prompt_id, "reference_text": text,
                "audio_path": info.filename, "duration_s": duration,
                "duration_source": "forced_alignment_xmax_verify_against_decoded_audio",
                "speaker_split": speaker["split"],
                "manual_annotation": f"{code}/annotation/{prompt_id}.TextGrid" in names,
                "transcript_sha256": sha(text_bytes), "textgrid_sha256": sha(grid_bytes),
                "audio_uncompressed_bytes": info.file_size, "audio_crc32": f"{info.CRC:08x}",
                "archive_name": f"{code}.zip", "source_revision": REVISION,
            })
    hashes = {path.name: sha(path.read_bytes()) for path in sorted(source.cache.glob("*.bin"))}
    return rows, {"speaker_id": code, "rows": len(rows), "manual_annotations": sum(row["manual_annotation"] for row in rows),
                  "metadata_range_bytes_cached": sum(path.stat().st_size for path in source.cache.glob("*.bin")),
                  "range_sha256": hashes, "whole_archive_sha256_verified": False,
                  "max_textgrid_vs_44_byte_pcm16_duration_difference_s": max_duration_difference}


def assign_content(rows: list[dict]) -> tuple[dict[str, str], dict]:
    # Union by prompt identity OR exact minimally normalized reference text.
    # This holds reference variants and repeated prompt strings in one family.
    parent = {row["prompt_id"]: row["prompt_id"] for row in rows}
    def find(key):
        while parent[key] != key:
            parent[key] = parent[parent[key]]
            key = parent[key]
        return key
    def union(left, right):
        a, b = sorted((find(left), find(right)))
        parent[b] = a
    seen = {}
    for row in rows:
        key = normalize_minimal(row["reference_text"])
        if not key:
            raise ValueError("Empty normalized reference")
        if key in seen:
            union(seen[key], row["prompt_id"])
        seen[key] = row["prompt_id"]
    families = collections.defaultdict(list)
    for prompt in parent:
        families[find(prompt)].append(prompt)
    families = [sorted(prompts) for prompts in families.values()]
    families.sort(key=lambda prompts: sha(f"{SEED}:content:{'|'.join(prompts)}".encode()))
    n_train = int(0.70 * len(families))
    n_dev = int(0.15 * len(families))
    assigned, report = {}, []
    for index, prompts in enumerate(families):
        split = "train" if index < n_train else "dev" if index < n_train + n_dev else "test"
        for prompt in prompts:
            assigned[prompt] = split
        report.append({"family_id": sha("|".join(prompts).encode()), "prompt_ids": prompts, "split": split})
    return assigned, {"families": report, "family_count": len(families), "prompt_count": len(parent),
                      "family_split_counts": dict(collections.Counter(row["split"] for row in report))}


def validate_primary(rows: list[dict]) -> dict:
    if not rows or {row["split"] for row in rows} != {"train", "dev", "test"}:
        raise ValueError("Primary manifest must contain nonempty train/dev/test and no other splits")
    report = {}
    for field in ("speaker_id", "prompt_id", "normalized_reference"):
        sets = {split: {normalize_minimal(row["reference_text"]) if field == "normalized_reference" else row[field]
                        for row in rows if row["split"] == split} for split in ("train", "dev", "test")}
        overlap = {f"{left}:{right}": sorted(sets[left] & sets[right])
                   for left, right in (("train", "dev"), ("train", "test"), ("dev", "test"))}
        if any(overlap.values()):
            raise ValueError(f"Primary {field} leakage: {overlap}")
        report[field] = overlap
    if len({row["utt_id"] for row in rows}) != len(rows):
        raise ValueError("Duplicate utterance IDs")
    by_split = {}
    for split in ("train", "dev", "test"):
        subset = [row for row in rows if row["split"] == split]
        by_split[split] = {"utterances": len(subset), "speakers": sorted({row["speaker_id"] for row in subset}),
                           "hours_from_alignment": sum(row["duration_s"] for row in subset) / 3600,
                           "by_native_language": dict(sorted(collections.Counter(row["native_language"] for row in subset).items()))}
    return {"leakage_checks": report, "splits": by_split}


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows))


def validate_audio_headers(data_dir: Path = DATA) -> dict:
    """Check downloaded train/dev WAV metadata without changing frozen manifests."""
    speakers = json.loads((data_dir / "speakers.json").read_text())
    rows = [json.loads(line) for line in (data_dir / "corpus_metadata.jsonl").read_text().splitlines()]
    results = []
    for speaker in speakers:
        if speaker["split"] == "test":
            continue
        path = data_dir / "audio_validation" / (speaker["speaker_id"] + ".json")
        if not path.exists():
            raise FileNotFoundError(f"Audio metadata not yet available: {path}")
        checked = json.loads(path.read_text())
        if checked["archive_sha256"] != speaker["archive_sha256"] or checked["archive_bytes"] != speaker["archive_bytes"]:
            raise ValueError(f"Audio archive identity mismatch: {path}")
        subset = [row for row in rows if row["speaker_id"] == speaker["speaker_id"]]
        audio = checked["audio_metadata"]
        if set(audio) != {row["prompt_id"] for row in subset}:
            raise ValueError(f"Audio/transcript inventory mismatch: {path}")
        differences = []
        for row in subset:
            metadata = audio[row["prompt_id"]]
            if metadata["sample_rate"] != 44100 or metadata["channels"] != 1:
                raise ValueError(f"Unexpected audio format for {row['utt_id']}")
            difference = abs(metadata["duration_s"] - row["duration_s"])
            if difference > 0.01:
                raise ValueError(f"Audio/alignment duration differs by over10ms: {row['utt_id']}")
            differences.append(difference)
        results.append({"speaker_id": speaker["speaker_id"], "rows": len(subset),
                        "max_abs_duration_difference_s": max(differences), "validation_file_sha256": sha(path.read_bytes())})
    report = {"all_train_dev_archives_verified": True, "test_audio_checked": False,
              "manifest_unchanged": True, "speakers": results,
              "note": "Use per-prompt decoded duration from audio_validation for exact duration matching; frozen alignment durations differ at most10ms."}
    write_json(data_dir / "audio_header_checks.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--check-audio", action="store_true", help="Check local train/dev decoded-WAV reports, without refetching or changing splits")
    args = parser.parse_args()
    if args.check_audio:
        print(json.dumps(validate_audio_headers(), indent=2))
        return
    DATA.mkdir(parents=True, exist_ok=True)
    src = DATA / "sources"
    src.mkdir(exist_ok=True)
    source_urls = {"README.original.md": f"{BASE}/README.original.md", "README.mirror.md": f"{BASE}/README.md",
                   "PROMPTS": f"{BASE}/PROMPTS", "LICENSE": f"{BASE}/LICENSE"}
    for name, url in source_urls.items():
        cached_url(url, src / name, args.offline)
    tree_path = DATA / "hub_tree.json"
    tree = json.loads(cached_url(f"https://huggingface.co/api/datasets/{REPO}/tree/{REVISION}?recursive=false", tree_path, args.offline))
    speakers = []
    for language, people in SPEAKERS.items():
        ranked = sorted(people, key=lambda code: sha(f"{SEED}:speaker:{code}".encode()))
        for code, split in zip(ranked, ("train", "train", "dev", "test")):
            entry = next(item for item in tree if item["path"] == f"{code}.zip")
            speakers.append({"speaker_id": code, "native_language": language, "gender": people[code], "split": split,
                             "archive_url": f"{BASE}/{code}.zip", "archive_sha256": entry["lfs"]["oid"], "archive_bytes": entry["size"]})
    write_json(DATA / "speakers.json", speakers)
    rows, acquisitions = [], []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(acquire_speaker, speaker, args.offline): speaker["speaker_id"] for speaker in speakers}
        for future in concurrent.futures.as_completed(futures):
            subset, acquisition = future.result()
            rows.extend(subset)
            acquisitions.append(acquisition)
            print(f"Metadata {acquisition['speaker_id']}: {len(subset)} rows", flush=True)
    rows.sort(key=lambda row: row["utt_id"])
    if len(rows) != 26867 or sum(row["manual_annotation"] for row in rows) != 3599:
        raise ValueError("Official corpus totals disagree")
    content, content_report = assign_content(rows)
    primary, shared, excluded = [], [], []
    for row in rows:
        row["content_split"] = content[row["prompt_id"]]
        if row["content_split"] == row["speaker_split"]:
            primary.append(dict(row, split=row["speaker_split"]))
        elif row["speaker_split"] in ("dev", "test") and row["content_split"] == "train":
            shared.append(dict(row, split=row["speaker_split"] + "_shared"))
        else:
            excluded.append(dict(row, split="excluded", exclusion_reason="speaker_and_content_partitions_differ"))
    validation = validate_primary(primary)
    write_jsonl(DATA / "corpus_metadata.jsonl", rows)
    write_jsonl(DATA / "manifest.jsonl", primary)
    write_jsonl(DATA / "shared_content_diagnostic.jsonl", shared)
    write_jsonl(DATA / "excluded.jsonl", excluded)
    write_json(DATA / "content_partitions.json", content_report)
    prompt_pairs = re.findall(r'^\(\s*(\S+)\s+"(.*)"\s*\)$', (src / "PROMPTS").read_text(), re.M)
    prompts = dict(prompt_pairs)
    differences = [row["utt_id"] for row in rows if normalize_minimal(row["reference_text"]) != normalize_minimal(prompts[row["prompt_id"]])]
    report = {"dataset": REPO, "revision": REVISION, "seed": SEED, "source_release": "L2-ARCTIC v5.0",
              "license": "CC-BY-NC-4.0", "scope": "scripted corpus; spontaneous suitcase subset not acquired",
              "official_authority_urls": [OFFICIAL_HOME, OFFICIAL_DOCS],
              "official_access_note": "Reviewed official pages through web browser text tool on2026-09-09; direct urllib HTTP403. Speaker mapping/counts checked against those pages.",
              "source_urls": source_urls, "source_sha256": {name: sha((src / name).read_bytes()) for name in source_urls},
              "mirror_identity_limit": "No official full-release archive downloaded; documented counts/layout verified. Whole archive SHA256 must be checked on compute host.",
              "transcript_limit": "Provided orthographic transcripts are references, not independently audio-adjudicated by this project. Manual pronunciation annotations cover3599 recordings only.",
              "duration_limit": "duration_s is forced-alignment xmax; decode WAV headers on compute host before duration-matched sampling.",
              "speaker_split_rule": "Within each official L1 sort SHA256('20260909:speaker:'+speaker_id); first2 train, third dev, fourth test.",
              "content_split_rule": "Union prompt IDs sharing exact minimal-normalized transcripts; sort SHA256('20260909:content:'+joined sorted prompt IDs); floor70% train, floor15% dev, remainder test.",
              "full_metadata_rows": len(rows), "manual_annotations": sum(row["manual_annotation"] for row in rows),
              "total_alignment_hours": sum(row["duration_s"] for row in rows) / 3600,
              "references_differing_from_canonical_prompt_after_minimal": len(differences),
              "reference_difference_ids": differences,
              "metadata_bytes_cached": sum(a["metadata_range_bytes_cached"] for a in acquisitions),
              "all_24_archive_bytes": sum(s["archive_bytes"] for s in speakers),
              "train_dev_archive_bytes": sum(s["archive_bytes"] for s in speakers if s["split"] != "test"),
              "speaker_acquisitions": sorted(acquisitions, key=lambda a: a["speaker_id"]),
              "validation": validation, "shared_diagnostic_counts": dict(collections.Counter(r["split"] for r in shared)),
              "excluded_rows": len(excluded), "final_test_inference_performed": False}
    outputs = ["speakers.json", "corpus_metadata.jsonl", "manifest.jsonl", "shared_content_diagnostic.jsonl", "excluded.jsonl", "content_partitions.json"]
    report["output_sha256"] = {name: sha((DATA / name).read_bytes()) for name in outputs}
    write_json(DATA / "provenance.json", report)
    print(json.dumps({"rows": len(rows), "primary": validation["splits"], "metadata_MB": report["metadata_bytes_cached"] / 1e6,
                      "manifest_sha256": report["output_sha256"]["manifest.jsonl"]}, indent=2))


if __name__ == "__main__":
    main()
