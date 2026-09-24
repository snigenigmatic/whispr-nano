"""Deterministic Unicode script distribution audit; no inference or scoring."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import unicodedata
from pathlib import Path


BLOCKS = {
    "Arabic": ((0x0600, 0x06FF), (0x0750, 0x077F), (0x08A0, 0x08FF), (0xFB50, 0xFDFF), (0xFE70, 0xFEFF)),
    "Devanagari": ((0x0900, 0x097F), (0xA8E0, 0xA8FF), (0x11B00, 0x11B5F)),
    "Bengali": ((0x0980, 0x09FF),), "Gurmukhi": ((0x0A00, 0x0A7F),),
    "Gujarati": ((0x0A80, 0x0AFF),), "Oriya": ((0x0B00, 0x0B7F),),
    "Tamil": ((0x0B80, 0x0BFF),), "Telugu": ((0x0C00, 0x0C7F),),
    "Kannada": ((0x0C80, 0x0CFF),), "Malayalam": ((0x0D00, 0x0D7F),),
    "Latin": ((0x0041, 0x005A), (0x0061, 0x007A), (0x00C0, 0x02AF), (0x1E00, 0x1EFF)),
}


def script_for(char: str) -> str | None:
    category = unicodedata.category(char)
    if not (category.startswith("L") or category.startswith("M")):
        return None
    codepoint = ord(char)
    for script, ranges in BLOCKS.items():
        if any(start <= codepoint <= end for start, end in ranges):
            return script
    name = unicodedata.name(char, "")
    for script in BLOCKS:
        if script.upper() in name:
            return script
    return None


def distribution(text: str) -> dict:
    counts: dict[str, int] = {}
    assigned = 0
    letter_mark_count = 0
    for char in text:
        if unicodedata.category(char).startswith(("L", "M")):
            letter_mark_count += 1
        script = script_for(char)
        if script is not None:
            counts[script] = counts.get(script, 0) + 1
            assigned += 1
    dominant = sorted(counts, key=lambda key: (-counts[key], key))[0] if counts else None
    return {"counts": dict(sorted(counts.items())), "assigned_count": assigned,
            "letter_mark_count": letter_mark_count,
            "unassigned_letter_mark_count": letter_mark_count - assigned,
            "assigned_fraction_of_letters_marks": assigned / letter_mark_count if letter_mark_count else None,
            "dominant_script": dominant, "dominant_count": counts.get(dominant, 0) if dominant else 0,
            "dominant_scope": "tracked Indic/Arabic/Latin scripts",
            "text_length_codepoints": len(text)}


def _sha_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _sha_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit(run_dirs: list[str | Path], output_dir: str | Path) -> dict:
    output = Path(output_dir); output.mkdir(parents=True, exist_ok=True)
    manifest_cache: dict[str, dict] = {}
    rows, runs = [], []
    for raw in run_dirs:
        directory = Path(raw); cfg = json.loads((directory / "config.json").read_text())
        manifest_path = Path(cfg["manifest_path"])
        if cfg.get("manifest_sha256") and _sha_file(manifest_path) != cfg["manifest_sha256"]:
            raise ValueError(f"manifest hash mismatch: {manifest_path}")
        manifest = manifest_cache.setdefault(str(manifest_path), {json.loads(line)["conversation_id"]: json.loads(line) for line in manifest_path.read_text().splitlines()})
        predictions = [json.loads(line) for line in (directory / "predictions.jsonl").read_text().splitlines()]
        runs.append(directory.name)
        for prediction in predictions:
            cid = prediction["conversation_id"]
            item = manifest[cid]
            reference_path, audio_path = Path(item["reference_path"]), Path(item["audio_path"])
            if _sha_file(reference_path) != item.get("reference_sha256"):
                raise ValueError(f"reference hash mismatch: {cid}")
            if _sha_file(audio_path) != item.get("audio_sha256"):
                raise ValueError(f"audio hash mismatch: {cid}")
            if prediction.get("reference_sha256") not in (None, item.get("reference_sha256")):
                raise ValueError(f"prediction/reference hash mismatch: {cid}")
            if prediction.get("audio_sha256") not in (None, item.get("audio_sha256")):
                raise ValueError(f"prediction/audio hash mismatch: {cid}")
            reference = json.loads(reference_path.read_text())["reference_text"]
            hypothesis = prediction.get("text_raw", "")
            ref_dist, hyp_dist = distribution(reference), distribution(hypothesis)
            ref_dom, hyp_dom = ref_dist["dominant_script"], hyp_dist["dominant_script"]
            rows.append({"run": directory.name, "model_id": prediction.get("model_id", cfg.get("model_id")),
                         "model_revision": prediction.get("model_revision", cfg.get("model_revision")),
                         "condition_id": prediction.get("condition_id", cfg.get("condition_id")),
                         "conversation_id": cid, "language": prediction.get("language", item.get("language")),
                         "status": prediction.get("status"), "reference_sha256": prediction.get("reference_sha256", item.get("reference_sha256")),
                         "audio_sha256": prediction.get("audio_sha256", item.get("audio_sha256")),
                         "reference_text_sha256": _sha_text(reference), "hypothesis_text_sha256": _sha_text(hypothesis),
                         "reference_distribution": ref_dist, "hypothesis_distribution": hyp_dist,
                         "dominant_script_mismatch": bool(ref_dom and hyp_dom and ref_dom != hyp_dom)})
    json_rows = []
    for row in rows:
        json_rows.append(row)
    report = {"contract": "unicode_script_audit_v1",
              "unicodedata_version": unicodedata.unidata_version,
              "scope": "script diagnostic only; not language identification, semantic correctness, or a score",
              "counting_rule": "Unicode letters and combining marks assigned to the named Indic, Arabic, or Latin ranges; digits, punctuation, symbols, whitespace, and unassigned scripts excluded",
              "runs": runs, "conversation_count": len(rows), "rows": json_rows}
    (output / "script_audit.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    fields = ["run", "model_id", "model_revision", "condition_id", "conversation_id", "language", "status", "reference_sha256", "audio_sha256", "reference_text_sha256", "hypothesis_text_sha256", "reference_distribution", "hypothesis_distribution", "dominant_script_mismatch"]
    with (output / "script_audit.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader()
        for row in rows:
            writer.writerow({**row, "reference_distribution": json.dumps(row["reference_distribution"], ensure_ascii=False, sort_keys=True), "hypothesis_distribution": json.dumps(row["hypothesis_distribution"], ensure_ascii=False, sort_keys=True)})
    return report


def default_runs(root: Path) -> list[Path]:
    names = list(json.loads((root / "expansion_runs.json").read_text()))
    names += ["hindi-turbo-cuda-beam5-v1", "hindi-qwen3-asr-1-7b-v1"]
    return [root / name for name in dict.fromkeys(names) if (root / name / "predictions.jsonl").exists()]


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifacts", default="research/bridge_open/artifacts")
    parser.add_argument("--output", default="research/bridge_open/artifacts/expanded_diagnostics")
    args = parser.parse_args(argv)
    report = audit(default_runs(Path(args.artifacts)), args.output)
    print(json.dumps({"runs": report["runs"], "conversation_rows": report["conversation_count"], "output": args.output}))


if __name__ == "__main__":
    main()
