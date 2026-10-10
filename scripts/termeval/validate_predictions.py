"""Inspect every decoded utterance for gross hallucination and cache integrity."""

from __future__ import annotations

from collections import Counter, defaultdict
import csv
import json
from pathlib import Path

from ast_asr.termeval.align import align_words, edit_counts
from ast_asr.termeval.normalize import normalize


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/night1"


def main() -> None:
    manifest = {r["utt_id"]: r for r in map(json.loads, (ROOT / "data/termeval/manifest.jsonl").open())}
    flagged = []
    report = {}
    for dataset in ("hiacc", "mucs_hien"):
        file = OUT / f"predictions_{dataset}.jsonl"
        rows = [json.loads(line) for line in file.open()]
        expected = {u for u, row in manifest.items() if row["dataset"] == dataset}
        if len(rows) != len(expected) or {r["utt_id"] for r in rows} != expected:
            raise ValueError(f"{dataset}: missing/duplicate predictions")
        if len({r["config_hash"] for r in rows}) != 1:
            raise ValueError(f"{dataset}: mixed decode configurations")
        reasons: Counter[str] = Counter()
        for pred in rows:
            utt = pred["utt_id"]
            ref = normalize(manifest[utt]["reference"]).split()
            hyp = normalize(pred["text"]).split()
            counts = edit_counts(align_words(ref, hyp))
            wer = (counts["SUB"] + counts["DEL"] + counts["INS"]) / len(ref)
            issues = []
            if not hyp:
                issues.append("empty_hypothesis")
            if len(hyp) > 3 * len(ref):
                issues.append("hyp_gt_3x_ref_tokens")
            if any(n >= 4 for n in Counter(hyp).values()):
                issues.append("token_repeated_ge_4")
            if wer >= 2:
                issues.append("utterance_wer_ge_200pct")
            if issues:
                reasons.update(issues)
                flagged.append({"dataset": dataset, "utt_id": utt,
                                "speaker": manifest[utt]["speaker"],
                                "duration_s": round(manifest[utt]["duration_s"], 3),
                                "reference_tokens": len(ref), "hypothesis_tokens": len(hyp),
                                "wer": round(wer, 4), "flags": ";".join(issues)})
        report[dataset] = {"predictions": len(rows), "config_hash": rows[0]["config_hash"],
                           "flag_counts": dict(reasons),
                           "flagged_utterances": sum(r["dataset"] == dataset for r in flagged)}
    with (OUT / "prediction_qc_flags.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["dataset", "utt_id", "speaker", "duration_s",
                                                       "reference_tokens", "hypothesis_tokens", "wer", "flags"])
        writer.writeheader()
        writer.writerows(flagged)
    (OUT / "prediction_qc.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report))


if __name__ == "__main__":
    main()
