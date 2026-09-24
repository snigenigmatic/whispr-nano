"""Create the deterministic, reference-free target teacher quality report."""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "research/publication/data"
PSEUDO = DATA / "spire_target_pseudo_large_v3_seed12.jsonl"
AUDIT = DATA / "spire_target_audit_120_blind.jsonl"
OUTPUT = DATA / "spire_target_teacher_quality.json"
ANOMALY_AUDIT = DATA / "spire_target_audit_anomalies_blind.jsonl"
EOS_ID = 50257


def quantiles(values: list[float], points: tuple[float, ...]) -> dict[str, float]:
    ordered = sorted(values)
    return {
        f"p{int(point * 100):02d}": ordered[round((len(ordered) - 1) * point)]
        for point in points
    }


def repetition(row: dict) -> dict:
    words = re.findall(r"[a-z0-9']+", row["pseudo_text"].lower())
    trigrams = list(zip(words, words[1:], words[2:]))
    repeat_fraction = 1 - len(set(trigrams)) / len(trigrams) if trigrams else 0.0
    longest_run = 0
    current_run = 0
    previous = None
    for word in words:
        current_run = current_run + 1 if word == previous else 1
        longest_run = max(longest_run, current_run)
        previous = word
    return {
        "words": len(words), "trigram_repeat_fraction": repeat_fraction,
        "longest_identical_word_run": longest_run,
        "flagged": len(words) >= 20 and (repeat_fraction >= 0.35 or longest_run >= 5),
    }


def main() -> None:
    rows = [json.loads(line) for line in PSEUDO.read_text().splitlines() if line]
    audit = [json.loads(line) for line in AUDIT.read_text().splitlines() if line]
    if len(rows) != 4800 or len(audit) != 120:
        raise ValueError("teacher artifact counts changed")
    repeated = {row["utt_id"]: repetition(row) for row in rows}
    flagged = [row for row in rows if repeated[row["utt_id"]]["flagged"]]
    token_limit = [row for row in rows if row["teacher_generated_token_ids"][-1] != EOS_ID]
    anomaly_ids = {
        row["utt_id"] for row in flagged
    } | {row["utt_id"] for row in token_limit}
    anomaly_fields = (
        "utt_id", "speaker_id", "native_language", "accent", "gender",
        "audio_path", "audio_sha256", "duration_s", "teacher_mean_logprob",
    )
    anomaly_rows = []
    for row in sorted((row for row in rows if row["utt_id"] in anomaly_ids), key=lambda row: row["utt_id"]):
        anomaly_rows.append({
            **{key: row[key] for key in anomaly_fields},
            "repetition_flag": repeated[row["utt_id"]]["flagged"],
            "token_limit_flag": row["teacher_generated_token_ids"][-1] != EOS_ID,
        })
    ANOMALY_AUDIT.write_text("".join(
        json.dumps(row, sort_keys=True) + "\n" for row in anomaly_rows
    ))
    accents = {}
    for accent in sorted({row["accent"] for row in rows}):
        group = [row for row in rows if row["accent"] == accent]
        accents[accent] = {
            "rows": len(group),
            "mean_logprob": quantiles(
                [row["teacher_mean_logprob"] for row in group],
                (0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0),
            ),
            "word_count": quantiles(
                [repeated[row["utt_id"]]["words"] for row in group],
                (0.0, 0.5, 0.9, 0.99, 1.0),
            ),
            "repetition_flags": sum(repeated[row["utt_id"]]["flagged"] for row in group),
            "token_limit_rows": sum(row["teacher_generated_token_ids"][-1] != EOS_ID for row in group),
        }
    common = Counter(row["pseudo_text"] for row in rows)
    report = {
        "status": "completed", "analysis_scope": "reference_free_teacher_diagnostics",
        "pseudo_manifest_sha256": hashlib.sha256(PSEUDO.read_bytes()).hexdigest(),
        "audit_manifest_sha256": hashlib.sha256(AUDIT.read_bytes()).hexdigest(),
        "rows": len(rows), "unique_utterance_ids": len({row["utt_id"] for row in rows}),
        "unique_prediction_texts": len(common),
        "duplicate_prediction_text_instances": sum(count - 1 for count in common.values()),
        "token_limit_rows": len(token_limit),
        "token_limit_utterance_ids": sorted(row["utt_id"] for row in token_limit),
        "repetition_rule": "at least 20 words and trigram-repeat fraction >=0.35 or identical-word run >=5",
        "repetition_flagged_rows": len(flagged),
        "repetition_flagged_utterance_ids": sorted(row["utt_id"] for row in flagged),
        "blind_anomaly_audit_rows": len(anomaly_rows),
        "blind_anomaly_audit_sha256": hashlib.sha256(ANOMALY_AUDIT.read_bytes()).hexdigest(),
        "audit_contains_token_limit_row": bool(
            {row["utt_id"] for row in audit} & {row["utt_id"] for row in token_limit}
        ),
        "by_accent": accents,
        "interpretation": (
            "These diagnostics detect truncation and mechanical repetition without target references. "
            "They do not establish transcript correctness; the frozen 120-clip human audit remains required."
        ),
    }
    OUTPUT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: report[key] for key in (
        "rows", "unique_prediction_texts", "token_limit_rows", "repetition_flagged_rows"
    )}, sort_keys=True))


if __name__ == "__main__":
    main()
