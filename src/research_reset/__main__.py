"""CLI: build the Days 1–3 artifacts from the committed large-v3 CSV."""

from __future__ import annotations

import argparse
from pathlib import Path

from .constants import (
    AUDIT_CSV,
    DATA_DIR,
    L1_GROUPS,
    SAMPLE_SEED,
    TEACHER_SYSTEM,
)
from .io_utils import read_scored_csv, write_csv, write_json
from .listening import REVIEWER_INSTRUCTIONS, listening_summary, sample_listening_set
from .partitions import partition_manifest, split_gate_rows
from .runs import build_run_specs, design_document


def build_artifacts(csv_path: Path = AUDIT_CSV, out_dir: Path = DATA_DIR) -> dict:
    rows = read_scored_csv(csv_path)
    listening = sample_listening_set(rows)
    summary = listening_summary(listening)
    parts = split_gate_rows(rows, set(summary["utt_ids"]))
    specs = build_run_specs(parts["train"])
    design = design_document(specs)

    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(out_dir / "listening_set.json", listening)
    write_json(out_dir / "listening_summary.json", summary)
    write_csv(
        out_dir / "listening_set.csv",
        listening,
        [
            "utt_id",
            "svarah_index",
            "primary_language",
            "family",
            "gender",
            "duration_s",
            "accepted",
            "cell",
            "disagreement_kind",
            "capstone_wer",
            "reference_text",
            "teacher_text",
        ],
    )
    blind = [
        {
            "utt_id": i["utt_id"],
            "svarah_index": i["svarah_index"],
            "primary_language": i["primary_language"],
            "family": i["family"],
            "gender": i["gender"],
            "duration_s": i["duration_s"],
            "clip_index": n,
        }
        for n, i in enumerate(listening, start=1)
    ]
    write_json(out_dir / "listening_set.blind.json", blind)
    # Reviewer worksheets omit both transcripts and the accepted/rejected tag.
    blank = [
        {
            "utt_id": i["utt_id"],
            "clip_index": n,
            "primary_language": i["primary_language"],
            "family": i["family"],
            "gender": i["gender"],
            "duration_s": f"{i['duration_s']:.2f}",
            "transcript": "",
            "notes": "",
        }
        for n, i in enumerate(listening, start=1)
    ]
    fields = [
        "clip_index",
        "utt_id",
        "primary_language",
        "family",
        "gender",
        "duration_s",
        "transcript",
        "notes",
    ]
    write_csv(out_dir / "reviewer_a_worksheet.csv", blank, fields)
    write_csv(out_dir / "reviewer_b_worksheet.csv", blank, fields)
    write_json(out_dir / "frozen_eval_ids.json", sorted(r["utt_id"] for r in rows))
    write_json(out_dir / "gate_split.json", partition_manifest(parts))
    write_json(out_dir / "run_specs.json", specs)
    write_json(out_dir / "design.json", design)
    write_json(
        out_dir / "l1_groups.json",
        {"groups": list(L1_GROUPS), "n": 6, "seed": SAMPLE_SEED, "teacher": TEACHER_SYSTEM},
    )
    (out_dir / "REVIEWER_INSTRUCTIONS.txt").write_text(REVIEWER_INSTRUCTIONS, encoding="utf-8")
    return {
        "n_scored": len(rows),
        "n_listening": len(listening),
        "n_train": len(parts["train"]),
        "n_eval": len(parts["eval"]),
        "n_runs": len(specs),
        "out_dir": str(out_dir),
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=AUDIT_CSV)
    parser.add_argument("--out", type=Path, default=DATA_DIR)
    args = parser.parse_args(argv)
    info = build_artifacts(args.csv, args.out)
    print(info)


if __name__ == "__main__":
    main()
