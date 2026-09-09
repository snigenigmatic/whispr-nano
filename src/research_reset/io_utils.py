"""CSV/JSON helpers for scored Svarah rows."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Iterable


def read_scored_csv(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="", encoding="utf-8") as f:
        rows = []
        for raw in csv.DictReader(f):
            row = dict(raw)
            row["wer"] = float(row["wer"])
            row["duration_s"] = float(row["duration_s"])
            row["n_sub"] = int(row["n_sub"])
            row["n_del"] = int(row["n_del"])
            row["n_ins"] = int(row["n_ins"])
            row["ref_len_words"] = int(row["ref_len_words"])
            row["accepted"] = row["wer"] == 0.0
            row["teacher_text"] = row["hypothesis_text"]
            rows.append(row)
        return rows


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: Iterable[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
