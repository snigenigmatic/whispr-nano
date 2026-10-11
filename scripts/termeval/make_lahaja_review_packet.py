"""Export a private, two-reviewer Lahaja development error-audit packet.

This script intentionally uses only the first Parquet shard. It does not read
the held-out prediction set and does not commit dataset audio or transcripts.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
from pathlib import Path
import random

from huggingface_hub import hf_hub_download
import pyarrow.parquet as pq
import soundfile as sf


SEED = 20261011
DATASET = "ai4bharat/Lahaja"
REVISION = "d4ffd2ecbdd933e37c917ddcf620eef159ceb3a7"
SHARD = "data/test-00000-of-00004.parquet"
PREDICTIONS = Path.home() / ".local/share/openresearch/lahaja-turbo-104-v1/predictions.jsonl"
OUT = Path.home() / ".local/share/openresearch/lahaja-review-dev-v1"


def _token() -> str | None:
    import os

    if os.environ.get("HF_TOKEN"):
        return os.environ["HF_TOKEN"]
    path = Path.home() / ".openresearch/env"
    if path.exists():
        for raw in path.read_text().splitlines():
            line = raw.strip().removeprefix("export ")
            if line.startswith("HF_TOKEN="):
                return line.split("=", 1)[1].strip().strip('"').strip("'")
    return None


def _pick(rows: list[dict]) -> list[dict]:
    rng = random.Random(SEED)
    picked = []
    for scenario in ("Read", "Extempore"):
        group = [r for r in rows if r["scenario"] == scenario]
        high = sorted(group, key=lambda r: (-r["word_errors"] / r["ref_words"], r["row_index"]))
        selected = []
        speakers = set()
        for row in high:
            if row["sp_id"] not in speakers:
                selected.append(row)
                speakers.add(row["sp_id"])
            if len(selected) == 8:
                break
        rest = [r for r in group if r["row_index"] not in {x["row_index"] for x in selected}]
        rng.shuffle(rest)
        for row in rest:
            if sum(x["sp_id"] == row["sp_id"] for x in selected) < 2:
                selected.append(row)
            if len(selected) == 15:
                break
        if len(selected) != 15:
            raise ValueError(f"could not build 15-item {scenario} review sample")
        picked.extend(selected)
    rng.shuffle(picked)
    return picked


def _write_csv(path: Path, fields: list[str], rows: list[dict]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    predictions = [json.loads(line) for line in PREDICTIONS.read_text().splitlines()]
    if len(predictions) != 104 or len({r["sp_id"] for r in predictions}) != 26:
        raise ValueError("expected the frozen 104-clip development pilot")
    picked = _pick(predictions)
    path = Path(hf_hub_download(DATASET, SHARD, repo_type="dataset",
                                revision=REVISION, token=_token()))
    pf = pq.ParquetFile(path)
    selected = {r["row_index"] for r in picked}
    clips = {}
    offset = 0
    for rg in range(pf.metadata.num_row_groups):
        count = pf.metadata.row_group(rg).num_rows
        wanted = selected & set(range(offset, offset + count))
        if wanted:
            audio_rows = pf.read_row_group(rg, columns=["audio_filepath"]).to_pylist()
            for idx in wanted:
                raw = audio_rows[idx - offset]["audio_filepath"]["bytes"]
                audio, sr = sf.read(io.BytesIO(raw), always_2d=True, dtype="float32")
                if sr != 16000 or audio.shape[1] != 1:
                    raise ValueError(f"unexpected audio format at row {idx}")
                clips[idx] = audio[:, 0]
        offset += count
    if set(clips) != selected:
        raise ValueError("review audio incomplete")

    if any(OUT.glob("reviewer_*_*.csv")):
        raise FileExistsError(f"review sheets already exist; refusing to overwrite: {OUT}")
    OUT.mkdir(parents=True, exist_ok=True)
    audio_dir = OUT / "audio"
    audio_dir.mkdir(exist_ok=True)
    manifest = []
    for number, row in enumerate(picked, 1):
        item_id = f"L{number:03d}"
        filename = f"{item_id}.wav"
        sf.write(audio_dir / filename, clips[row["row_index"]], 16000, subtype="PCM_16")
        manifest.append({"item_id": item_id, "audio": f"audio/{filename}",
                         "row_index": row["row_index"], "speaker_id": row["sp_id"],
                         "scenario": row["scenario"], "native_state": row["native_state"],
                         "reference": row["reference"], "hypothesis": row["prediction"],
                         "development_wer": row["word_errors"] / row["ref_words"]})

    for reviewer in ("A", "B"):
        order = manifest.copy()
        random.Random(SEED + ord(reviewer)).shuffle(order)
        ref_rows = [{"item_id": r["item_id"], "audio": r["audio"],
                     "released_reference": r["reference"],
                     "reference_matches_audio_yes_no_uncertain": "", "reference_issue_notes": ""}
                    for r in order]
        asr_rows = [{"item_id": r["item_id"], "audio": r["audio"],
                     "released_reference": r["reference"], "asr_hypothesis": r["hypothesis"],
                     "numeric_value_correct_yes_no_na_uncertain": "",
                     "named_entity_correct_yes_no_na_uncertain": "",
                     "domain_term_correct_yes_no_na_uncertain": "",
                     "other_acoustic_error_yes_no_uncertain": "",
                     "hallucination_yes_no_uncertain": "", "notes": ""}
                    for r in order]
        _write_csv(OUT / f"reviewer_{reviewer}_reference.csv", list(ref_rows[0]), ref_rows)
        _write_csv(OUT / f"reviewer_{reviewer}_asr.csv", list(asr_rows[0]), asr_rows)

    private_manifest = OUT / "private_manifest.json"
    private_manifest.write_text(json.dumps({"dataset_revision": REVISION,
                                            "source_predictions_sha256": hashlib.sha256(
                                                PREDICTIONS.read_bytes()).hexdigest(),
                                            "selection": "8 highest-WER distinct-speaker plus 7 seeded controls per scenario; development shard only",
                                            "items": manifest}, ensure_ascii=False, indent=2) + "\n")
    (OUT / "README.md").write_text(
        "# Lahaja development listening packet\n\n"
        "This private folder contains 30 first-shard clips (15 Read, 15 Extempore). "
        "It contains released audio and transcripts; keep it within the authorized team.\n\n"
        "Two bilingual reviewers should work independently. In each reviewer's "
        "`reviewer_<A/B>_reference.csv`, listen before reading the ASR hypothesis "
        "and mark whether the released reference matches the speech. Then use "
        "that reviewer's `reviewer_<A/B>_asr.csv` to label numeric value, named "
        "entity, domain term, other acoustic "
        "error, and hallucination. Use `na` where the category is absent and "
        "`uncertain` rather than guessing. Resolve disagreements only after both "
        "sheets are complete. These are development examples; do not use held-out "
        "speaker predictions to choose or tune a correction.\n"
    )
    print(json.dumps({"packet": str(OUT), "items": len(manifest),
                      "read": sum(x["scenario"] == "Read" for x in manifest),
                      "extempore": sum(x["scenario"] == "Extempore" for x in manifest),
                      "speakers": len({x["speaker_id"] for x in manifest})}))


if __name__ == "__main__":
    main()
