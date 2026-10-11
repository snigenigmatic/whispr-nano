"""Falsify the safety of a text-only loop rule on genuine-repetition references.

This is an oracle structural check, not ASR inference on the audio. The public
test dataset is used only for this held-out diagnostic, never for rule tuning.
"""

from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
import unicodedata

from huggingface_hub import hf_hub_download
import jiwer
import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from publication_study.loop_repair import suppress_adjacent_word_loops  # noqa: E402


DATASET = "Scicom-intl/Whisper-Hallucination"
REVISION = "8126be4a4b83afefeef9394ff1312b635a499d82"
FILE = "data/reduplication/test-00000-of-00001.parquet"
OUT = ROOT / "results/night1/lahaja_loop_repair_positive_control_20261011.json"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _norm(value: str) -> str:
    value = unicodedata.normalize("NFC", value).lower()
    return " ".join("".join(
        ch if unicodedata.category(ch)[0] in "LMN" else " " for ch in value
    ).split())


def main() -> None:
    path = Path(hf_hub_download(DATASET, FILE, repo_type="dataset", revision=REVISION))
    rows = pq.read_table(path, columns=["reference_text", "n_repeats", "pattern"]).to_pylist()
    if len(rows) != 1440 or Counter(r["n_repeats"] for r in rows) != {
        3: 240, 4: 240, 5: 240, 6: 240, 8: 240, 12: 240
    }:
        raise ValueError("public positive-control sample changed")
    by_repetition = Counter()
    changed_by_repetition = Counter()
    errors = words = removed_words = 0
    for row in rows:
        repeats = row["n_repeats"]
        reference = row["reference_text"]
        repaired, removed = suppress_adjacent_word_loops(reference)
        counts = jiwer.process_words(_norm(reference), _norm(repaired))
        by_repetition[repeats] += 1
        changed_by_repetition[repeats] += removed > 0
        errors += counts.substitutions + counts.deletions + counts.insertions
        words += len(_norm(reference).split())
        removed_words += removed
    result = {
        "status": "oracle_structural_counterexample",
        "dataset": DATASET,
        "dataset_revision": REVISION,
        "config": "reduplication/test",
        "source_parquet_sha256": _sha(path),
        "policy_sha256": _sha(ROOT / "src/publication_study/loop_repair.py"),
        "rows": len(rows),
        "perfect_transcripts_altered": sum(changed_by_repetition.values()),
        "by_intended_repetition_count": {
            str(n): {"rows": by_repetition[n], "perfect_transcripts_altered": changed_by_repetition[n]}
            for n in sorted(by_repetition)
        },
        "reference_words": words,
        "words_deleted_from_perfect_transcripts": removed_words,
        "oracle_wer_percent_if_repaired": 100 * errors / words,
        "interpretation": "Applying this text-only policy to perfect transcripts deletes genuinely repeated words. This is a structural safety counterexample, not an observed ASR false-positive rate.",
        "limit": "The dataset uses synthetic repeated speech; actual model predictions on its audio have not been evaluated here.",
        "script_sha256": _sha(Path(__file__)),
    }
    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"output": str(OUT),
                      "perfect_transcripts_altered": result["perfect_transcripts_altered"],
                      "oracle_wer_percent_if_repaired": result["oracle_wer_percent_if_repaired"]}))


if __name__ == "__main__":
    main()
