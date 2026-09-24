"""Gate-only partitions of the already-scored 2,500 Svarah utterances.

The 120 listening clips are held out from training. Remaining IDs split
80/20 train/eval, stratified by primary_language. This split is for the
Days 1–3 label-source gate only.

SPIRE-SIES and L2-ARCTIC are not touched. Days 4–10 replace this split
with verified speaker identities on an Indian-English training corpus.
"""

from __future__ import annotations

from typing import Any, Sequence

from .constants import GATE_EVAL_FRACTION, SAMPLE_SEED


def _rng(seed: int):
    import numpy as np

    return np.random.default_rng(seed)


def split_gate_rows(
    rows: Sequence[dict[str, Any]],
    listening_ids: set[str],
    *,
    seed: int = SAMPLE_SEED,
    eval_fraction: float = GATE_EVAL_FRACTION,
) -> dict[str, list[dict[str, Any]]]:
    listening = [r for r in rows if r["utt_id"] in listening_ids]
    rest = [r for r in rows if r["utt_id"] not in listening_ids]
    if len(listening) != len(listening_ids):
        missing = listening_ids - {r["utt_id"] for r in listening}
        raise ValueError(f"listening ids not in scored CSV: {sorted(missing)[:8]}")

    by_lang: dict[str, list[dict[str, Any]]] = {}
    for row in rest:
        by_lang.setdefault(row["primary_language"], []).append(row)

    rng = _rng(seed)
    train: list[dict[str, Any]] = []
    eval_rows: list[dict[str, Any]] = []
    for lang in sorted(by_lang):
        pool = sorted(by_lang[lang], key=lambda r: r["utt_id"])
        n_eval = max(1, int(round(len(pool) * eval_fraction))) if pool else 0
        n_eval = min(n_eval, len(pool) - 1) if len(pool) > 1 else n_eval
        idx = rng.permutation(len(pool))
        eval_idx = set(int(i) for i in idx[:n_eval])
        for i, row in enumerate(pool):
            (eval_rows if i in eval_idx else train).append(row)

    return {
        "listening": sorted(listening, key=lambda r: r["utt_id"]),
        "train": sorted(train, key=lambda r: r["utt_id"]),
        "eval": sorted(eval_rows, key=lambda r: r["utt_id"]),
    }


def partition_manifest(parts: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    def ids(name: str) -> list[str]:
        return [r["utt_id"] for r in parts[name]]

    return {
        "seed": SAMPLE_SEED,
        "eval_fraction": GATE_EVAL_FRACTION,
        "n_listening": len(parts["listening"]),
        "n_train": len(parts["train"]),
        "n_eval": len(parts["eval"]),
        "listening_ids": ids("listening"),
        "train_ids": ids("train"),
        "eval_ids": ids("eval"),
        "note": (
            "Gate-only split of the frozen 2,500 Svarah audit IDs. "
            "Do not train on SPIRE-SIES or L2-ARCTIC. Do not retune on "
            "this eval after looking at it. Days 4–10 rebuild speaker-verified "
            "train/dev; this eval is not the paper test set."
        ),
    }
