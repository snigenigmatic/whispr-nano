"""Word edit alignments with an optional zero-cost script-equivalence relation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Collection

import jiwer


@dataclass(frozen=True)
class Edit:
    ref_idx: int | None
    hyp_idx: int | None
    op: str


def align_words(
    reference: list[str],
    hypothesis: list[str],
    equivalences: Collection[tuple[str, str]] | None = None,
) -> list[Edit]:
    """Return an operation for each aligned token position.

    Raw alignment is expanded from jiwer/rapidfuzz, which guarantees identical
    edit counts. Condition C uses deterministic DP with SUB, DEL, INS tie order.
    """
    if not equivalences:
        output = jiwer.process_words(" ".join(reference), " ".join(hypothesis))
        edits: list[Edit] = []
        for chunk in output.alignments[0]:
            op = {"equal": "EQ", "substitute": "SUB", "delete": "DEL", "insert": "INS"}[chunk.type]
            count = max(chunk.ref_end_idx - chunk.ref_start_idx, chunk.hyp_end_idx - chunk.hyp_start_idx)
            for offset in range(count):
                ri = chunk.ref_start_idx + offset if op != "INS" else None
                hi = chunk.hyp_start_idx + offset if op != "DEL" else None
                edits.append(Edit(ri, hi, op))
        return edits

    pairs = set(equivalences)
    n, m = len(reference), len(hypothesis)
    dist = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dist[i][0] = i
    for j in range(m + 1):
        dist[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            a, b = reference[i - 1], hypothesis[j - 1]
            same = a == b or (a, b) in pairs or (b, a) in pairs
            dist[i][j] = min(dist[i - 1][j - 1] + (not same), dist[i - 1][j] + 1, dist[i][j - 1] + 1)
    out: list[Edit] = []
    i, j = n, m
    while i or j:
        if i and j:
            a, b = reference[i - 1], hypothesis[j - 1]
            same = a == b or (a, b) in pairs or (b, a) in pairs
            if dist[i][j] == dist[i - 1][j - 1] + (not same):
                out.append(Edit(i - 1, j - 1, "EQ" if same else "SUB"))
                i -= 1
                j -= 1
                continue
        if i and dist[i][j] == dist[i - 1][j] + 1:
            out.append(Edit(i - 1, None, "DEL"))
            i -= 1
            continue
        out.append(Edit(None, j - 1, "INS"))
        j -= 1
    return list(reversed(out))


def edit_counts(edits: list[Edit]) -> dict[str, int]:
    return {op: sum(e.op == op for e in edits) for op in ("EQ", "SUB", "DEL", "INS")}
