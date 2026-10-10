"""Reference-anchored term and writing-system ASR error accounting."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
import random

import jiwer

from .align import Edit, align_words, edit_counts
from .normalize import normalize
from .tokens import token_class


@dataclass
class ScoredUtterance:
    utt_id: str
    speaker: str
    reference: str
    hypothesis: str
    edits: list[Edit]
    totals: Counter[str]
    term_rows: list[dict]
    token_rows: list[dict]


def _term_status(span_ops: list[str]) -> str:
    if all(op == "EQ" for op in span_ops):
        return "CORRECT"
    if all(op == "DEL" for op in span_ops):
        return "DELETED"
    if "SUB" in span_ops and "DEL" not in span_ops:
        return "SUBSTITUTED"
    return "PARTIAL"


def score_utterance(
    utt_id: str,
    speaker: str,
    reference: str,
    hypothesis: str,
    *,
    equivalences: set[tuple[str, str]] | None = None,
    spans: list[dict] | None = None,
) -> ScoredUtterance:
    ref_text, hyp_text = normalize(reference), normalize(hypothesis)
    ref, hyp = ref_text.split(), hyp_text.split()
    edits = align_words(ref, hyp, equivalences)
    counts = edit_counts(edits)
    totals: Counter[str] = Counter({f"word_{k}": v for k, v in counts.items()})
    totals["n_ref"] = len(ref)
    ref_ops = {e.ref_idx: e.op for e in edits if e.ref_idx is not None}
    by_class: dict[str, Counter[str]] = defaultdict(Counter)
    for i, token in enumerate(ref):
        klass = token_class(token)
        by_class[klass]["n"] += 1
        if ref_ops[i] in {"SUB", "DEL"}:
            by_class[klass]["wrong"] += 1
    for klass, klass_counts in by_class.items():
        totals[f"class_{klass}_n"] += klass_counts["n"]
        totals[f"class_{klass}_wrong"] += klass_counts["wrong"]

    # Under C, only zero-cost aligned pairs are mapped to a common spelling
    # before character scoring. This preserves raw CER for all other text.
    char_ref, char_hyp = ref.copy(), hyp.copy()
    if equivalences:
        for edit in edits:
            if edit.op == "EQ" and edit.ref_idx is not None and edit.hyp_idx is not None:
                a, b = ref[edit.ref_idx], hyp[edit.hyp_idx]
                if a != b:
                    char_ref[edit.ref_idx] = a
                    char_hyp[edit.hyp_idx] = a
    chars = jiwer.process_characters(" ".join(char_ref), " ".join(char_hyp))
    totals["char_sub"] = chars.substitutions
    totals["char_del"] = chars.deletions
    totals["char_ins"] = chars.insertions
    totals["n_chars"] = len(" ".join(char_ref))

    span_list = spans or []
    term_rows: list[dict] = []
    indexed_spans = [(idx, s) for idx, s in enumerate(span_list) if s.get("label") == "TERM"]
    in_term: dict[int, list[int]] = defaultdict(list)
    for idx, span in indexed_spans:
        start, end = int(span["start"]), int(span["end"])
        if not (0 <= start < end <= len(ref)):
            raise ValueError(f"invalid TERM span for {utt_id}: {span}")
        for i in range(start, end):
            in_term[i].append(idx)
        ops = [ref_ops[i] for i in range(start, end)]
        status = _term_status(ops)
        # A boundary insertion sits between a term and its neighbouring token.
        # Count it separately without changing the reference-anchored outcome.
        boundary_ins = 0
        for pos, edit in enumerate(edits):
            if edit.op != "INS":
                continue
            prev_ref = next((e.ref_idx for e in reversed(edits[:pos]) if e.ref_idx is not None), None)
            next_ref = next((e.ref_idx for e in edits[pos + 1 :] if e.ref_idx is not None), None)
            if (prev_ref == start - 1 and next_ref == start) or (
                prev_ref == end - 1 and next_ref == end
            ):
                boundary_ins += 1
        term_rows.append(
            {"utt_id": utt_id, "speaker": speaker, "term_id": idx, "start": start, "end": end,
             "status": status, "boundary_ins": boundary_ins, "text": " ".join(ref[start:end])}
        )
        totals["terms_n"] += 1
        totals[f"terms_{status.lower()}"] += 1
        totals["term_boundary_ins"] += boundary_ins

    token_rows = []
    for edit in edits:
        token_rows.append(
            {"utt_id": utt_id, "speaker": speaker, "ref_idx": edit.ref_idx,
             "ref_tok": ref[edit.ref_idx] if edit.ref_idx is not None else None,
             "hyp_tok": hyp[edit.hyp_idx] if edit.hyp_idx is not None else None,
             "op": edit.op,
             "tok_class": token_class(ref[edit.ref_idx]) if edit.ref_idx is not None else None,
             "in_term": bool(in_term.get(edit.ref_idx)),
             "term_id": ",".join(str(x) for x in in_term.get(edit.ref_idx, [])) or None}
        )
    return ScoredUtterance(utt_id, speaker, ref_text, hyp_text, edits, totals, term_rows, token_rows)


def pooled_metrics(rows: Iterable[ScoredUtterance]) -> dict[str, tuple[float | None, int, int]]:
    """Return metric -> (rate, numerator, denominator), pooling raw counts."""
    totals: Counter[str] = Counter()
    for row in rows:
        totals.update(row.totals)
    return _metrics_from_totals(totals)


def _metrics_from_totals(totals: Counter[str]) -> dict[str, tuple[float | None, int, int]]:
    def rate(name: str, num: int, den: int) -> tuple[str, tuple[float | None, int, int]]:
        return name, (num / den if den else None, num, den)

    values = dict(
        [rate("WER", sum(totals[f"word_{x}"] for x in ("SUB", "DEL", "INS")), totals["n_ref"]),
         rate("CER", sum(totals[f"char_{x}"] for x in ("sub", "del", "ins")), totals["n_chars"]),
         rate("INS_rate", totals["word_INS"], totals["n_ref"]),
         rate("TermER", totals["terms_n"] - totals["terms_correct"], totals["terms_n"]) ]
    )
    for klass in ("LAT", "DEV", "NUM", "MIX", "OTHER"):
        values.update([rate(f"TokER[{klass}]", totals[f"class_{klass}_wrong"], totals[f"class_{klass}_n"])])
    return values


def cluster_bootstrap(
    rows: list[ScoredUtterance], *, n_resamples: int = 1000, seed: int = 20261011
) -> dict[str, tuple[float | None, float | None]]:
    """95% percentile intervals from resampling speakers/recordings."""
    by_cluster: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        by_cluster[row.speaker].update(row.totals)
    keys = sorted(by_cluster)
    if not keys:
        return {}
    rng = random.Random(seed)
    samples: dict[str, list[float]] = defaultdict(list)
    for _ in range(n_resamples):
        selected: Counter[str] = Counter()
        for key in rng.choices(keys, k=len(keys)):
            selected.update(by_cluster[key])
        for metric, (value, _, _) in _metrics_from_totals(selected).items():
            if value is not None:
                samples[metric].append(value)
    intervals = {}
    for metric, values in samples.items():
        values.sort()
        intervals[metric] = (values[int(0.025 * (len(values) - 1))], values[int(0.975 * (len(values) - 1))])
    return intervals
