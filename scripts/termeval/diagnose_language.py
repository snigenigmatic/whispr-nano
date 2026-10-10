"""Compare the forced-Hindi diagnostic to matched auto-language outputs."""

from __future__ import annotations

import json
from pathlib import Path

from ast_asr.termeval.metrics import pooled_metrics, score_utterance
from ast_asr.termeval.normalize import normalize
from ast_asr.termeval.qc import has_repetition_loop
from ast_asr.termeval.tokens import token_class


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/night1"


def _load(path: Path) -> dict[str, dict]:
    return {r["utt_id"]: r for r in map(json.loads, path.open())}


def _score(ids: set[str], predictions: dict[str, dict], manifest: dict[str, dict]) -> dict:
    rows = [score_utterance(u, manifest[u]["speaker"], manifest[u]["reference"], predictions[u]["text"])
            for u in sorted(ids)]
    metrics = pooled_metrics(rows)
    return {name: {"value": value, "errors": num, "denominator": den}
            for name, (value, num, den) in metrics.items() if name in {"WER", "CER", "TokER[LAT]", "TokER[DEV]"}}


def main() -> None:
    forced = _load(OUT / "predictions_hiacc_forced_hi_diagnostic.jsonl")
    auto_file = (OUT / "predictions_hiacc.jsonl" if (OUT / "predictions_hiacc.jsonl").exists()
                 else OUT / "predictions_hiacc_auto_partial.jsonl")
    auto = _load(auto_file) if auto_file.exists() else {}
    manifest = _load(ROOT / "data/termeval/manifest.jsonl")
    ids = set(forced) & set(auto)
    summary = {"forced_hi_app_id": "ap-OMGEIKntgPKsA2aHsgSked",
               "forced_hi_rows": len(forced), "auto_rows_available": len(auto),
               "matched_rows": len(ids), "not_full_benchmark": True,
               "forced_hi_reference_latin_only": sum(
                   bool(tokens := normalize(manifest[u]["reference"]).split())
                   and all(token_class(t) == "LAT" for t in tokens) for u in forced),
               "forced_hi_output_longer_than_3x_ref": sum(
                   len(normalize(p["text"]).split()) > 3 * len(normalize(manifest[u]["reference"]).split())
                   for u, p in forced.items()),
               "forced_hi_outputs_with_repetition_loop_ge4": sum(
                   has_repetition_loop(normalize(p["text"]).split())
                   for p in forced.values()),
               "forced_hi_metrics_all_cached": _score(set(forced), forced, manifest),
               "matched_forced_hi": _score(ids, forced, manifest) if ids else None,
               "matched_auto": _score(ids, auto, manifest) if ids else None}
    (OUT / "language_conditioning_qc.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
