"""Descriptive paired analysis of the complete recovered Svarah shard.

Use only after independent source/run acceptance. No speaker inference, no
confidence intervals based on treating recordings as independent speakers.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path

from research.bridge_open.accept_svarah_runs import validate

ROOT = Path(__file__).resolve().parents[2]
ART = ROOT / "research/bridge_open/artifacts"
BUNDLE = ROOT / "data/bridge_open/svarah_native_shard02_full_v1/bundle"


def load(path): return json.loads(path.read_text())
def lines(path): return [json.loads(s) for s in path.read_text().splitlines()]
def errors(row): return sum(row["metrics"][k] for k in ("substitutions", "deletions", "insertions"))


def aggregate(rows):
    words = sum(r["metrics"]["ref_words"] for r in rows)
    chars = sum(r["metrics"]["ref_chars"] for r in rows)
    e = sum(errors(r) for r in rows)
    ce = sum(r["metrics"]["character_errors"] for r in rows)
    return dict(n=len(rows), word_errors=e, reference_words=words,
                wer=e/words if words else None, char_errors=ce, reference_chars=chars,
                cer=ce/chars if chars else None, empty_outputs=sum(r["status"]=="empty_output" for r in rows),
                wer_ge_1=sum(errors(r)>=r["metrics"]["ref_words"] for r in rows))


def analyze():
    models = {"Qwen3-ASR-1.7B": ART/"qwen_svarah_full_v1", "OmniASR-CTC-1B": ART/"omni_svarah_full_v1"}
    accepted = {k:validate(v, BUNDLE, 518) for k,v in models.items()}
    preds = {k:{r["conversation_id"]:r for r in lines(v/"predictions.jsonl")} for k,v in models.items()}
    manifest = {r["conversation_id"]:r for r in lines(BUNDLE/"manifest.jsonl")}
    source_audit = {r["conversation_id"]:r for r in lines(ART/"svarah_input_audit_shard02_full_v1/all_records.jsonl")}
    assert set(source_audit)==set(manifest)
    assert sum(r['clipping'] for r in source_audit.values())==16
    assert sum(r['exposed_priorly'] for r in source_audit.values())==209
    overall = {k:aggregate(list(v.values())) for k,v in preds.items()}
    paired=[]
    for uid,src in manifest.items():
        q,o=[preds[k][uid] for k in models]
        assert q["metrics"]["ref_words"]==o["metrics"]["ref_words"]
        paired.append(dict(conversation_id=uid,reference_words=q["metrics"]["ref_words"],
            qwen_errors=errors(q),omni_errors=errors(o),omni_minus_qwen_errors=errors(o)-errors(q),
            duration_seconds=src["duration_seconds"],source_row=src["source_row"],
            reference_path=src["reference_path"],audio_sha256=src["audio_sha256"],reference_sha256=src["reference_sha256"]))
    counts=dict(qwen_fewer=sum(r["qwen_errors"]<r["omni_errors"] for r in paired),
                equal=sum(r["qwen_errors"]==r["omni_errors"] for r in paired),
                omni_fewer=sum(r["omni_errors"]<r["qwen_errors"] for r in paired))
    report=dict(scope="All 518 available shard02 recordings of 6656 Svarah; availability-selected and preexposed exploratory",
        manifest_sha256=hashlib.sha256((BUNDLE/"manifest.jsonl").read_bytes()).hexdigest(),models=overall,
        paired_recording_counts=counts,input_qc="structural, not human adjudication",speaker_mapping="unavailable",
        uncertainty="No speaker-level interval or p-value; dependence mapping unavailable",
        costs="Unverified provider billing; reservations and worker runtimes are not measured cost",
        acceptance=accepted)
    out=ART/"svarah_comparison_v1";out.mkdir(exist_ok=True)
    (out/"analysis.json").write_text(json.dumps(report,indent=2)+"\n")
    (out/"paired_counts.jsonl").write_text("".join(json.dumps(r)+"\n" for r in paired))
    with (out/"largest_disagreements.csv").open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(paired[0]));w.writeheader();w.writerows(sorted(paired,key=lambda r:(-abs(r["omni_minus_qwen_errors"]),r["conversation_id"])))
    table="\n".join(f"| {k} | {v['n']} | {100*v['wer']:.2f}% | {100*v['cer']:.2f}% | {v['empty_outputs']} | {v['wer_ge_1']} |" for k,v in overall.items())
    text=f"""# Svarah recovered-shard comparison

**Scope: 518 source-verified recordings from one recovered shard, not the complete 6,656-record corpus.** The other two cached shards failed transfer. All available valid recordings are included, including 16 clipping-flagged recordings. This availability-selected subset includes 209 previously exposed IDs. No speaker IDs were inferred. The 12-item acceptance cohort overlaps this table and is not additional evidence.

| Declared native system | Scored recordings | Micro WER | Micro CER | Empty output | Recordings with WER ≥ 100% |
|---|---:|---:|---:|---:|---:|
{table}

On paired recordings, Qwen has fewer word edits in {counts['qwen_fewer']}, the counts are equal in {counts['equal']}, and Omni has fewer in {counts['omni_fewer']}. These are descriptive finite-cohort comparisons. WER ≥ 100% counts word edits; it is not a human-verified semantic failure category.

## Protocol and acceptance

The source WAV bytes are unchanged mono 16 kHz PCM16. Every reference/audio pair was independently compared to its original Arrow row and the pinned 6,656-row metadata. Every recording is decoded in full (0.1993125–27.8063125 seconds). No decoder, model revision or weights changed after the 12-item runtime check. Native BF16 Qwen uses explicit English, greedy batch 8; native BF16 Omni CTC uses batch 1 and verifies public-API/retained-token agreement on every output. Results compare these declared systems, not architecture in isolation.

Both runs completed all 518 predictions, passed source/model/code hash checks and independent rescoring, and retained tokens/termination evidence. Use [analysis.json](analysis.json) for exact counts, accepted job URLs and immutable source receipts; [paired counts](paired_counts.jsonl) for each source recording. The [ranked disagreements](largest_disagreements.csv) are a review aid, not completed human judgments. Worker runtime and reservations are not verified provider billing.

The frozen primary scorer applies Unicode NFC/casefold and punctuation-to-space normalization, preserving marks and making no numeral conversion. No post-hoc text normalization replaces this primary result.

## Limits and decision

This is an exploratory Indian-English extension of the existing system audit. It neither estimates all Svarah speakers nor isolates accent effects. Public-data training overlap is unknown. There are no speaker-level confidence intervals or p-values because reliable speaker mapping is unavailable. Reference/source checks are structural; fluent human listening is pending. This table does not establish a new recognition method, fairness, phone latency or publication novelty.

Next: examine paired disagreements with the audio, distinguish reference/normalization issues from recognition errors, and define any repair on development data before using an untouched confirmation set. Do not choose a repair and validate it on this same inspected subset.
"""
    (out/"RESULTS.md").write_text(text)
    print(json.dumps({k:v for k,v in report.items() if k!="acceptance"},indent=2))
    return report


if __name__=="__main__": analyze()
