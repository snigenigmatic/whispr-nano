# Dual-transcription protocol (120 clips)

Two reviewers. Independent. Blind to the dataset reference, the teacher hypothesis, and each other. This is Days 1–3 step 2 of the research reset.

## Why this exists

The label-source experiment treats Svarah’s `text` field as gold and whisper-large-v3’s greedy transcript as the teacher. If a large share of “teacher errors” are actually **reference errors** or **normalization differences**, the 12-run comparison is not about teacher mistakes. We listen before we train.

## Sample (frozen)

- Source: `audit/svarah_fairness_audit/results_whisper-large-v3_svarah_clean.csv` (2,500 scored Svarah utterances).
- Six L1 groups: Hindi, Bengali, Marathi, Tamil, Telugu, Bodo.
- Ten **accepted** (capstone WER = 0) and ten **rejected** (WER > 0) per L1 → **120**.
- RNG seed `20260909`. Rebuild with `PYTHONPATH=src python -m research_reset`.
- IDs: `data/research_reset/listening_set.json`. Reviewers never receive this file until both transcripts are in.
- Blind worksheet: `data/research_reset/reviewer_{a,b}_worksheet.csv` and `listening_set.blind.json`.

These 120 IDs are **held out of the gate train/eval split**. They are not training targets. If we repair a reference, the repair applies to this set and to any later training item with the same issue after a documented pass — not silently to the frozen SPIRE / L2-ARCTIC tests.

## Reviewer procedure

1. Pick a role: Reviewer A or Reviewer B. Do not switch mid-set.
2. Headphones. Quiet room. One clip at a time, in the worksheet order.
3. Play as often as needed. Type what was said.
4. Disfluencies that are clearly spoken stay in. Do not “fix” Indian-English grammar.
5. Unintelligible span: `[inaudible]`. Unsure word: `[uncertain: guess]`.
6. Numbers as heard (digits vs words). Do not expand abbreviations you did not hear.
7. No discussion of a clip until **both** reviewers have submitted that clip.
8. Do not open `listening_set.json`, the audit CSV, or the teacher/reference columns.

Full text: `data/research_reset/REVIEWER_INSTRUCTIONS.txt`. The listening lab at `listen/` enforces hiding of sealed transcripts.

## Reconciliation (together, after both finish)

For each clip, sit with both transcripts and the audio. Then reveal the reference and the teacher. Assign **one** primary label:

| Label | Meaning |
|---|---|
| `teacher_mistake` | Audio matches the reference, not the teacher |
| `reference_mistake` | Audio matches the teacher, not the reference |
| `both_wrong` | Neither matches what was said |
| `normalization` | Same words; punctuation, casing, numerals, spelling |
| `genuine_variation` | Both are plausible hearings |
| `other` | Write a note |

Secondary tags allowed: `proper_noun`, `code_mix`, `numeral`, `disfluency`, `hallucination`, `truncation`.

Write the agreed transcript in `data/research_reset/reconciliation.json` (see template below). If **reference_mistake + both_wrong** exceed 15% of the 60 rejected clips, **stop**: repair or drop bad references before any of the 12 runs. Normalization-only disagreements on the accepted cell are expected; they are not a repair trigger.

## Template

```json
{
  "reviewer_a": "name",
  "reviewer_b": "name",
  "protocol_version": "2026-09-09",
  "clips": [
    {
      "utt_id": "svarah_00000",
      "transcript_a": "",
      "transcript_b": "",
      "agreed_transcript": "",
      "label": "teacher_mistake",
      "secondary": [],
      "notes": ""
    }
  ]
}
```

## Audio

Svarah is gated. From a machine with `HF_TOKEN`:

```bash
PYTHONPATH=src python -m research_reset.fetch_audio --out listen/public/audio
```

Until wavs land, the lab still supports typing and export; the play button is disabled.
