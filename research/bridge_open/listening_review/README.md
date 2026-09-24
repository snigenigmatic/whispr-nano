# Blinded listening and reference-review packet

Status: **UNREVIEWED**. This packet contains no human linguistic judgments. It was built from the frozen FLEURS manifest, corrected Whisper Turbo/Large-v3 predictions, and the accepted Qwen Hindi predictions.

The packet has 50 deduplicated utterances:

- Assamese: 16 (10 deterministic stratified-random + 6 targeted)
- Bengali: 16 (10 random + 6 targeted)
- Hindi: 18 (10 random + 8 targeted)
- Targeted source queue: 21 source entries representing 20 distinct utterances before deduplication

The random sample uses seed `20260923`, with 10 recordings per language. Targeted cases come from `artifacts/benchmark_review/human_review_queue.json`. Every selected audio and reference file was SHA-256 checked against `data/bridge_open/fleurs_test_v1/corpus-v2.jsonl`; every available prediction was checked against the same audio/reference hashes.

`review_items.json` and `index.html` provide the blinded reviewer view. The page contains audio controls, the reference, anonymous transcript cards, a rubric, editable responses, and JSON/CSV download buttons. Labels are deterministically randomized independently for each utterance. True system names and prediction metadata are only in `sealed_mapping.json`; keep that file away from the reviewer. `review_responses.csv` is intentionally blank and every item is marked `UNREVIEWED`.

To serve audio locally from the repository root:

```bash
python3 -m http.server 8768 --directory /Users/prateekraj/whispr-nano
```

Open `http://127.0.0.1:8768/research/bridge_open/listening_review/index.html`. The page loads only relative local files; no sealed file is loaded by browser JavaScript. Downloaded responses preserve `review_id` and `system_label`.

For each utterance, review:

1. **Reference correctness:** Does the reference match the spoken content? Mark `yes`, `no`, or `uncertain`; explain suspected omissions, insertions, substitutions, or segmentation problems.
2. **Language/script correctness:** Is the spoken language the declared language? Is the reference/transcript in an appropriate script? Record uncertainty rather than correcting text from memory.
3. **Meaning preservation:** For each anonymous transcript, does it preserve the utterance’s meaning? Use `yes`, `partly`, `no`, or `uncertain`. Do not score orthographic preferences as semantic errors unless they change intelligibility or meaning.
4. **Audio quality:** Note clipping, silence, overlapping speech, severe noise, truncation, or unintelligibility.
5. **Notes:** Record only observations grounded in the audio and displayed reference/transcript.

The packet is a qualitative audit sample, not a replacement for the full automatic metrics and not evidence of population-level human agreement.
