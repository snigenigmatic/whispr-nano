# Blinded listening and reference review v2

Status: **UNREVIEWED**. This is a new blinded packet using exactly the 50 utterance IDs and selection flags in `../listening_review` v1. It has three transcript sources for every utterance and one additional source for Hindi: 168 cards in total. Labels are independently randomized per utterance with seed `20260924`; true names and provenance remain in `sealed_mapping.json` and `packet_metadata.json`. Keep both files away from reviewers until annotations are frozen.

The packet contains Assamese 16, Bengali 16, and Hindi 18 utterances. The targeted queue is a fixed audit subset; it is not a prevalence sample, and manual reviewers must assess fluency and meaning from the audio, reference, and anonymous transcripts rather than infer population rates. Overlap between random and targeted selection flags is preserved.

Serve the repository root and open the v2 reviewer page:

```bash
python3 -m http.server 8768 --bind 127.0.0.1 --directory /Users/prateekraj/whispr-nano
```

Open `http://127.0.0.1:8768/research/bridge_open/listening_review_v2/index.html`. The page loads relative local assets, provides audio controls, references, anonymous transcript cards, per-transcript language, expected-script, and meaning judgments, guarded local persistence, and JSON/CSV downloads. All packet judgments start `UNREVIEWED`; partial input is `IN_PROGRESS`; `REVIEWED` requires every required field.

Reference review asks whether the reference matches the spoken audio. The packet is a qualitative audit sample and does not establish population-level prevalence or human agreement.
