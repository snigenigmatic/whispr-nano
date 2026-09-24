# English resource-panel listening review

This is a separate English extension for the fixed 12-item Svarah resource
panel. It is not part of the frozen 50-item primary listening packet.

The twelve recordings come from a pre-exposed exploratory corpus and are the
same recordings used for accepted batch-one hardware resource profiles. They
add no independent observations. Each item displays the source audio and its
source reference alongside six anonymous transcript cards. Those cards are
the final (`pass_index: 3`) outputs of six accepted batch-one profiles:
Whisper Small, Whisper Turbo, Whisper Large-v3, Qwen3-ASR 0.6B, Qwen3-ASR
1.7B, and OmniASR CTC 1B. The model-to-card correspondence is only in the
separate sealed mapping.

All 12 items are currently `UNREVIEWED`. Blank ratings are deliberate pending
state; no listening judgments were invented or transferred from any other
artifact.

## Reviewer delivery

Serve the repository root, because the viewer's audio URLs are rooted at the
repository's `data/` directory:

```sh
cd /Users/prateekraj/whispr-nano
.venv/bin/python -m http.server 8000
```

Open [the English viewer](http://localhost:8000/research/bridge_open/listening_review_core_en_v1/index.html).
Give a blinded reviewer only the viewer, `review_items.json`, and access to
the referenced audio. Do not share `sealed_mapping.json`, profile artifacts,
metric summaries, or model names. The viewer retains blank and incomplete
responses in its JSON download.

The primary `core_study/review_import.py` is intentionally not used for this
extension: it enforces the frozen 50-item Hindi/Bengali/Assamese packet and
would reject this distinct 12-item English panel. Any future English importer
must preserve this packet's 12-item/72-card scope and the pre-exposure note.

Use this packet's separate importer for an actual viewer JSON download. It
checks all 12 IDs and 72 anonymous cards, keeps blank and `uncertain` answers
as pending state, rejects changed or unknown ratings, and writes a create-only
receipt without opening the sealed mapping:

```sh
.venv/bin/python research/bridge_open/listening_review_core_en_v1/review_import.py \
  --input listening_review_core_en_responses.json \
  --reviewer reviewer_english_01 \
  --output-dir research/bridge_open/listening_review_core_en_v1/import_receipts
```

The resource results used here were already accepted batch-one measurements,
and these final-pass transcripts are matched to their own batch-one timing
condition. Do not pair primary batch-eight quality with batch-one latency as a
matched comparison. This pre-exposed panel is also not a general-English
population estimate. No reviewer has been contacted and this document makes no
schedule commitment.
