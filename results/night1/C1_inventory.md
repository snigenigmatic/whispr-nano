# Night 1 term-level ASR pilot

## C1 Code inventory

- [OBSERVED] The current `whispr-nano` checkout does not contain an `ast_asr` package. The reference file `src/ast_asr/metrics.py` was read from `snigenigmatic/ast-asr` branch `fair-cispo-work` (SHA-256 `3efb5f238719ca813e77d7b14fbfb22d4b0f225c59309da3507b8d99d32298f6`). The local `src/research_reset/scoring.py` reproduces that project's former capstone normalizer on a committed Svarah CSV, but its local implementation strips Devanagari entirely.
- [OBSERVED] `normalize_for_wer(text: str) -> str` in the reference package applies NFKC, Unicode lowercase, replaces characters outside `\w`, whitespace, and apostrophe with spaces, and collapses whitespace. It returns text, not alignments. Unicode combining marks in Devanagari words are stripped by this regex, splitting or damaging words. It is unsuitable for this pilot.
- [OBSERVED] `word_edit_counts(reference: str, hypothesis: str) -> EditCounts` in the reference package normalizes both strings and runs word-level Levenshtein. It returns substitution, deletion, insertion, and reference-word counts plus WER; it does not return alignments.
- [OBSERVED] This checkout's `modal_app.py` accepts a budget for its `smoke` and `poc` entrypoints, checks estimates with `estimate_cost` and `enforce_budget` from `src/onpolicy_distill/config.py`, then sends configuration dictionaries to `run_poc.remote`. It returns a run summary, not word alignments. The cost helper takes GPU type, step count, seconds per step, and a maximum dollar cost; it returns a cost estimate and raises if that estimate exceeds the cap. Its training-step estimate must be adapted for ASR inference clips.
- [OBSERVED] `audit/svarah_fairness_audit/run_saaras_audit.py::transcribe_cached(wav_path, api_key, model)` sends one WAV to Saaras V3, retries selected HTTP failures, caches the JSON response, and returns `(transcript, was_cached)`. It returns no alignments. Its existing mode is `transcribe` with `en-IN`, so the pilot needs a separate `codemix` call path.
- [OBSERVED] `audit/svarah_fairness_audit/assemble.py::assemble(directory, strict)` takes scored Svarah result CSVs, validates the exact schema, pools word-error counts, and returns per-system summary objects. It consumes counts from existing rows and does not produce alignments.

[OBSERVED] Exact outputs of the reference `normalize_for_wer` on the requested strings:

| Input | Output |
|---|---|
| `अब हम Python का code run करेंगे।` | `अब हम python क code run कर ग` |
| `फ़ाइल को save कीजिए` | `फ इल क save क ज ए` |
| `LibreOffice Calc में cell A1` | `libreoffice calc म cell a1` |
| `१२३ और 123` | `१२३ और 123` |
| `e-mail‌ भेजें` (ZWNJ after `mail`) | `e mail भ ज` |

- [ASSUMED] This pilot will preserve the requested new-code layout `ast_asr/termeval/` and `scripts/termeval/` in this checkout, while leaving existing Phase 3 paths unchanged.
