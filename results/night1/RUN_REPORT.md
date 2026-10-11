# Night 1 term-level ASR pilot — evidence report

[OBSERVED] This is a public-corpus feasibility audit for C. Kaustubh's 16 October 2026 review. It is **not a submission-ready fairness or domain-term result**. All numbers below come from the linked local artifacts; the two restricted datasets remain unexamined.

## Status

| Step | Status | Evidence / reason |
|---|---|---|
| C1 code inventory | Done | [`C1_inventory.md`](C1_inventory.md) |
| C2 datasets | Partial | MUCS and HiACC verified; Lahaja and VITB-HEBiC returned HTTP 401 without accepted access. [`data_card.md`](data_card.md) |
| C3 normalization/scoring | Done for public data | Devanagari-safe normalizer, jiwer-matched raw alignment, pooled metrics and 1,000-resample cluster CIs (HiACC speakers; MUCS tutorials/recordings); `tests/termeval/`. |
| C4 ASR decode | Done for public data | Whisper large-v3 over every clip: HiACC auto-language and MUCS forced Hindi as frozen in [`decode_policy.json`](decode_policy.json). Full MUCS auto-language output retained as a diagnostic. Initial HiACC forced-Hindi run stopped after QC failure. Saaras skipped because vendor permission and a call cap were unavailable. |
| C5 script lexicon | Provisional | Corpus-derived candidates, explicit collision/low-support guards, no human approval. [`lexicon_v0.csv`](lexicon_v0.csv) |
| C6 TERM annotation | Partial | 200 cluster-balanced utterances receive reproducible dictionary candidate spans, **not LLM or human labels**. [`annotation_guideline.md`](annotation_guideline.md) |
| C7 group slices | Descriptive | HiACC adult/child and gender only where speaker support is adequate; no hypothesis tests. [`metrics.csv`](metrics.csv) |
| C8 plots/examples | Done with support limits | Vector plots in [`figs/`](figs/) and seed-selected candidate errors in [`examples_term_errors.md`](examples_term_errors.md). |

## Data card summary

[OBSERVED] MUCS Hindi-English test: 3,136 sentence segments from 30 long recordings and 30 source-provided `utt2spk` IDs; 5.178 hours by segment timestamps. Each ID maps one-to-one to a recording here. The [dataset authors](https://arxiv.org/html/2104.00235) state that speaker identity was unavailable and they assumed each tutorial had a different speaker; these are **recording/tutorial clusters, not verified distinct people**. The full archive SHA-256 and source license are in [`data_card.md`](data_card.md).
[OBSERVED] HiACC: 5,176 clips, 44 speaker IDs, 5.274 hours by WAV headers. Adult and child speaker IDs recur in **all three** released splits. `AD63` has 119 utterances but no demographic row; `AD65` has a metadata row but no audio. The released code-switch labels are utterance-level.
[OBSERVED] Lahaja and VITB-HEBiC content could not be accessed in this runtime. Their metadata claims were not converted into measured counts.
[OBSERVED] Every local clip was reread before scoring: all 8,312 are mono 16 kHz with manifest duration agreement and no exact duplicate audio hashes. MUCS has 4 near-silent clips and 1 clip with ≥0.1% saturated samples; all remain in the denominator. [`audio_qc.json`](audio_qc.json)

## Headline metrics

[OBSERVED] Primary decode is HiACC auto-language greedy and MUCS forced-Hindi greedy, frozen before the full forced-Hindi MUCS result existed. Rates are pooled by reference-token or character counts. CER counts spaces as characters. Brackets are 95% percentile CIs from 1,000 cluster resamples: verified speaker IDs for HiACC and recording/tutorial IDs for MUCS. `C` is a **provisional scoring condition**, not a different ASR model. `TermER` uses only the 200-item dictionary-candidate sample and is omitted when fewer than 30 candidate spans exist. Token classes with fewer than 30 reference tokens are retained in `metrics.csv` but omitted from this headline table.

| Dataset | Scoring | Metric | Rate % [95% CI] | Errors / denominator | Utterances / clusters |
|---|---|---|---:|---:|---:|
| mucs_hien | A_raw | WER | 59.19 [53.28, 66.24] | 22198 / 37500 | 3136 / 30 |
| mucs_hien | A_raw | CER | 50.31 [44.93, 56.78] | 96383 / 191566 | 3136 / 30 |
| mucs_hien | A_raw | TokER[LAT] | 61.06 [56.66, 65.45] | 5553 / 9095 | 3136 / 30 |
| mucs_hien | A_raw | TokER[DEV] | 35.97 [30.93, 42.08] | 10130 / 28164 | 3136 / 30 |
| mucs_hien | A_raw | TokER[NUM] | 52.15 [42.86, 62.67] | 97 / 186 | 3136 / 30 |
| mucs_hien | A_raw | TokER[MIX] | 100.00 [100.00, 100.00] | 55 / 55 | 3136 / 30 |
| mucs_hien | A_raw | INS_rate | 16.97 [14.05, 20.32] | 6363 / 37500 | 3136 / 30 |
| mucs_hien | A_raw | TermER | 60.38 [43.33, 76.09] | 32 / 53 | 3136 / 30 |
| mucs_hien | C_script_equiv_provisional | WER | 58.43 [52.43, 65.54] | 21910 / 37500 | 3136 / 30 |
| mucs_hien | C_script_equiv_provisional | CER | 49.37 [43.89, 55.86] | 94580 / 191566 | 3136 / 30 |
| mucs_hien | C_script_equiv_provisional | TokER[LAT] | 58.41 [54.06, 62.66] | 5312 / 9095 | 3136 / 30 |
| mucs_hien | C_script_equiv_provisional | TokER[DEV] | 35.99 [30.97, 42.17] | 10136 / 28164 | 3136 / 30 |
| mucs_hien | C_script_equiv_provisional | TokER[NUM] | 52.15 [42.86, 62.67] | 97 / 186 | 3136 / 30 |
| mucs_hien | C_script_equiv_provisional | TokER[MIX] | 100.00 [100.00, 100.00] | 55 / 55 | 3136 / 30 |
| mucs_hien | C_script_equiv_provisional | INS_rate | 16.83 [13.92, 20.15] | 6310 / 37500 | 3136 / 30 |
| mucs_hien | C_script_equiv_provisional | TermER | 60.38 [43.33, 76.09] | 32 / 53 | 3136 / 30 |
| hiacc | A_raw | WER | 37.03 [33.42, 40.63] | 18765 / 50674 | 5176 / 44 |
| hiacc | A_raw | CER | 29.43 [26.46, 32.34] | 70673 / 240161 | 5176 / 44 |
| hiacc | A_raw | TokER[LAT] | 33.80 [29.01, 41.32] | 6752 / 19977 | 5176 / 44 |
| hiacc | A_raw | TokER[DEV] | 26.91 [22.99, 31.53] | 8261 / 30694 | 5176 / 44 |
| hiacc | A_raw | INS_rate | 7.40 [5.22, 9.66] | 3749 / 50674 | 5176 / 44 |
| hiacc | C_script_equiv_provisional | WER | 35.90 [32.23, 39.56] | 18191 / 50674 | 5176 / 44 |
| hiacc | C_script_equiv_provisional | CER | 28.06 [25.07, 31.08] | 67401 / 240161 | 5176 / 44 |
| hiacc | C_script_equiv_provisional | TokER[LAT] | 30.96 [26.80, 37.42] | 6184 / 19977 | 5176 / 44 |
| hiacc | C_script_equiv_provisional | TokER[DEV] | 26.99 [23.06, 31.58] | 8284 / 30694 | 5176 / 44 |
| hiacc | C_script_equiv_provisional | INS_rate | 7.34 [5.16, 9.61] | 3720 / 50674 | 5176 / 44 |

## Change from A to C

[OBSERVED] Absolute percentage-point differences (A minus C); these quantify spelling-equivalence accounting only. The candidate lexicon was discovered from this evaluation material, so they are optimistic diagnostics and not held-out gains.

| Dataset | Metric | A % | C % | A − C points |
|---|---|---:|---:|---:|
| mucs_hien | WER | 59.19 | 58.43 | +0.77 |
| mucs_hien | CER | 50.31 | 49.37 | +0.94 |
| mucs_hien | TokER[LAT] | 61.06 | 58.41 | +2.65 |
| mucs_hien | TokER[DEV] | 35.97 | 35.99 | -0.02 |
| mucs_hien | TokER[NUM] | 52.15 | 52.15 | +0.00 |
| mucs_hien | TokER[MIX] | 100.00 | 100.00 | +0.00 |
| mucs_hien | INS_rate | 16.97 | 16.83 | +0.14 |
| mucs_hien | TermER | 60.38 | 60.38 | +0.00 |
| hiacc | WER | 37.03 | 35.90 | +1.13 |
| hiacc | CER | 29.43 | 28.06 | +1.36 |
| hiacc | TokER[LAT] | 33.80 | 30.96 | +2.84 |
| hiacc | TokER[DEV] | 26.91 | 26.99 | -0.07 |
| hiacc | INS_rate | 7.40 | 7.34 | +0.06 |

## Provisional lexicon and annotation support

[OBSERVED] Same-corpus candidate lexicon: 5211 Latin–Devanagari pairs; 129 automatically accepted. Status counts: {'COLLISION': 2760, 'AUTO_ACCEPT': 129, 'REVIEW': 2322}. It has no human approvals. The first 30 pairs by observed substitution count follow; complete list and examples are in [`lexicon_v0.csv`](lexicon_v0.csv).

| Latin | Devanagari | Substitutions | Status |
|---|---|---:|---|
| click | क्लिक | 350 | COLLISION |
| and | और | 88 | COLLISION |
| box | बॉक्स | 87 | COLLISION |
| button | बटन | 74 | COLLISION |
| file | फाइल | 72 | COLLISION |
| type | टाइप | 64 | COLLISION |
| text | टेक्स्ट | 63 | COLLISION |
| old | ओल्ड | 60 | AUTO_ACCEPT |
| slide | स्लाइड | 60 | COLLISION |
| cave | केव | 50 | REVIEW |
| house | हाउस | 45 | REVIEW |
| city | सिटी | 44 | REVIEW |
| next | नेक्स्ट | 44 | REVIEW |
| favourite | फेवरेट | 44 | COLLISION |
| veer | वीर | 39 | COLLISION |
| menu | मेन्यू | 39 | COLLISION |
| task | टास्क | 38 | AUTO_ACCEPT |
| so | तो | 37 | COLLISION |
| place | प्लेस | 37 | REVIEW |
| window | विंडो | 36 | COLLISION |
| like | लाइक | 35 | COLLISION |
| pack | पैक | 33 | REVIEW |
| friends | फ्रेंड्स | 33 | REVIEW |
| festival | फेस्टिवल | 33 | AUTO_ACCEPT |
| book | बुक | 33 | COLLISION |
| mission | मिशन | 31 | REVIEW |
| snacks | स्नैक्स | 30 | AUTO_ACCEPT |
| friendship | फ्रेंडशिप | 30 | REVIEW |
| document | डॉक्यूमेंट | 29 | REVIEW |
| dialog | डायलोग | 29 | REVIEW |

[OBSERVED] The coarse collision guard also rejects `click` ↔ `क्लिक`, despite 350 observed substitutions. C therefore misses plausible loanword equivalences as well as risking false accepts; its small A-to-C change is not an estimate of the total script-mismatch burden.

[UNVERIFIED] A sanity scan flags the automatically accepted `spot` ↔ `स्पोर्ट` pair as a probable false friend: *spot* and *sport* are different words. Bilingual review is required before treating C as validated script equivalence.

[OBSERVED] The 200 sampled utterances contain 54 dictionary-generated TERM spans. Across the full reference corpora, this dictionary matches 1504 MUCS and only 44 HiACC candidate TERM spans. The HiACC sample cannot support a term-error comparison. No LLM annotation service or human reviewer was used. The exact candidate guideline, coverage, and sample IDs are in [`annotation_guideline.md`](annotation_guideline.md), [`annotation_coverage.json`](annotation_coverage.json), and [`annotation_sample_ids.json`](annotation_sample_ids.json).

## Speaker-heldout script-equivalence control

[OBSERVED] A cluster-disjoint lexicon-development/evaluation split was frozen before the full decode completed: by verified speaker for HiACC and by tutorial/recording surrogate ID for MUCS. Only development-cluster references and predictions generate pairs; no heldout reference enters its lexicon. Exact normalized reference text still repeats across clusters, so a second analysis removes those repeated-text utterances. MUCS speaker identities are unverified, so its split is **not proven speaker-disjoint**. The pairs remain automatic and unreviewed. This is a **scoring protocol**, not a trained ASR improvement.

| Dataset | Heldout subset | Dev / test clusters | Accepted pairs | Utterances | WER A % | WER C % | Paired A − C points [95% CI] |
|---|---|---:|---:|---:|---:|---:|---:|
| mucs_hien | all_heldout | 15 / 15 | 29 | 1584 | 58.08 | 57.68 | +0.41 [+0.22, +0.64] |
| mucs_hien | novel_reference | 15 / 15 | 29 | 1441 | 58.21 | 57.83 | +0.38 [+0.19, +0.64] |
| hiacc | all_heldout | 22 / 22 | 47 | 2782 | 38.34 | 37.49 | +0.84 [+0.61, +1.11] |
| hiacc | novel_reference | 22 / 22 | 47 | 2377 | 40.40 | 39.64 | +0.76 [+0.53, +1.04] |

[OBSERVED] Exact normalized reference text overlaps the lexicon-development clusters for 143/1584 MUCS and 405/2782 HiACC heldout utterances. Complete paired rates and denominators are in [`heldout_lexicon_metrics.csv`](heldout_lexicon_metrics.csv); the frozen assignment (whose historical `unit: speaker ID` label is only a surrogate for MUCS) and overlap audit are in [`speaker_holdout_split.json`](speaker_holdout_split.json) and [`speaker_holdout_overlap.json`](speaker_holdout_overlap.json).

## HiACC descriptive group slices

[OBSERVED] These are pooled A-raw rates by released group metadata, with speakers as the bootstrap unit. Adult/child and gender are not causal or population-fairness comparisons. The full A/C matrix is in [`metrics.csv`](metrics.csv).

| Field | Level | Metric | Rate % [95% CI] | Errors / denominator | Utterances / speakers |
|---|---|---|---:|---:|---:|
| group | adult | WER | 37.64 [32.53, 42.74] | 12267 / 32593 | 3318 / 24 |
| group | adult | TokER[LAT] | 26.83 [22.66, 31.64] | 4398 / 16395 | 3318 / 24 |
| group | adult | TokER[DEV] | 30.27 [24.39, 38.37] | 4903 / 16196 | 3318 / 24 |
| group | adult | INS_rate | 9.09 [6.44, 11.66] | 2964 / 32593 | 3318 / 24 |
| group | children | WER | 35.94 [32.15, 40.25] | 6498 / 18081 | 1858 / 20 |
| group | children | TokER[LAT] | 65.72 [53.23, 79.22] | 2354 / 3582 | 1858 / 20 |
| group | children | TokER[DEV] | 23.16 [20.21, 26.54] | 3358 / 14498 | 1858 / 20 |
| group | children | INS_rate | 4.34 [2.27, 7.39] | 785 / 18081 | 1858 / 20 |
| gender | F | WER | 34.97 [29.75, 40.56] | 8434 / 24121 | 2325 / 19 |
| gender | F | TokER[LAT] | 33.67 [26.85, 44.32] | 3364 / 9992 | 2325 / 19 |
| gender | F | TokER[DEV] | 24.20 [20.88, 27.83] | 3419 / 14128 | 2325 / 19 |
| gender | F | INS_rate | 6.84 [3.80, 10.08] | 1650 / 24121 | 2325 / 19 |
| gender | M | WER | 39.44 [34.49, 44.36] | 9928 / 25174 | 2732 / 24 |
| gender | M | TokER[LAT] | 32.85 [26.11, 42.05] | 3150 / 9589 | 2732 / 24 |
| gender | M | TokER[DEV] | 30.15 [23.88, 37.45] | 4699 / 15583 | 2732 / 24 |
| gender | M | INS_rate | 8.25 [5.07, 11.52] | 2077 / 25174 | 2732 / 24 |

[OBSERVED] Group TermER is not reported where fewer than 30 candidate spans exist; missing HiACC gender metadata for AD63 excludes its 119 utterances from the gender slice only.


## Decoder QC decision

[OBSERVED] The forced-Hindi condition was stopped after 176 cached HiACC clips. In 176 matched early clips, forced Hindi had WER 115.84% (1975/1705) and auto-language decoding had WER 30.56% (521/1705). These order-biased clips are an engineering QC comparison, not a benchmark result. The forced condition generated contiguous phrase loops repeated at least four times in 11 of its cached clips. [`language_conditioning_qc.json`](language_conditioning_qc.json)
[OBSERVED] A separate 160-item matched MUCS engineering probe had auto-language WER 77.77% and forced-Hindi WER 62.96%; Arabic-script outputs changed from 44 to 0. The sample came from the first 640 ordered clips and is not a benchmark. [`mucs_language_probe_qc.json`](mucs_language_probe_qc.json)
[OBSERVED] On all 3136 MUCS clips, auto-language WER is 74.21% and forced-Hindi WER is 59.19%. The paired auto − forced difference is +15.02 percentage points [95% recording-cluster CI +9.79, +20.86]. Arabic-script outputs are 799 versus 0; the auto Arabic-script outputs contribute 10858 of 27830 total word errors. Contiguous phrase loops repeated at least four times occur in 23 versus 30 outputs. This is a post-QC decoder-setting comparison, not a held-out model improvement. [`mucs_full_language_comparison.json`](mucs_full_language_comparison.json)
[OBSERVED] The primary-output QC flags 51 HiACC and 63 MUCS utterances for at least one of: blank output, >3× reference length, a contiguous 1–4-token phrase repeated four times, or ≥200% utterance WER. HiACC has 31 loop-flagged clips contributing 2568 of 18765 word errors. Forced-Hindi MUCS has 30 such clips contributing 2289 of 22198 word errors. All remain in the corpus denominator. See [`prediction_qc.json`](prediction_qc.json), [`prediction_qc_flags.csv`](prediction_qc_flags.csv), and ten seed-selected [`prediction_qc_examples.md`](prediction_qc_examples.md).
[ASSUMED] The runbook's forced `language=hi` setting was replaced with auto language detection for HiACC after a matched QC failure. MUCS uses the original forced-Hindi setting after a separate matched QC probe. The per-corpus choice is adaptive to evaluation-source outputs, so neither corpus is an untouched decoder-selection benchmark. Model, revision, greedy decode, and 30-second chunk policy remained fixed.

## Assumptions and reproducibility

- [ASSUMED] NFC text, zero-width removal, Devanagari-to-ASCII digits, nukta removal, Latin-only lowercasing, punctuation removal, and collapsed whitespace are applied identically to references and hypotheses. Number words are unchanged.
- [ASSUMED] HiACC filenames' four-character `PID` prefixes are speaker IDs. This matches `speaker_info.csv` for all but the documented AD63/AD65 mismatch.
- [ASSUMED] C uses only `AUTO_ACCEPT` pairs with ITRANS-based folded distance ≤0.34, at least two observed substitutions, and no static or observed function-word collision. The threshold and guard need human audit.
- [OBSERVED] ASR model `openai/whisper-large-v3` is pinned to revision `06f233fe06e710322aca913c1bc4249a0d71fce1`; per-dataset config hashes, archive hashes, package versions, audio sample-rate checks, predictions, and token tables are local. Both full launch records and the frozen primary policy are in [`decode_launch.json`](decode_launch.json), [`forced_hi_launch.json`](forced_hi_launch.json), and [`decode_policy.json`](decode_policy.json). Full corpus text/audio and full predictions are intentionally not committed.
- [OBSERVED] Raw word alignments match jiwer on 20 fixed random utterances in the scoring preflight; the C relation is realigned with zero-cost approved pairs. Insertions are reported separately because TokER is reference-anchored.

## Questions for the team

1. Can the team obtain approved Lahaja and VITB-HEBiC access, and does VITB actually release speaker-region metadata?
2. Which HiACC group comparisons are scientifically justified after its split and metadata issues? Adult/child may still be descriptive; gender is confounded with collection context.
3. Can a bilingual reviewer audit all accepted script pairs and label domain terms in a fixed 200-item sample? The present dictionary candidates are not ground truth.
4. How should the cluster-heldout script-equivalence analysis be externally validated, given the lexicon is still automatically generated and MUCS speaker identities are unknown?
5. Which target paper venue and claim fit the data without implying a model improvement or population fairness result?

## Cost and time

[OBSERVED] Modal receipts report function wall time and estimated resource cost; provider billing is not verified and image build, startup, storage, and data transfer may add charges.

| Dataset | GPU | Function wall seconds | Estimated function-resource USD | Clips >30 s | Clips |
|---|---|---:|---:|---:|---:|
| hiacc_primary | NVIDIA H100 80GB HBM3 | 2358.9 | $2.80 | 0 | 5176 |
| mucs_hien_primary | NVIDIA H100 80GB HBM3 | 2362.8 | $2.80 | 4 | 3136 |
| mucs_hien_auto_diagnostic | NVIDIA H100 80GB HBM3 | 2029.3 | $2.40 | 4 | 3136 |
| mucs_hien_forced_hi_probe | NVIDIA H100 80GB HBM3 | 83.0 | $0.10 | — | 160 |
| **Primary decode sum** | | | **$5.59** | | **8312** |
| **All completed functions above** | | | **$8.10** | | |
[UNVERIFIED] Actual Modal invoice charge and the aborted HiACC forced-Hindi diagnostic charge are not included in this estimate.

## Known limitations

- Training-data contamination is unknown for the model on both public corpora.
- The C lexicon was mined on evaluation references, may reject genuine loanwords and accept false friends, and must not be used as a validated held-out claim; candidate TERM labels are unverified and HiACC has insufficient candidate terms for TermER.
- Number words are not normalized. CER under C maps only aligned equivalent token pairs to a common spelling before character alignment.
- HiACC's released splits share speakers; it cannot support an unseen-speaker claim without a new speaker-disjoint split.
- MUCS supplies 30 tutorial/recording surrogate IDs, not verified speaker identities or demographics; HiACC adult/child and gender cluster counts limit group precision.
- The [MUCS authors](https://arxiv.org/html/2104.00235) document subtitle/segment misalignment and incomplete audio, and describe manual transcript correction for the blind test. This pilot uses the released *test* split, so high-error examples need listening before any model-failure attribution.
- This is a one-model pilot with per-corpus decode settings, not a comparison of state-of-the-art open models or a trained-model intervention improving ASR.
- Some MUCS high-error hypotheses resemble neighbouring transcript content; it is unverified whether segment boundaries, annotation timing, or decoder behaviour caused this. Listening review is required.
- ASR output quality on very short clips and transcript errors need human listening review before publication.

## C1 code inventory

- [OBSERVED] Before this pilot, the `whispr-nano` checkout did not contain an `ast_asr` package. The reference file `src/ast_asr/metrics.py` was read from `snigenigmatic/ast-asr` branch `fair-cispo-work` (SHA-256 `3efb5f238719ca813e77d7b14fbfb22d4b0f225c59309da3507b8d99d32298f6`). The local `src/research_reset/scoring.py` reproduces that project's former capstone normalizer on a committed Svarah CSV, but its local implementation strips Devanagari entirely.
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
