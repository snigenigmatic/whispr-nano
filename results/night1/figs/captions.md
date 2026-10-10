# Draft captions for descriptive pilot figures

- `forest_metrics`: Error rates on MUCS (forced-Hindi greedy) and pooled HiACC (auto-language greedy) for Whisper large-v3, following the frozen per-corpus decoder policy; intervals are 95% speaker-cluster bootstrap over 1,000 resamples. Candidate TermER uses unreviewed dictionary spans; C is a provisional script equivalence computed from the evaluation corpora.
- `conditions_a_c`: Raw and provisional script-equivalent error rates. C is a scoring diagnostic, not a model improvement; its lexicon was discovered from the same corpus.
- `term_error_types`: Distribution of substitution, deletion, and mixed errors among wrong dictionary candidate terms, shown only for datasets with at least 30 candidate spans. Denominators are in `metrics.csv`.
- `hiacc_groups`: Adult/child and gender slices from pooled HiACC. The released train/validation/test splits share every speaker, and 119 AD63 utterances lack gender metadata. Descriptive only.
