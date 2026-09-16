# Reproducing the CPU analysis

Run commands from the repository root with the environment installed from
`paper/requirements.lock.txt`. No GPU, API key, or `ast_asr` package is needed.
The data and spelling-map files are frozen inputs; original prediction hashes
and utterance IDs are checked against `paper/data/input_manifest.json`.

```bash
.venv/bin/python -m unittest paper.tests.test_analysis -v

.venv/bin/python -m paper.analysis.analyze \
  --results-dir audit/svarah_fairness_audit \
  --matched-results-dir paper/data/matched_predictions \
  --spelling-map paper/data/whisper_normalizer.json \
  --metadata paper/data/svarah_metadata.csv \
  --cluster-column recording_cluster_id --cluster-kind recording \
  --output-dir paper/results

.venv/bin/python -m paper.analysis.analyze \
  --results-dir audit/svarah_fairness_audit \
  --matched-results-dir paper/data/matched_predictions \
  --spelling-map paper/data/whisper_normalizer.json \
  --metadata paper/data/svarah_metadata.csv \
  --cluster-column demographic_profile_id --cluster-kind demographic_profile \
  --output-dir paper/results/demographic_profile

.venv/bin/python -m paper.analysis.error_sensitivity \
  --matched-results-dir paper/data/matched_predictions
```

Omit `--matched-results-dir` to analyze only historical predictions. The complete
matched panel must contain exactly six models, each on the frozen Saaras 450 IDs;
Saaras predictions must remain unchanged. Partial panels fail validation. The
historical 156-utterance intersection is explicitly labelled provisional.
The attempted full-benchmark expansion stopped at a newline provenance guard and
was not retried after the request for faster delivery. No partial full-corpus
predictions are used. The optional `--full-results-dir` code
is retained for future work; it requires exactly five open systems and all 6,656
pinned metadata IDs and references, and is not part of the commands above.

`analyze.py` saves:

- `analysis.json`: provenance, options, frozen protocol hash, input validation,
  overlap reproducibility, full cohort metrics, uncertainty status, and diagnostics.
- `metrics.csv`: word-weighted overall, coarse-group, and native-language WER.
- `contrasts.csv`: fixed subgroup and paired model differences. Every inferential
  p-value is blank for filename or demographic-profile grouping.
- `support.csv`: utterance counts, language support, duration, gender counts, and
  counts of the selected blocks/profiles. These counts are not numbers of speakers.
- `diagnostics.json`: legacy independent-Poisson calculations, dispersion,
  three-group spread, language ranks, edit counts, and high-WER tail sensitivity.

`error_sensitivity.py` saves `per_utterance_scores.csv` for both normalizers and
all available models/panels, and `error_sensitivity.json` with the post hoc
Qwen-versus-small decomposition and digit-reference partition. It also includes
three short, non-sensitive transformation examples. Scores are independently
computed with JiWER; the core analysis uses its own word Levenshtein algorithm.
The fixed decomposition replaces error counts first, then reference lengths;
these are arithmetic components, not separate acoustic effects.

`inspect_normalization.py` optionally exports a larger set of deterministic
transformation examples. That output can include benchmark prompts containing
long numeric identifiers; the manuscript uses the shorter examples chosen in
`error_sensitivity.py` instead.

The protocol and its disclosed amendments are in `PROTOCOL.md`. Neither public
metadata grouping identifies verified speakers. `--cluster-kind recording` means
filename-derived blocks with uncertain recording semantics. Demographic profiles
are an exploratory sensitivity and can merge distinct people. Their percentile
intervals must not be presented as verified speaker-level inference.
