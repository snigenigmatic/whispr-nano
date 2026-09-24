# Validator correction: capped output can normalize to empty text

On 2026-09-22 the Turbo Assamese short-recording job completed and exported all
980 predictions, then exited with code 2 during validation. Its original HF
status remains failed. The sole validation error concerned recording
`fleurs-as_in-test-9437161079844375131`: it generated punctuation until the token
limit, without EOS. The frozen scorer removes punctuation, so its scored
hypothesis is empty. Its preserved metrics are WER 1.0 and CER 1.0, with all 20
reference words and 119 reference characters counted as deletions.

The runner correctly labeled this record `generation_limit`. The validator
incorrectly assumed a capped generation must have nonempty normalized text.
Removing that assumption permits this bounded recognition failure to be
included in evaluation. The validator still requires raw token evidence,
consistent EOS status, complete audio coverage, exact source/model hashes,
and independently recomputed metrics and aggregates. `success` with empty text
still fails, as does capped output mislabeled as success or empty output.

No inference was rerun. No raw prediction, token sequence, scoring rule or
metric was modified. Previous validation receipts are retained alongside the
new receipt, which records the corrected validator's SHA-256. Capped and
normalized-empty diagnostics overlap and are reported as such. The original
remote validator and its failed receipt remain in the immutable run snapshot.

Regression checks cover a complete punctuation-only capped artifact, retained
100% deletion errors, mislabeled success, and overlapping diagnostics. The
validator/analysis test suite passed 14 tests. All 980 records subsequently
passed independent validation.
