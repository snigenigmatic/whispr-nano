# Analysis protocol frozen at the start of the paper sprint

Date: 2026-09-09. This is an analysis plan written after inspection of the
existing pilot results, not a prospective preregistration. The pilot inspection
revealed unmatched Saaras samples, overdispersed error counts, a Bodo-only
Sino-Tibetan category, and occasional very large insertion errors. These
observations informed the choices below. New results will not be selected for
statistical significance.

## Samples and reuse

- The existing five open models share 2,500 utterance IDs. Analyze that exact
  intersection as the open-system panel.
- The existing Saaras sample contains 450 utterances, 150 per coarse group.
  Complete predictions for the other models on the 294 IDs they lack. Analyze
  all six on those same 450 IDs as the primary cross-system panel.
- Until completion, the available six-system intersection has 156 utterances.
  Label this panel provisional; never present it as the completed 450 panel.
- Merge predictions by immutable utterance ID. Verify identical reference text,
  language, and other shared metadata for overlapping predictions. Never silently
  overwrite an existing hypothesis. Require a dataset revision and check reference
  identity before recovering metadata by row index.
- Retain every utterance and all hallucinated insertions in the primary metric.
  No adaptation/training is performed on this benchmark.

## Outcomes and interpretation

- Corpus WER is total substitutions + deletions + insertions divided by total
  reference words. Report percentages; differences are percentage points.
- Report native-language WER and support for all 19 represented languages.
- Report coarse groups Indo-Aryan, Dravidian, and **Bodo**. Bodo is the only
  represented Sino-Tibetan language; this result does not generalize to that
  entire family. The max-minus-min metric is a *three-group WER spread*, not
  demographic parity. Report the equal-three-group mean as a separate descriptive
  quantity alongside word-weighted overall WER.
- The primary fixed subgroup contrast is Bodo minus pooled non-Bodo WER.
  Secondary contrasts are Bodo minus Dravidian, Bodo minus Indo-Aryan, and
  Indo-Aryan minus Dravidian. Individual-language results are descriptive;
  do not selectively test whichever language has the largest observed error.
- Paired system contrasts compare each other system to Whisper large-v3 on
  precisely the same utterances. This baseline was chosen before completing the
  matched panel and is not the empirically best system.
- These are associations conditional on the dataset and evaluation configuration,
  not causal estimates of accent discrimination or population prevalence.

## Normalization

Apply each normalizer identically to reference and hypothesis. Recompute edit
counts from text, rather than transform existing WER values.

1. `minimal`: lowercase; replace characters outside Unicode word characters,
   whitespace, and ASCII apostrophes with spaces; collapse whitespace. This
   self-contained rule reproduces the stored pilot normalizations. Its exact
   regular expression is `[^\w\s']`.
2. `whisper_english`: Transformers' Whisper `EnglishTextNormalizer`, with the
   pinned OpenAI Whisper English spelling map in
   `paper/data/whisper_normalizer.json`. Record the implementation version and
   spelling-map SHA-256. This includes shared numeral/contraction handling.

Both results are reported. The minimal score preserves the historical evaluation;
the English-normalized score is a presentation sensitivity analysis. A changed
score can reflect reference-denominator changes and is not an acoustic improvement.
No vendor-specific fixes or hand-edited transcripts are permitted. If a normalizer
creates an empty reference, retain hypothesis insertions in corpus totals and
report the count; never divide by zero at utterance level.

## Uncertainty and multiplicity

- Only verified speaker identifiers support speaker-level inference. Never derive
  speakers from utterance indexes, invent an identifier, or equate a recording
  identifier to a person without evidence. Metadata must cover every utterance of
  a panel before intervals are calculated.
- With verified speakers, bootstrap speakers with replacement within each
  native-language stratum (2,000 resamples, seed 20260909). Keep all utterances of
  each sampled speaker together; use the identical resampling weights for every
  model and normalization in the panel. Thus paired differences preserve
  dependence, and uncertainty is conditional on observed language composition.
- Report percentile 95% intervals. If fewer than two independent clusters support
  either side of a contrast, withhold its interval. With fewer than ten clusters
  on either side, intervals are explicitly exploratory and p-values are withheld.
  The threshold of ten is a conservative reporting choice, not a guarantee that
  asymptotic inference is valid.
- Where cluster support meets that threshold and identifiers are verified speakers,
  report a two-sided centered-bootstrap tail probability with a +1 finite-sample
  correction, clearly labelled approximate. Apply Holm adjustment to the four
  fixed subgroup contrasts within each panel/model/normalization, and separately
  to paired model contrasts within each panel/normalization. These are sensitivity
  analyses, not the evidentiary basis for a blanket fairness verdict.
- If only verified recording IDs exist, optionally calculate recording-bootstrap
  intervals labelled diagnostic. Suppress all p-values and explicitly state that
  repeated recordings of one speaker can invalidate person-level interpretation.
- If metadata are absent or incomplete, publish point estimates and support;
  inferential uncertainty remains unsupported. Do not replace missing speaker
  uncertainty with an utterance bootstrap.

## Diagnostics and audit trail

Recompute the historical independent-observation Poisson family deviance test
only as a diagnostic and quantify its Pearson dispersion. No p-value becomes a
"structural" verdict. Report duplicate reference counts, total error components,
and the fraction of errors contributed by utterances whose WER exceeds 100%.
Report tail-excluded WER separately as a sensitivity result, preserving those
utterances in primary results. This is not a claim that difficult examples may be
discarded in deployment. Save full input file hashes, analysis options,
normalizer provenance, sample sizes, cluster coverage, and machine-readable
results. Record any later change to this protocol explicitly.

## Amendment before new predictions (2026-09-09, first sprint hour)

To avoid mixing model/dataset revisions within one panel, the inference owner
will rerun all 450 matched utterances for each open model, with the historical
decoding configuration, rather than splice 294 newly decoded predictions into
156 old predictions. The historical Saaras 450 predictions remain unchanged.
The new 450 panel is a separate input directory. Historical overlaps are compared
as a reproducibility diagnostic; differing hypotheses are never silently merged.
This change was made before observing new predictions and does not affect sample
selection, metrics, or statistical contrasts.

Metadata inspection subsequently verified that current published files expose no
speaker IDs. The filenames supply recurring `f` tags, with matching language and
demographic fields within each tag. Their interpretation as physical recordings
is not independently established. Therefore the planned fallback uses those
**filename-derived blocks** only as a diagnostic of block dependence. The CLI
label `--cluster-kind recording` denotes this fallback and does not assert that
these blocks identify persons or verified recording sessions. All p-values remain
withheld. Any later alternate filename grouping is a labelled sensitivity, not a
replacement primary result selected for a narrower interval.

## Exploratory metadata sensitivity amendment

After the first normalization results were observed, metadata inspection found
115 distinct exact profiles of eight categorical fields in the full dataset,
compared with 117 speakers reported by the dataset paper. Neither proximity of
counts nor matching demographics proves speaker identity. At the inference
owner's request, analyze these exact profile IDs as an additional *exploratory*
grouping sensitivity, saving results in a separate directory. Profiles may merge
distinct people or split one person across inconsistent records. The 450 sample
contains 92 profiles, and Bodo is supported by only five profiles in each panel.
Use the same paired, language-stratified resampling mechanism and report wider or
narrower intervals without selection. Suppress every p-value, retain the original
filename-block diagnostics, and never describe the profile results as verified
speaker bootstrap. The related coarser filename prefix may be analyzed with the
same non-person interpretation if useful.

After seeing normalization sensitivity, add deterministic tables of Bodo's rank
among observed native languages versus its rank among the coarse groups. These
are descriptive decompositions of the same scored rows, not new hypothesis tests.

## Full-benchmark expansion amendment, before full predictions

After the complete 450 panel exposed a 200-word repetitive Whisper-small output
on a one-word Tamil clip, the inference owner expanded the open-model evaluation
to **every one of the 6,656 pinned Svarah test utterances**. This decision was
triggered by the observed sample/tail sensitivity and was made before examining
full-corpus outputs. All five open systems are evaluated under the existing
decoding settings; every utterance remains in the primary corpus totals. The
full corpus supplies a benchmark census, while the 450 panel remains the paired
comparison including the historical commercial API predictions. The historical
2,500 panel remains separately reported for replication and sample sensitivity.

The new `full_open_6656` cohort must contain exactly the five frozen open models
and all pinned metadata IDs. Validate every reference, native language, family,
gender, and available duration. Store new prediction hashes and compare overlaps
with both earlier panels, without silently replacing earlier hypotheses. Use the
same scoring, fixed contrasts, tail diagnostics, and two unverified block/profile
sensitivities. A census removes subset selection within this particular benchmark;
it does not add independent people or establish speaker-population inference.
Extend the explicitly post hoc Qwen/small decomposition to this full panel, while
keeping its exploratory status. Partial full-corpus jobs must not be reported as
the full benchmark.

## Cancellation of full-benchmark expansion

The full-corpus attempt stopped at 2026-09-09 07:00:30 UTC when a provenance guard
encountered a CRLF newline comparison at `svarah_01927`: the host CSV reader had
normalized newlines. The reader was corrected to preserve them with `newline=''`.
The user requested faster delivery at 07:00:53 UTC, so the broader attempt was
not retried. No partial full-corpus outputs are used in this paper or its reported analyses. The final
artifact reports only the completed 450-utterance paired commercial/open panel,
the historical 2,500-utterance open panel, and explicitly labelled historical
support/intersection diagnostics. Optional full-corpus validation code remains
for a future extension, but no `full_open_6656` cohort is included in final results.
The decision not to retry was for delivery time, not selection based on
full-corpus outcomes. The completed 450/2,500 panels were unaffected. The earlier
amendment is retained to make the attempted scope change transparent.
