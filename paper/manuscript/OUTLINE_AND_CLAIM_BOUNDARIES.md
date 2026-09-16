# Manuscript outline and claim boundaries

Working title: **Which Indian-Accent Disparities Survive Evaluation Choices? A Replication and Reliability Study on Svarah**

Intended format: short empirical paper, ordinary Markdown, numeric references with linked bibliography. An anonymous author line is appropriate for a review draft; author order and venue declarations are not invented.

## Central contribution

An independent 2026 case study asks whether observed native-language disparities remain stable after matching the evaluated utterances and changing the scoring and uncertainty assumptions. It adds a Saaras V3 snapshot and contemporary open recognizers to a benchmark that already documented accent differences. The contribution is the measured stability, instability, and inferential limits of these particular comparisons, not a new fairness statistic or the first discovery of Indian-accent differences.

## Structure

1. Abstract: empirical question, two evaluation panels, strongest observed result, qualification about speaker identity and generalization.
2. Introduction and related work: Svarah already reports 19 accent breakdowns; distinguish this replication from ASR-FAIRBENCH, foundational ASR inference methods, and the public indian-asr-bench artifact.
3. Evaluation design: 450 shared utterances with 150 each from Indo-Aryan, Dravidian, and Bodo labels; fresh open predictions and archived Saaras; historical 2,500-utterance open panel. Version provenance and overlap checks matter.
4. Scoring and estimands: legacy and Whisper English normalization applied symmetrically; corpus WER; Bodo-minus-pooled-others; three-group spread; all 19 language summaries; uncertainty limited to verified clustering units.
5. Results: sample support, matched model/group table, normalization stability table, legacy Poisson diagnostic, cluster/sampling sensitivity.
6. Discussion: distinguish recognition differences from representation/scoring artifacts; coarse grouping can conceal individual-language effects; uncertainty and data access are findings rather than inconveniences to suppress.
7. Limitations, ethics, reproducibility, conclusion, references.

## Claims that can be made if supported by the final artifacts

- On the specified utterances, a named model has a lower/higher corpus WER under a specified normalizer.
- A named native-language group's observed WER gap retains/reverses its sign across two scoring procedures.
- The original independent-utterance Poisson assumptions are contradicted by a reported dispersion diagnostic; its p-values should not be interpreted as validated population evidence.
- A difference persists under the particular available resampling procedure, with that procedure's clustering unit and interval limitations stated.
- All tables regenerate from released transcript/count artifacts without new inference.

## Claims that must not appear

- First Indian-accent audit; first multi-system Svarah audit; first significance-tested ASR fairness study.
- A p-value below .05 proves a structural or causal bias; a p-value above .05 establishes parity.
- Bodo represents all Sino-Tibetan accents; native language is a direct acoustic measurement of accent.
- Recording identifiers or demographic tuples are verified speakers.
- The 450-utterance balanced panel estimates the average experience of Indian English speakers or the full Svarah corpus.
- The 450 and 2,500 panels are independent replications: they overlap and come from the same corpus.
- A vendor/API snapshot is a timeless claim about that vendor, or checkpoint selection establishes state of the art.
- Relative performance changes identify the causal effect of model scale or distillation.
- A preprint or repository has passed peer review without a publisher/venue record.
- The sprint's analysis was preregistered before observing any audit results. Bodo was selected using prior results; this is a retrospective reliability analysis.

## Completion status

The manuscript and supplement contain the completed Matched-450 and Open-2500 results only. The six-system matched table, 156-utterance overlap check, normalizer provenance, verified metadata availability, source scoring checks, figures, and bibliography are incorporated. Speaker identity remains an explicit limitation rather than an unresolved drafting placeholder. No partial full-corpus inference results enter the paper. Root owns the final PDF rendering and end-to-end reproduction check.
