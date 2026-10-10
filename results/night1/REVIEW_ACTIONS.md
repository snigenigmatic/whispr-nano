# What to show on 16 October, and what still needs evidence

This pilot is a completed, reproducible public-corpus result, **not yet a publishable paper**. Present the measured decoder-setting failure and the data-quality audit from [`RUN_REPORT.md`](RUN_REPORT.md). Do not present the automatic script lexicon as a model improvement or the HiACC adult/child slice as a fairness result.

| Work | Concrete completion criterion | Suggested owner |
|---|---|---|
| Listen to flagged audio | Start with the ten seed-selected items in [`prediction_qc_examples.md`](prediction_qc_examples.md), then sample more from [`prediction_qc_flags.csv`](prediction_qc_flags.csv). Record whether the speech, released transcript, segment boundary, or decoder caused each discrepancy. | Two bilingual team members |
| Validate script pairs | Review all 129 `AUTO_ACCEPT` pairs in [`lexicon_v0.csv`](lexicon_v0.csv) plus high-frequency `COLLISION` pairs such as `click`/`क्लिक`. Record approve/reject/ambiguous and have a second reviewer independently label a shared subset. Re-score the frozen speaker-heldout partition with only approved pairs. | Two bilingual team members; agent can rerun scoring |
| Validate terms | Correct the 200 candidate annotations in [`annotations_heuristic_v0.jsonl`](annotations_heuristic_v0.jsonl) while blinded to ASR output. The fixed sample has only 53 MUCS and one HiACC candidate TERM span; if term analysis remains the paper focus, freeze a term-rich MUCS sample from reference text before looking at its predictions. | Team annotators; agent can prepare queue and rerun TermER |
| Add a comparison model | Choose an open Hindi/Hinglish-capable ASR model with a documented inference recipe, then decode the same IDs and use the same scorer. Freeze its language/task settings using a development partition before evaluating heldout speakers. | Agent |
| Make the paper claim | Use the 16 October review to decide whether the supported direction is decoder reliability on Hindi-English task speech, a validated script-scoring benchmark, or something else. A submission needs a second model or intervention, reviewed labels, and listening-checked errors. | Kaustubh and guide |

The two gated datasets were inaccessible in this runtime. Region or speaker-fairness claims need approved access and verified metadata; the present public corpora do not establish them. No contact with Humyn Labs or Sarvam is required to use these public-corpus results.
