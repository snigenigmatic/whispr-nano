# Natural Hindi reference-repetition check

This descriptive follow-up was added after the 1,101-input decoder matrix, while the short-Hindi jobs were running. It addresses the constructed nature of the keyword controls. It is not a preregistered external confirmation.

Search the pinned Lahaja `verbatim` references among all **4,556 recordings from non-development speakers**. After the same NFC/case/punctuation normalization, 4,095 have no adjacent repeated word, 434 have a longest run of two, 23 a run of three, three a run of four, and one a run of five or more. Retain every recording with at least three adjacent identical words and duration at most 25 seconds: **27 clips, 24 speakers, 343.593 audio seconds**. Six recording IDs overlap the preceding 801-clip study; 21 are additional recordings. No model output was used for selection.

The references are existing dataset annotations. They have not been independently adjudicated by two new human listeners. This set is richer than synthetic concatenation, but it does not remove every annotation uncertainty. Only one reference crosses the five-word repair threshold, so most of the set is a near-threshold preservation check rather than a direct test of deleting a correctly transcribed five-word run.

Run official Small and official Turbo on the identical manifest with pinned decoding recipes and H100 hardware, and rescore the raw, frozen text-collapse, and published duration-cap outputs. Report every recording in aggregate, word edits, paired speaker-bootstrap intervals, and helped/harmed counts. The set is too small and selected to estimate a population repetition failure rate. No threshold tuning or training follows from these labels.

Manifest SHA-256: `bcfedf6257fca9c6293cf435433483c312c9e1826b9c9bd73d529df701e023db`.

| Model | Commit | ORX experiment |
|---|---|---|
| Small | `c431a6a` | `1f643aa3-a071-43fc-9e51-95030a5ac5e1` |
| Turbo | `cb2ab15` | `348cb56e-125c-4959-8e07-90e51eefef75` |

Each run has a 15-minute Modal timeout, a 10-minute inference guard, and a $2.25 reservation. Including these controls, Modal reservations are **$19.00**, within the separate $20 cap; no further Modal job fits the conservative minimum useful-run allowance. Reservations are not final charges. Complete hash-checked bundles remain private; the same independent validator accepts this explicitly pinned 27-input protocol and publishes only aggregates.
