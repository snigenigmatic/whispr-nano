# Humyn Labs call brief

## Purpose

Use the 20-minute call to test whether the proposed correction-transfer question
matches real Voice AI failure modes. Do not pitch a finished method, request
confidential customer data, or describe the discussion as a partnership.

## Two-minute opening

> We are studying Indian-English ASR reliability. Our first controlled experiment
> tested whether teacher-error filtering removed useful accented speech, but the
> registered effect did not appear. We are now testing a narrower question:
> whether corrections learned from pseudo-label errors in one accent domain
> transfer to unseen Indian-English speakers. We would value your criticism of
> the evaluation and your view of which recurring error classes matter in real
> Voice AI systems.

## Agenda

- **0–3 minutes:** problem and completed negative experiment.
- **3–7 minutes:** proposed source-to-Indian correction-transfer setup.
- **7–15 minutes:** Humyn's experience with recurring ASR failures.
- **15–18 minutes:** feedback on evaluation and useful non-sensitive artifacts.
- **18–20 minutes:** agree on one concrete follow-up, if any.

## Questions to ask

1. Which errors recur across speakers: names, numbers, entities, code-switching,
   pronunciation variants, noise, or turn-taking?
2. Which aggregate ASR metrics have failed to reflect production quality for
   your team?
3. When an upstream recognizer is wrong, do downstream systems amplify the same
   error pattern?
4. Would a model comparison split by error class and speaker background be useful
   to an industry evaluator?
5. What would make a correction-transfer result operationally convincing rather
   than merely statistically interesting?
6. Can you recommend public test material or a non-confidential taxonomy of
   failures that the study should cover?
7. Would you be willing to review a one-page frozen protocol after the call?

## Evidence to show only if useful

- Nine controlled H100 runs, all independently audited.
- The registered filtering-harm hypothesis was not supported: filtered minus
  matched-random macro WER was -0.118 percentage points.
- Existing assets include a teacher/student pipeline, pseudo-labels, immutable
  run ledgers, raw predictions, and offline metric reproduction.
- The next experiment uses speaker-disjoint Indian-English evaluation rather
  than presenting the negative selection result as a contribution.

## Statements to avoid

- “Humyn Labs is collaborating with us” unless both parties explicitly agree.
- “We have solved accent bias.”
- “The current result is publication-ready.”
- “The references are unquestionably correct” before the manual audio audit.
- Any promise to share recordings or company information without permission.

## Concrete follow-up

Send a one-page protocol containing the research question, data partitions,
baselines, metrics, and error taxonomy. Ask for comments on the protocol only.
Record any advice as external feedback; scientific claims still require the
project's own reproducible evidence.

