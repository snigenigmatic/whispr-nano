# Publication recovery plan for the 16 October 2026 review

Prepared 15 September 2026 for PW25_BJD_05.

## The commitment to make now

The October review deliverable is a working, reproducible experiment on whether
pseudo-label corrections learned from a labeled source accent corpus transfer to
Indian-English speech. It is not a promise of acceptance by October.

The first target is a four-page short paper for the BHASHA Workshop at ICON 2026,
conditional on written confirmation that Indian-accented English is in scope.
Its official call lists a 15 October deadline and 10 November notification;
accepted, presented papers enter the ACL Anthology. Contact the organizers at
`bhashaworkshop@gmail.com` before committing to this route.
[Official BHASHA call](https://bhasha-workshop.github.io/),
[submission details](https://bhasha-workshop.github.io/submission.html)

The committed safety net is the EACL 2027 Student Research Workshop. Its official
call accepts undergraduate first authors, includes speech recognition in scope,
explicitly welcomes informative negative results, and lists a direct submission
deadline of 15 December 2026 with notification on 5 January 2027. Accepted papers
are presented at EACL and included in the workshop proceedings.
[Official EACL SRW call](https://2027.eacl.org/calls/srw/)

INTERSPEECH 2027 remains the target for a stronger extended study. These backup
submissions are not claims that acceptance is guaranteed.

## Research question

**Do correction vectors learned from pseudo-label errors in one accent domain
improve pseudo-label adaptation on speaker-disjoint Indian-English speech, and
what predicts beneficial versus harmful transfer?**

This directly tests a limitation of Pseudo2Real rather than renaming its method.
Pseudo2Real learns a parameter-space difference between source models trained on
reference and pseudo transcripts and applies that correction to a pseudo-label
adapted target model. It already reports accent-specific correction on
AfriSpeech-200, so generic pseudo-label correction cannot be claimed as new.
[Pseudo2Real, Findings of ACL 2026](https://aclanthology.org/2026.findings-acl.59/)

## Minimum October 16 demonstration

1. Freeze speaker-disjoint SPIRE-SIES target train, development, and test
   partitions before model outcomes are inspected. Keep the final test labels
   unavailable to model and hyperparameter selection.
2. Have two reviewers independently audit 120 target-training recordings,
   balanced across native-language groups and teacher-confidence strata. Record
   teacher errors, reference errors, spoken departures, and normalization-only
   disagreements.
3. Implement and verify the following Whisper-tiny baselines with identical
   initialization and exposure budgets:
   - pretrained model;
   - ordinary target pseudo-label adaptation;
   - target pseudo-label adaptation with confidence filtering;
   - Pseudo2Real using L2-ARCTIC as the labeled source;
   - target-reference training as a disclosed upper bound, not a label-free method.
4. Select correction strength and checkpoints on development speakers only.
   Report macro WER, micro WER, every sufficiently supported native-language
   group, worst-group WER, and paired speaker-level uncertainty.
5. Publish a local reproduction command, frozen manifests, run receipts, raw
   predictions, and an independent metric check. The review demo must work from
   stored evidence without paid inference.

## Calendar and decision gates

### 15–18 September: protocol and data integrity

- Send the advisor response below and circulate this plan to the team.
- Email the BHASHA chairs for a written scope ruling using the draft below.
- Confirm the Humyn Labs call; ask for methodological criticism and real-world
  error categories, without implying a partnership or access to proprietary data.
- Freeze SPIRE-SIES speakers, group-support rules, metrics, baselines, compute
  ceiling, and final-test policy.
- Generate the blind audio-audit form and complete the first 20 shared examples
  to calibrate the two reviewers.

**Gate A:** Stop or repair the dataset if the audit shows that transcript quality
cannot support the intended error claims.

### 19–25 September: reproduce the mechanism

- Extend the existing trainer to switch between reference and teacher targets
  while preserving the same scheduled audio exposure.
- Run the fixed reference-versus-teacher target diagnostic on L2-ARCTIC.
- Reproduce the Pseudo2Real parameter arithmetic and validate weight hashes,
  scaling, and source-only tuning.

**Gate B:** Continue only if teacher targets create a measurable, repeatable
penalty or the correction baseline changes that penalty. A null result is kept
and reported; it does not trigger metric or subgroup shopping.

### 26 September–5 October: Indian-English transfer study

- Generate teacher transcripts for the frozen SPIRE-SIES target-training split.
- Train the target pseudo-label and confidence-filtered baselines.
- Apply the source correction vector and run three training seeds for decisive
  comparisons.
- Audit failures by speaker, native-language group, utterance duration, and
  teacher error type.

### 6–12 October: robustness and independent checks

- Repeat the main comparison with one additional source partition or teacher if
  the primary effect is nontrivial.
- Run speaker-level paired bootstrap intervals and check both project
  normalizers.
- Recompute all headline metrics from raw predictions using an independent
  script.

### 13–16 October: review package

- Deliver an eight-slide deck: problem, closest work, hypothesis, protocol,
  implementation, main result, failure analysis, and submission timeline.
- Show a live offline reproduction and one model comparison, not a collection of
  unrelated experiments.
- If BHASHA confirms scope and the evidence passes the gates, upload by the
  internal 14 October deadline and show the submission receipt. Otherwise show
  the EACL SRW mentorship and submission plan. Present INTERSPEECH as the later
  extended target.

## Draft response to the guide

> Dear Sir,
>
> We agree that the INTERSPEECH 2027 decision timeline is too late to be our only
> publication plan. We are confirming scope with the BHASHA Workshop at ICON
> 2026, whose short-paper deadline is 15 October and notification date is
> 10 November. Accepted, presented papers are included in the ACL Anthology. Our
> committed safety net is the EACL 2027 Student Research Workshop: its direct
> submission deadline is 15 December and its notification date is 5 January.
>
> For the 16 October review, we will demonstrate a speaker-disjoint Indian-English
> experiment on whether pseudo-label error corrections learned from a labeled
> source corpus transfer to SPIRE-SIES. The package will include the implemented
> baselines, frozen protocol, reproducible runs, independent metric checks, and
> initial results. INTERSPEECH will remain the target for a stronger extended
> version if the evidence supports it.
>
> We are speaking with the Voice AI team at Humyn Labs for methodological and
> real-world evaluation feedback. We will describe this as external discussion
> unless a concrete collaboration is agreed.
>
> We understand that a submission is not a publication and that acceptance
> cannot be promised. We will report progress and evidence at the October review.
>
> Best,  
> Aditya Sharma  
> On behalf of PW25_BJD_05

## Draft scope inquiry to BHASHA

> Subject: Scope inquiry: Indian-English ASR evaluation and adaptation
>
> Dear BHASHA Workshop Chairs,
>
> We are preparing a short paper on Indian-English automatic speech recognition.
> The study evaluates speaker-disjoint speech grouped by the speakers' Indian
> native-language backgrounds and examines whether pseudo-label corrections
> learned from a source accent corpus transfer to Indian-English speech. It uses
> Indian-English audio and English transcripts; it is not recognition of an
> Indian language itself.
>
> Would this topic fall within the scope of BHASHA 2026, particularly its focus
> on benchmarks, evaluation metrics, and human-centric AI for Indian languages?
> We would appreciate a scope ruling before preparing the submission.
>
> Best,  
> Aditya Sharma  
> On behalf of PW25_BJD_05, PES University

## Team ownership

- **Experiment lead:** target-source switching, Pseudo2Real reproduction, and
  run ledger.
- **Data lead:** SPIRE-SIES split audit, speaker/group support, and provenance.
- **Evaluation lead:** independent scoring, speaker bootstrap, and error tables.
- **Paper lead:** literature matrix, EACL format, review deck, and artifact index.

Every result receives one technical owner and one independent checker. No result
enters the deck from a screenshot or manually copied number.
