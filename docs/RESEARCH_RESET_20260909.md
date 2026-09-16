# Research assessment and replacement experiment

9 September 2026. This supersedes the earlier recommendation to treat the
three-hour manuscript as a submission candidate. It is a research decision memo,
not a new paper or a claim of improved recognition.

## Reviewer decision

**I would reject the current manuscript for a main conference track or a journal,
and I would not recommend submitting it in its current form.** A workshop focused
on evaluation might consider a substantially reframed case study, but that would
not meet the ambition now expressed for this project. This is my assessment,
not a prediction of an actual committee decision.

The sprint produced useful data repair and an inspectable technical report.
Calling the result a submission candidate gave too much weight to having a
finished document and too little to its scientific contribution. Passing code
tests establishes neither novelty nor the importance of the research question.

### Reasons for rejection

1. **The central contribution is weak against known work.** Svarah already
   compares recognizers across native languages. Text normalization and
   dependent-observation inference are established issues. Adding contemporary
   model snapshots is useful maintenance, but the manuscript has not shown a
   general finding that substantially advances those accounts. Its own related
   work identifies a close public benchmark with overlapping analyses.
2. **Several results primarily repair our earlier claims.** Correcting unmatched
   cohorts or a formatting-sensitive ranking is necessary. It does not establish
   a new property of acoustic recognition. The dramatic counterexample involving
   one repetitive short-clip output is fragile to decoding choices and does not
   yet explain a general mechanism.
3. **The evidence cannot carry strong accent-generalization claims.** One corpus,
   an uneven 450-clip panel, five unverified Bodo profiles, and no recovered
   speaker identifiers limit independent replication and uncertainty. Thousands
   of clips cannot substitute for independent speakers. The 450/2,500 panels
   overlap and are not independent datasets.
4. **There is no tested intervention or causal explanation.** The study does not
   determine whether an error difference comes from pronunciation, content,
   recording conditions, decoding, or training-data selection. It does not
   improve the system. A strong empirical paper need not invent an algorithm,
   but it must establish consequential new knowledge; this draft has not yet
   earned that claim.
5. **The presentation follows the available analyses instead of one resolved
   question.** Normalization, an isolated repetition, coarse labels and uncertain
   blocking all receive attention, but their combination is not a demonstrated
   mechanism. A better title or venue template would not fix this.

The existing investment remains useful: raw hypotheses, deployment experience,
working teacher/student interfaces, and error accounting make real experiments
cheaper to design and diagnose. The report should become background and an
internal evaluation reference, not the destination of the project.

## What the literature now rules out

The [primary-source frontier review](RESEARCH_FRONTIER_20260909.md) gives detailed
method overlap and sources. Its main constraints are:

| Tempting proposal | Existing precedent | Consequence |
|---|---|---|
| On-policy distillation for ASR | [Ark-ASR](https://arxiv.org/abs/2605.28139) | The basic transfer to speech is already demonstrated. |
| Protect groups from a biased teacher | [Teacher's Pet](https://arxiv.org/abs/2106.10494) | Per-group teacher/reference mixing is a baseline. |
| Use ground truth when the teacher is wrong | [Conditional T/S, ICASSP 2019](https://arxiv.org/abs/1904.12399) | Correctness gating alone is not a new method. |
| Keep reliable words and mask unreliable ones | [Du et al., Interspeech 2022](https://www.isca-archive.org/interspeech_2022/du22_interspeech.pdf) | Both loss masking and history masking already exist. |
| Learn across missing transcript spans | [Star Temporal Classification](https://arxiv.org/abs/2201.12208) | Partial supervision requires comparison with existing sequence objectives. |
| Add fairness losses to Whisper | [Fairness-prompted fine-tuning, ICASSP 2026](https://doi.org/10.1109/ICASSP55912.2026.11464390) | Ordinary fine-tuning, balancing and Group DRO must be strong baselines. |
| Gate OPD by correctness | [OPDVR](https://arxiv.org/abs/2608.24696) | A correctness-aware OPD label is insufficient novelty. |

These are not reasons to abandon the problem. They specify the methods a new
contribution must improve upon. Merely combining them with a new acronym would
repeat the same mistake as the first manuscript.

## Recommended question

**Can we preserve the speech examples a small recognizer most needs to learn
without transferring the teacher's mistakes?**

The specific mechanism to investigate is whole-utterance rejection during
distillation. A teacher makes a few errors on a difficult utterance; filtering
removes the entire recording. This can change both the number of examples from
an accent and the kinds of pronunciations, words and acoustic conditions that
remain *within* that accent. Equalizing group counts after rejection cannot
restore examples with zero selection probability. This is a reasoning statement
about selection support, not evidence that a particular accent gap was caused
by this mechanism.

The research target is to demonstrate that within-accent loss of useful
supervision causes a reproducible failure, then remove that failure more
effectively than existing conditional supervision or token masking. A simple
existing method may win. In that case we should use it and reassess the paper's
contribution rather than hide it behind a new loss.

### Initial diagnostic actually completed

I implemented and ran a new analysis on the frozen Open-2500 predictions.
The primary teacher was fixed as Whisper-large-v3, the student as Whisper-tiny,
and rejection thresholds as 10%, 20%, 40% before computing this diagnostic.
The 10% choice comes from the original Distil-Whisper filtering analysis; its
quality/coverage tradeoff is already acknowledged there. The following values
use symmetric Whisper English normalization. This is a simulation of selection
on an existing test panel, **not a distillation training result**.

| Native-language label | Clips | Rejected at >10% teacher WER |
|---|---:|---:|
| Nepali | 218 | 74 / 218 = 33.94% |
| Bodo | 139 | 43 / 139 = 30.94% |
| All non-Bodo labels pooled | 2,361 | 525 / 2,361 = 22.24% |
| Hindi | 93 | 14 / 93 = 15.05% |
| Tamil | 151 | 18 / 151 = 11.92% |

Across all 568 rejected clips, 4,523 of 5,554 reference words align correctly
with the teacher output (81.44%). For rejected Bodo clips the corresponding
figure is 304/379 (80.21%). Alignment matches are not a human verification of
audio or a proof that their token gradients are correct. All 19 native-language
results and both normalizers are retained, not only the examples above.

**Attempts to break the hypothesis matter.** Standardizing the non-Bodo
rejection rate to Bodo's reference-length bins gives 22.47%, leaving an 8.46-point
descriptive difference. However, only 19 identical minimally normalized reference
strings occur across Bodo and other groups, covering 22 Bodo and 151 other clips.
These are not verified shared elicitation prompts. Equal-weighting the reference
strings gives rejection rates of 15.79% versus 13.60% under English normalization;
under Minimal normalization they are 15.79% versus 18.93%, reversing direction.
This small, selected subset cannot resolve the question, but it sharply weakens
any claim that the aggregate difference is already an accent-specific effect.
Content, task and speaker coverage are central to the next experiment.

Outputs: [selection results](../research/teacher_coverage/results/selection.csv),
[controls and split audit](../research/teacher_coverage/results/confounders.json),
[diagnostic protocol and command](../research/teacher_coverage/README.md).
No new paid compute was used.

## Experiment that can earn a paper

### 1. Establish the mechanism with controlled training

Start with one frozen teacher and one student initialization in a **fully labeled
mechanism experiment**. The same allowed reference annotations are available to
every arm. Freeze speaker/content splits,
normalization and decoding before training. Use a real training corpus with
verified speaker IDs; do not train on Svarah to improve a Svarah headline.

The first experiment separates three interventions:

* **Amount:** full data versus random removal, with the same verified reference
  targets. Preserve native-language proportions and length distribution when
  sampling the smaller pool.
* **Selection and within-group coverage:** compare teacher-filtered selection
  with random selection matched to its retained audio duration, native-language
  counts and reference-length bins. First train both on verified reference
  targets, so selection does not simultaneously change teacher-label noise.
  Add quota-restored filtered selection and match its corresponding random
  control. Restoring group proportions tests whether missing within-group input
  support remains consequential.
* **Label fidelity:** cross those fixed input selections with teacher labels,
  verified references, and conditional teacher/reference targets. Compare target
  sources on identical audio and model settings. This factorial design separates
  discarded input support from erroneous supervision and their interaction.

Train with both a fixed number of updates and an equal-exposure sensitivity;
record unique audio, repeated exposures, reference usage and teacher FLOPs.
Filtered and full datasets cannot simultaneously have identical unique support;
that is the treatment, not something to disguise. Repeat the decisive contrasts
with at least three independent training seeds before interpreting an advantage.

If filtration does not cause an independent-speaker performance loss after these
controls, stop pursuing this mechanism. Do not turn the descriptive table into
a causal conclusion.

### 2. Build an intervention only where the baselines fail

The candidate design preserves useful supervision within hard utterances while
avoiding incorrect teacher targets. Begin with known conditional KD and token
masking. Measure whether these already recover the discarded information.

The difficult extension is student-generated histories: an insertion or deletion
changes where the learner is relative to the reference. Blindly applying the
reference's token at the same array index is invalid. A proposed on-policy method
must define how learner prefixes align to supported reference/audio spans,
handle insertions, omissions, EOS and uncertain alignments, and demonstrate a
benefit over the same method on reference/teacher histories. CTC-based mixed
histories are also prior art, as covered in the frontier review.

This is a candidate technical problem, **not an already invented new algorithm**.
Use an oracle reference-derived gate first to measure attainable benefit, then
replace it with a learned correctness estimator. Compare detector precision and
recall at matched coverage for every accent. Teacher confidence and agreement
between models are estimates, not truth. Charge any additional annotation or
teacher computation to the method. Any later annotation-efficiency claim must
use a fixed labeled seed and a separate unlabeled pool; the fully labeled
mechanism experiment cannot establish such a claim.

Required comparisons are ordinary supervised fine-tuning, accent-balanced
fine-tuning, unfiltered and globally filtered KD, quota-matched filtering,
conditional KD, subgroup-sensitive teacher mixing, token masking, and Group DRO.
Add partial-label CTC/STC and a modern CTC/attention method if the proposed
mechanism uses those components. Include off-policy/student-rollout ablations
only after simpler supervision controls are working. The present repo's
teacher-rollout reverse-KL arm is a useful controlled baseline; it is not an
implementation of the complete Distil-Whisper recipe.

Use a second teacher and a second student capacity after the mechanism appears
on the first pair. Compare identical adaptation capacity across arms, and check
whether freezing the encoder conceals the acoustic learning needed for accents.
Do not train a large architecture matrix before the intervention has a signal.

### 3. Demand generalization and useful improvement

The primary outcome is absolute recognition error for a prespecified set of
supported native-language groups on unseen speakers. Report each group's change,
macro WER, micro WER and the worst-group error. A smaller gap achieved by making
the best group worse is a failure. Use paired speaker-level comparisons where
speaker identity is verified, with uncertainty that also acknowledges training
seeds. Group selection and model selection happen on development data only.

Include matched prompts spoken by different speakers, a separate unseen-content
test, spontaneous speech, and names/numbers alongside ordinary WER. Manually
adjudicate a stratified sample of the errors on which the mechanism depends.
Test on a second corpus without selecting checkpoints from its results. Existing
Svarah observations remain diagnostic evidence; a repeatedly inspected benchmark
does not become untouched again by assigning a new random split.

An example *proposed target*, to freeze after an independent pilot but before
the confirmation experiment, is at least 10% relative reduction in the
prespecified worst-group WER versus the strongest tuned same-size baseline,
with no more than 0.5 percentage points aggregate degradation. These are research
design targets, not achieved numbers, universal venue rules, or a substitute for
uncertainty. Reassess practical effect size and speaker sample size from the
pilot rather than treating three seeds or an arbitrary clip count as sufficient.

## Data plan and genuine constraints

Compute is no longer the organizing constraint. Speaker coverage, trustworthy
labels and a test set that has not driven the method are more consequential.

* [L2-ARCTIC](https://psi.engr.tamu.edu/l2-arctic-corpus/) provides known speakers
  and repeated prompts for an initial mechanism study: its current release has
  24 speakers across six L1s, with four Hindi speakers and no basis for broad
  Indian-L1 claims. Its small speaker count makes it a pilot, not sufficient
  confirmation. Keep held-out speakers and held-out prompt IDs separate, with
  an additional shared-prompt diagnostic clearly labeled.
* [SPIRE-SIES](https://arxiv.org/abs/2312.00698) reports 170.83 hours and 1,607
  speakers, with 23 transcribed hours in the paper's release. Its speaker IDs
  and image prompts could support speaker/content splits after checking the
  actual files and annotation provenance. Bodo, Nepali and Assamese coverage
  cannot be assumed from its named categories or its undifferentiated “Others”
  category. The paper also describes model-based curation; audit that upstream
  selection before studying a downstream filter.
* [EdAcc](https://groups.inf.ed.ac.uk/edacc/) supplies conversational speech and
  speaker linguistic profiles for external generalization, subject to verifying
  actual release and subgroup support. Retain its designated test data for final
  assessment; broad accent diversity does not imply enough Bodo speakers.
* A serious claim about currently poorly supported Indian native-language
  backgrounds may require new, consented English recordings with explicit
  anonymous speaker IDs. Collect both common content and spontaneous speech,
  balance recording conditions, and hold out people from the outset. Plan
  independent speakers based on pilot variability, not more clips from the same
  five profiles. No participants have been contacted and no collection is
  claimed complete.

The old Svarah OPD pilot cannot substitute for this design. Its loader samples
45 training and 24 evaluation utterances from the public test split without
speaker separation. A reconstruction from its saved seed and pinned row order
matches the saved example IDs; 14/24 evaluation clips share an unverified
demographic profile with training, and one shares normalized reference text.
That does not prove speaker identity, but the code does not enforce independence.
The on-policy arm's WER is also slightly worse than the off-policy arm in both
small pilots. Lower teacher–student KL is agreement, not independently measured
confidence calibration or a demonstrated fairness improvement.

## Execution order and decision

Completed now: candid manuscript review, primary-source novelty comparison,
teacher-filter diagnostic, length/content counterchecks, and reconstruction of
the old pilot's split weaknesses. The manuscript is marked as not recommended
for submission. All existing prediction artifacts remain intact.

Next concrete deliverable: verified training/development/test manifests and a
controlled pilot comparing the interventions above, followed by an evidence-based
decision on whether a new local/on-policy mechanism is warranted. Measure
throughput on the actual corpus before specifying a large GPU budget. This is
research over days to weeks; independent data collection can take longer. More
compute can shorten training but cannot manufacture the missing observations.

I would support spending effort on this falsifiable experiment. I would not yet
claim that its hypothesis is established, its proposed extension is novel, or
that it will outperform the strongest baseline. Those are the results the next
research stage has to earn.
