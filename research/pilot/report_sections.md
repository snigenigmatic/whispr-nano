# Controlled pilot of teacher-based training-data selection

## Motivation and question

A teacher transcript can contain several errors while the recording still
contains useful supervision for a smaller recognizer. Rejecting the entire
recording changes which speech the student encounters, as well as the quality
of any retained pseudo-labels. The two effects should be separated before
proposing a more elaborate distillation objective.

This pilot asks whether a teacher-WER filter harms adaptation to new speakers
and new prompts **through its selection of inputs**, compared with a random
control matched to the filtered pool. Every arm uses the same corpus-provided
reference transcripts as its training targets. Teacher hypotheses determine
selection only; they are never substituted for the reference targets, and no
teacher-distribution loss is applied. Thus the experiment isolates selection
from the transfer of erroneous teacher labels. It does not test practical
knowledge distillation or annotation efficiency.

This question follows an established tradeoff. Distil-Whisper explicitly
examines the loss of coverage caused by stricter pseudo-label filtering and
normalizes transcripts before applying its WER threshold. Its findings motivate
the control experiment; they do not establish the outcome of our different
teacher, student, corpus and training schedule. [Distil-Whisper, §§4.2 and 9.1](https://arxiv.org/pdf/2311.00430).

## Data and partitions

We use the scripted L2-ARCTIC v5.0 release, which contains 26,867 recordings from
24 identified speakers across Arabic, Mandarin, Hindi, Korean, Spanish and
Vietnamese, with four speakers per L1. The release has 3,599 recordings with
manual pronunciation annotations. These annotations do not independently
certify every word in the full corpus's orthographic references.
[Official corpus documentation](https://psi.engr.tamu.edu/l2-arctic-corpus-docs/),
[official speaker mapping](https://psi.engr.tamu.edu/l2-arctic-corpus/).

Within each L1, a deterministic hash ordering with seed `20260909` assigns two
speakers to training, one to development and one to a sealed test. Prompt
families are independently divided into 792 training, 169 development and 171
test families. A family joins all renditions of one prompt and any prompt IDs
sharing an exact minimally normalized reference. Only recordings whose speaker
and prompt partitions agree enter the primary evaluation.

| Primary partition | Recordings | Speakers | L1 coverage |
|---|---:|---:|---|
| Training pool | 9,495 | 12 | Two speakers per L1 |
| Development | 974 | 6 | One speaker per L1 |
| Sealed test | 1,026 | 6 | One speaker per L1; 171 recordings per L1 |

Speaker IDs, prompt IDs and normalized reference strings have zero pairwise
overlap between these primary partitions. Shared-prompt diagnostics are stored
separately and do not enter this pilot's primary evaluation. Test and
`test_shared` inference remain prohibited. The raw mirror is pinned to
`cc02fd37197966e5de11ad737f7affd577c61992`; archive hashes and per-recording metadata
provide the provenance record. Counts and structure match the corpus owners'
documentation, but the complete mirror has not been compared byte-for-byte with
an independently acquired official archive. [Pinned raw mirror](https://huggingface.co/datasets/chikingsley/l2-arctic-release-v5.0/tree/cc02fd37197966e5de11ad737f7affd577c61992).

## Intervention and controls

A frozen Whisper-large-v3 teacher, revision
`06f233fe06e710322aca913c1bc4249a0d71fce1`, transcribes the training pool. English
transcription uses greedy decoding, no sampling, one beam and at most 224 new
tokens. The filter accepts an utterance when its English-normalized teacher
WER is at most 10%, including the boundary. References and hypotheses receive
the same frozen English normalizer.

The registered arms are:

- **`full_gold`:** sample from the complete 9,495-recording training pool.
- **`filtered10_gold`:** sample from recordings accepted by the teacher filter.
- **`random_matched_gold`:** sample from a random control that matches the
  filtered pool's counts within each speaker and reference-length bin, with
  approximate matching of total audio duration in each stratum.

Reference-length bins are fewer than 5, 5–9, 10–19, and at least 20 normalized
words. The random control starts from a random subset and uses duration-only
swaps; teacher errors do not guide those swaps. When the greedy search stalls in a small stratum, the corrected implementation samples uniformly from exhaustively enumerated subsets meeting the unchanged tolerance. The preflight repair was recorded before any proper training outcome. Its target tolerance is the
larger of 0.05 seconds and 0.5% of the filtered stratum's duration. Achieved
balance must be reported rather than assumed. Both accepted and rejected
recordings are eligible for this control. The selection seed is fixed at
`20260909`; the three training seeds do not create three independent random
selection pools.

The main comparison is filtered versus matched-random training. Their eligible
pools have equal counts within each speaker and reference-length bin while the
selected recordings differ. Sampling only part of each pool does not guarantee
equal realized length-bin counts, audio duration or supervised tokens. Because
the control inherits the filtered pool's speaker and L1 allocations, this
contrast tests selection within those allocations; it does not test harm caused
by changing the overall representation of an L1. The full-pool arm supplies
context for ordinary supervised adaptation. The suffix `gold` is an experiment
identifier for the supplied reference targets; it does not assert that this
project manually adjudicated every transcript.

## Student training and measurement

All nine registered runs start from the same Whisper-tiny weights, revision
`169d4a4341b33bc18d8881c4b69c2e104e1cc0af`, with three paired seeds: 0, 1 and 2.
The encoder and decoder are both trainable. Each run uses 600 AdamW updates,
batch size 8, no gradient accumulation, constant learning rate `1e-5`, weight
decay `0.01`, and gradient clipping at norm 1.0. Reference content tokens and
the first EOS contribute to cross-entropy; forced decoder-prefix positions and
padding are masked from the loss.

Each run presents exactly 4,800 training examples. Sampling shuffles each
selected pool without replacement and reshuffles when exhausted. Python,
NumPy and Torch seeds are fixed; evaluation does not advance the training RNG
stream. Every selected pool exceeds 4,800 recordings, so all completed runs see
4,800 unique examples without repetition. This controls update and unique-example
counts, but does not equalize realized audio seconds or supervised tokens. The
full-pool arm does not encounter all 9,495 eligible recordings; it changes the
eligible input composition rather than the training-example budget. Exposure
logs are therefore part of the evidence.

Development evaluation occurs before training and after update 600, using all
974 primary development recordings, batch size 8, greedy English transcription,
one beam and a 224-token generation limit. There is no intermediate checkpoint
selection. The primary score is the unweighted mean of the six L1 WERs; each
L1 WER aggregates word errors and reference words within that L1. Micro WER,
worst-L1 WER, every L1's absolute error, and every training seed are also
reported.

The English normalizer is frozen as local source with a pinned spelling map.
The evaluation sensitivity uses Unicode NFKC normalization, lowercase,
punctuation-to-space conversion and collapsed whitespace. This evaluation
convention is distinct from the apostrophe-preserving minimal normalization
used to construct content families; the evaluator also verifies no cross-split
reference overlap under the evaluation convention.

The registered primary contrast is
`filtered10_gold minus random_matched_gold` in final English L1-macro WER.
A positive value means the matched random control is better. The pilot's
predeclared signal criterion requires **all nine valid runs**, a mean contrast
of at least **0.5 percentage points**, a strictly positive contrast in at least
two of the three seeds, and a nonnegative mean contrast under the minimal
normalization sensitivity. This criterion authorizes consideration of further
replication; it is neither a significance test nor a publication standard.
The schedule was frozen after an infrastructure smoke test using 16 training
and 12 development clips, and before any full-development training outcome.
Those two smoke evaluations are disclosed development accesses.

## Evidence standard and interpretation

A separate evaluator checks the exact development IDs, speaker identities and
references; recomputes word-error counts and aggregates from raw predictions;
and compares stored metrics, initialization fingerprints, code and normalizer
hashes, decoder configuration, updates and exposure sequences. Missing required
runs make the result incomplete. Operational smoke tests do not count toward
the nine-run experiment.

AutoResearch's persistent work units, execution receipts and separate criticism
informed the evidence workflow. Our review distinguishes workflow closure from
scientific adequacy: finishing a run queue does not establish the hypothesis.
Reviewer scores do not select models or override the registered numerical
criterion. [Inspected workflow completion checks](https://github.com/EvoMap/AutoResearch/blob/6b85443f9e3d33d357c86c8bb826f9ea03c780b3/ar-runtime/scripts/ar-workflow-engine.py#L3655-L3687).

## Limits and the novelty bar

There is only one development speaker per L1. Seed variation measures training
randomness, not the uncertainty from drawing new speakers. The hash split is
not gender-stratified, public ARCTIC prompt exposure during model pretraining
is unknown, and exact text separation does not guarantee semantic novelty.
References have not been independently audio-adjudicated by this project. The
release already excludes some recordings for quality, so the intervention
operates within an upstream-selected corpus. The fixed short schedule does not
establish convergence, and an equal-audio or equal-token exposure sensitivity
is still needed. Four Hindi speakers do not support claims about the breadth
of Indian English accents.

Even a positive, valid pilot would support a harmful effect of this filter's
**selection policy** on reference-supervised adaptation under the specified
conditions. It would not identify lost phonetic coverage as the cause: lexical
composition, recording characteristics and realized exposure can also differ.
It would not show that discarding noisy pseudo-labels is generally harmful, or
that a new distillation method improves the accuracy–coverage tradeoff. Conditional
teacher/reference supervision already exists in ASR; subgroup-dependent
teacher mixing and token masking also have direct precedents.
[Conditional Teacher–Student Learning](https://www.microsoft.com/en-us/research/uploads/prod/2019/09/ICASSP2019_Conditional_Teacher_Student_Learning.pdf),
[Teacher's Pet](https://arxiv.org/pdf/2106.10494),
[pseudo-label masking in ASR](https://www.isca-archive.org/interspeech_2022/du22_interspeech.pdf).

The next empirical step, if justified, is to cross these fixed input selections
with teacher labels, reference labels and conditional or locally masked
targets, while accounting for annotation access and computation. An improvement
must survive stronger existing baselines, new speakers, another corpus and
appropriate exposure controls. On-policy ASR distillation already has recent
precedents; a student-history extension must solve alignment after insertion
or deletion and demonstrate added value over simpler supervision.
[Ark-ASR, preprint](https://arxiv.org/html/2605.28139v1),
[Star Temporal Classification](https://arxiv.org/pdf/2201.12208).

Failure to meet the registered signal criterion leaves this within-allocation
selection hypothesis unsupported by the pilot. It does not establish equivalence
or rule out harm from reduced L1 representation, different filter thresholds or
longer training. The result should redirect the research rather than motivate a
more favorable metric, selective seed reporting, or a new acronym for an
unvalidated remedy.
