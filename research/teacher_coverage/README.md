# Teacher filtering and accent coverage: diagnostic, not a trained method

Created September 9, 2026 after the first manuscript was reviewed critically.
This is hypothesis generation using already-inspected Svarah predictions. It
does not turn Svarah into training data or an untouched confirmatory test set.

Question: does an utterance-level teacher WER filter discard a different share
of speech across native-language labels, and how much correct reference-word
alignment is discarded with the inaccurate teacher transcripts?

Before computing this diagnostic, fix Whisper-large-v3 as the primary teacher,
Whisper-tiny as the student, and thresholds 10%, 20%, 40%. The 10% threshold is
motivated by Distil-Whisper Section 9.1, not selected to maximize a subgroup
difference. Repeat with the historical Minimal and Whisper English normalizers.
Count utterances, reference words, and audio duration; report every native
language and Bodo versus pooled others. Correctly aligned reference words are
only a text-alignment diagnostic, not verified acoustic/token-level supervision.

Also report teacher/student correctness overlap and an utterance oracle over
Whisper-large-v3 and Qwen. The oracle uses the reference and is not deployable.
It tests complementarity, not the existence of a learned selector.

No p-values are estimated: verified speaker identities remain unavailable.
An observed filter disparity cannot establish that filtering caused an existing
distilled checkpoint's disparity. That needs a controlled training intervention
with the same student initialization, data, labels, decoding and compute.

Run from the repository root:

```bash
.venv/bin/python -m research.teacher_coverage.diagnose
.venv/bin/python -m research.teacher_coverage.check_confounders
```

Source motivating the fixed threshold:
[Distil-Whisper, Section 9.1](https://arxiv.org/pdf/2311.00430).

After observing the first diagnostic, `check_confounders.py` adds descriptive
reference-length standardization and an exact shared-text sensitivity. It also
reconstructs the old Svarah OPD pilot's utterance split. These are explicitly
post hoc checks; they do not establish causal accent effects or speaker identity.
In particular, the small shared-text sensitivity sharply reduces the aggregate
filter disparity and changes direction under Minimal scoring. Keep that
counterevidence alongside the original table.
