# Decision memo template (Days 11–14)

Fill this after the comparison is **frozen** and SPIRE-SIES / L2-ARCTIC have been scored once. Do not retune.

## Central finding

One paragraph. What transferred, what failed, on which L1s. No novelty language unless the finding survived independent speakers.

## Strongest baseline

Name the arm (ordinary pseudo-label SFT / uDistil-Whisper filter / Pseudo2Real / privileged reference SFT). Report gate-eval and held-out numbers separately. Privileged reference SFT is a diagnostic, not the winner of a fair comparison.

## Surviving limitation

The failure that still exists after the strongest baseline. If none, write that an existing method closed the gap on this data, and **stop the paper**.

## Reproducible evidence

- Design file hash / commit:
- Primary checkpoint (must be 2,400 unless this memo is closing the mechanism at Days 1–3):
- Primary normalizer / alternate normalizer:
- Seeds:
- Macro WER table (six L1s; Bodo marked descriptive):
- Speaker-resampled 95% interval (training seeds listed separately):
- Listening-set labels (n=120): teacher_mistake / reference_mistake / both_wrong / normalization / genuine_variation / other

## Paper / no-paper

Continue toward a paper only if **both** are true:

1. The finding survives independent speakers (SPIRE-SIES cohort and L2-ARCTIC, frozen).
2. It is more than applying Pseudo2Real or uDistil-Whisper to another Indian-English set.

Otherwise keep this memo as the technical report and change direction.

## Meetings

If a meeting with Dr. Das (or the team) changed a decision recorded here, note the date and the decision. Do not retrofit the frozen design.
