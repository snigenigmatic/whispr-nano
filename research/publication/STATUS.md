# Publication study status — 16 September 2026

## Decision

This is not yet a submission-ready paper. It is now a controlled study with
real H100 evidence, a sealed target test, falsifiable baselines, and one
uncompleted experiment. The central target-domain claim cannot be made until
the two missing cluster-0 endpoints, the correction-bank scale selection, and
the one-time SPIRE evaluation are complete.

## Accepted evidence

All WER values below are on the frozen 974-utterance L2-ARCTIC development
set. They diagnose source retention and select correction scale; they are not
SPIRE target-domain results.

| Arm | English macro WER | English micro WER | Status |
|---|---:|---:|---|
| Pretrained Whisper tiny | 29.159% | 28.818% | Frozen common baseline |
| Source-real SFT | 24.773% | 24.521% | Complete |
| Source-pseudo SFT | 24.723% | 24.498% | Complete |
| Target-pseudo SFT | 31.498% | 31.187% | Complete |
| Target confidence top-50% | 35.049% | 34.649% | Complete; negative result |
| Global Pseudo2Real, lambda 1 | 32.395% | 32.055% | Complete; negative result |
| Global Pseudo2Real, source-dev lambda 0.4 | **30.912%** | **30.645%** | Complete; improves target-pseudo by 0.586 macro-WER points |
| C=2 source-real endpoint, cluster 1 | 27.179% | 26.878% | Complete |
| C=2 source-pseudo endpoint, cluster 1 | 27.068% | 26.810% | Complete |

The 4,800-example SPIRE training manifest covers six accents equally, 799
speakers, and 13.24 hours. It has zero speaker overlap with the 198 sealed
validation speakers. Whisper large-v3 pseudo-labels cover all 4,800 examples.
The blind 120-clip audit includes exact high/low confidence strata. Mean
log-probability filtering selected both detected repetition-loop failures and
performed substantially worse than full target-pseudo SFT on source dev.

Pinned ECAPA embeddings now cover all 9,495 source and 4,800 target training
utterances. K-means C=2 produced source clusters of 7,912 and 1,583 utterances.
Equal-weight target-speaker routing assigned 688 and 111 of 799 target
speakers, freezing the proposed bank weights to 0.861076 and 0.138924. The
assignment SHA-256 is
`6a504fc5e7cce90dd2f11bb862f395284d1e182413cb79640e5e63efa3e12ce4`.

## Rejected evidence

The last H100 batch completed both cluster-1 endpoints. Both cluster-0 arms
reached update 347 of 600 before the enforced timeout, so their partial states
are rejected. Their immutable histories are retained under
`results/sc_c2/rejected_batch_v3/`; no partial checkpoint enters a comparison.

The live ledger records US$11.975644 of measured or conservatively bounded
cost against the user-authorized US$12 hard cap. GPU dispatch is frozen. The
large gross reservation total is historical reservation bookkeeping, not the
amount spent; the guard counts measured terminal cost and unresolved
reservations.

## Next executable gate

1. Add credit or explicitly raise the hard cap.
2. Train only `source_real_c0` and `source_pseudo_c0`; do not repeat the two
   accepted cluster-1 endpoints.
3. Run the fixed source-dev lambda grid for the published uniform P2R-SC bank
   and the target-speaker-conditioned bank.
4. Complete the 120-clip human audit and freeze the method.
5. Open the 198-speaker SPIRE set once and report every frozen comparator,
   per-accent WER, macro WER, micro WER, and paired uncertainty.

The proposed contribution is the target-speaker-conditioned correction bank.
It remains a hypothesis until step 5. The negative confidence-filtering and
global-correction results should remain in the paper because they clarify why
speaker-conditioned transfer is needed.
