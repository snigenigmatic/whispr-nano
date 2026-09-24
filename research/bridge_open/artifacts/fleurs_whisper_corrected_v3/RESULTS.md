# Corrected Whisper comparison

All 2,322 recordings per model are covered. Short clips retain the audited HF beam-5 route; every >30-second clip uses pinned native transcription. This is a post-result correction, not a held-out improvement.

| Model | Language | n | Short WER | Long WER (n) | Pooled routed WER | Pooled CER |
|---|---|---:|---:|---:|---:|---:|
| Turbo | hi | 418 | 31.02% | 78.87% (1) | 31.35% | 13.62% |
| Turbo | bn | 920 | 85.79% | 89.03% (9) | 85.86% | 35.61% |
| Turbo | as | 984 | 135.02% | 100.00% (4) | 134.71% | 116.35% |
| Large-v3 | hi | 418 | 28.04% | 53.52% (1) | 28.21% | 11.72% |
| Large-v3 | bn | 920 | 74.39% | 88.28% (9) | 74.70% | 27.86% |
| Large-v3 | as | 984 | 113.40% | 102.89% (4) | 113.30% | 108.23% |

The pooled column describes the declared duration-based routing rule. It is not a homogeneous decoder condition. Long-stratum sample counts are small. Preserve the previous fixed-split results as sensitivity analysis; do not substitute only favorable cases.

Known short-form caps remain in the denominator. Native EOS/cap evidence is unavailable; detailed decode attempts and timestamp flags remain in the source artifacts. Sentence-cluster bootstrap intervals in comparison.json do not support speaker/population fairness claims.

No human semantic transcript review was performed. Input/source fidelity, model state, native parity samples and metric arithmetic were verified.
