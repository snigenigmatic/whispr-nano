# Publication positioning check — 2026-09-23

[Script collapse in multilingual ASR](https://arxiv.org/abs/2604.08786) already studies wrong-script output on FLEURS across model sizes and languages. It uses forced language tokens and greedy decoding, and explicitly excludes alternative decoding strategies. It also cautions that shared scripts cannot establish language correctness. Therefore neither script diagnostics nor discovering wrong-script Whisper output should be claimed as our invention. Our forced-versus-automatic native experiment can supply additional controlled evidence, but a small diagnostic alone does not demonstrate publishable novelty.

[Sequential Adapter Stacking for Cross-Lingual Low-Resource ASR](https://arxiv.org/abs/2609.15758) explicitly treats Assamese as unsupported in the evaluated Whisper setting and evaluates related-language transfer. Token availability must not be confused with demonstrated recognition competence. This paper's model and training conditions differ from our frozen inference benchmark; no direct numerical comparison is warranted.

[Vimarsha](https://arxiv.org/abs/2609.24199) is also directly relevant to Indian-language evaluation and orthographic alternatives. A three-language FLEURS table does not replicate its scope or establish BRIDGE-equivalent robustness.

The current deliverable is a reproducible benchmark and decoder diagnostic with preserved failure evidence. A submission should state these limitations and make its incremental contribution explicit; acceptance is not established by running more models. Human review remains pending.

Retrieval: ORX semantic search followed by full-text inspection; the first two titles/abstracts were independently checked against arXiv on2026-09-23.
