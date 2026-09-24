# Pseudo2Real and Pseudo2Real-SC: implementation note

## Primary sources

- Yi-Cheng Lin et al., “Pseudo2Real: Task Arithmetic for Pseudo-Label Correction in Automatic Speech Recognition,” *Findings of ACL 2026*, pp. 1160–1175 ([paper](https://aclanthology.org/2026.findings-acl.59.pdf), [record](https://aclanthology.org/2026.findings-acl.59/)).
- The [author’s publication page](https://dlion168.github.io/bibliography/lin2025pseudoreal/) links the [official repository](https://github.com/nukkk97/pseudo2real). Code observations below refer to commit [`f1fcfffd5600984257e5c9ce27b3b744da05d5b7`](https://github.com/nukkk97/pseudo2real/tree/f1fcfffd5600984257e5c9ce27b3b744da05d5b7).
- The author-released [derived AfriSpeech dataset](https://huggingface.co/datasets/dlion168/afrispeech200_syn2real/tree/e05d04e8741ae58c8b6a2223997f0827a5e4a89c) was inspected at revision `e05d04e8741ae58c8b6a2223997f0827a5e4a89c`; it contains gold transcripts and Whisper tiny/base/small/medium/large-v2 pseudo-label columns, but no pseudo-label generation recipe.

## Exact published methods

Let the labeled source data be \(D_s=(S_s,T_s)\), unlabeled target speech be \(S_t\), and a teacher’s transcripts be \(\hat T_s\) and \(\hat T_t\). All student endpoints must start from the same pretrained student parameters \(\theta_{pre}\).

**Pseudo2Real.** Fine-tune one source student on real labels and a second source student on teacher labels, then subtract the latter from the former:

\[
\tau = \theta^{real}_s-\theta^{pseudo}_s.
\]

Fine-tune another copy of the same student on target pseudo-labels, then apply the source correction:

\[
\theta^{corrected}_t=\theta^{pseudo}_t+\lambda\tau.
\]

The paper applies arithmetic to **all model parameters**, rather than a selected layer set ([paper, §3.2 and §4](https://aclanthology.org/2026.findings-acl.59.pdf)). The repository implements the same sign and order by subtracting the pseudo source parameters from the real source parameters and adding the result to the target model ([`test_afris.py`, lines 20–32](https://github.com/nukkk97/pseudo2real/blob/f1fcfffd5600984257e5c9ce27b3b744da05d5b7/test_afris.py#L20-L32)). Teacher and student may have different capacities; the teacher only produces text, while all three arithmetic endpoints have the same student architecture.

**Pseudo2Real-SC.** Extract one ECAPA-TDNN representation **per source utterance** using [`speechbrain/spkrec-ecapa-voxceleb`](https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb), partition those representations with k-means, and train a real-label and pseudo-label student from \(\theta_{pre}\) for every cluster \(c\):

\[
\tau_c=\theta^{real}_{s,c}-\theta^{pseudo}_{s,c},\qquad
\theta^{corrected}_t=\theta^{pseudo}_t+\frac{\lambda}{C}\sum_{c=1}^{C}\tau_c.
\]

The released assignments are keyed by `audio_id`, confirming utterance-level membership; the paper does not average utterances into one representation per known speaker or require speaker-disjoint clusters. Thus published SC is a **uniform average of source-cluster vectors**. It does not select or weight vectors using a target speaker. Target-conditioned selection or mixing is explicitly listed as future work ([paper, §6 and §8](https://aclanthology.org/2026.findings-acl.59.pdf)). This distinction makes a similarity-weighted, target-conditioned correction bank a real extension rather than a reproduction of SC.

The published clustering uses scikit-learn k-means with k-means++ initialization, at most 300 iterations, tolerance \(10^{-4}\), and Lloyd’s algorithm. The paper calls this the default configuration but gives no random seed or `n_init`; its ablation uses \(C\in\{1,2,4,8\}\), and the main SC comparison uses \(C=8\) ([paper, Appendix E and §6](https://aclanthology.org/2026.findings-acl.59.pdf)). Each cluster costs two complete source fine-tunings. On LARGE→SMALL, mean WER over the five fold-1 accents changes 45.36 → 44.34 → 43.68 → 41.94 for 1, 2, 4, and 8 clusters. With eight clusters, SC improves over global Pseudo2Real by 4.0% relative for a MEDIUM teacher and 6.1% for a LARGE teacher, but slightly worsens Igbo in both cases.

## Data and evaluation protocol

The study keeps the official AfriSpeech-200 train/dev/test splits and selects ten English accent groups with all three splits. It performs two-way cross-fold adaptation:

- fold 1: Igbo, Swahili, Hausa, Zulu, Twi — 21,621 train, 1,141 dev, 1,305 test utterances;
- fold 2: Yoruba, Ijaw, Afrikaans, Idoma, Setswana — 21,670 train, 756 dev, 939 test utterances.

One fold supplies labeled source speech for correction-vector construction while the other supplies target speech for pseudo-labeling; roles are then swapped. Per-accent counts appear in [Table 7](https://aclanthology.org/2026.findings-acl.59.pdf). The released [`train_afris_target_domain.sh`](https://github.com/nukkk97/pseudo2real/blob/f1fcfffd5600984257e5c9ce27b3b744da05d5b7/sh_scripts/train_afris_target_domain.sh) trains a separate target-pseudo endpoint for each accent, while the source endpoint pools its five source accents. A target model pooled across several target accents is therefore a defensible variant, but not the released experiment’s exact target construction.

The paper reports test WER per accent and macro-averages the accent WERs in its cross-fold tables. The principal baselines are pretrained student, source-real SFT, target-pseudo SFT, confidence-filtered target-pseudo SFT, T5 error correction, EMA, and a target-real topline. Confidence filtering retains hypotheses above a development-tuned average word-level log-probability threshold; the public repository supplies 25/50/75-percent filter lists but not the script that computes their confidences.

Published student training fully fine-tunes Whisper TINY or SMALL with AdamW, learning rate \(3\times10^{-5}\), weight decay 0.1, batch size 16, FP16, 500 linear-warmup steps, up to 40,000 updates, evaluation every 50 updates, greedy decoding, and best-checkpoint selection by development WER. The paper tunes \(\lambda\) only on held-out **source-domain development data**, searching \(\{0.1,0.2,\ldots,1.0\}\). Its ablation finds a U-shaped response, usually best around 0.2–0.3 and unstable when too large ([paper, §4 and §5.3](https://aclanthology.org/2026.findings-acl.59.pdf)). The exact teacher pseudo-label generation procedure, model revisions, decoding prompt, random seeds, checkpoint hashes, and multi-seed uncertainty are not specified in the paper.

The released environment pins Python 3.9.18, PyTorch 2.0.1, Transformers 4.37.2, Datasets 2.18.0, Evaluate 0.4.1, JiWER 3.0.3, and scikit-learn 1.4.0 ([`environment.yml`](https://github.com/nukkk97/pseudo2real/blob/f1fcfffd5600984257e5c9ce27b3b744da05d5b7/environment.yml)). The tiny YAML matches most paper settings, including LR \(3\times10^{-5}\), weight decay 0.1, batch 16, FP16, 500 warmup steps, generation length 225, and eval/save every 50 steps ([`whisper_tiny.yaml`](https://github.com/nukkk97/pseudo2real/blob/f1fcfffd5600984257e5c9ce27b3b744da05d5b7/configs/whisper_tiny.yaml)).

## Reproducibility hazards in the official repository

The paper should be treated as the algorithmic authority and the repository as partial implementation evidence. The released tree does not include pseudo-label generation, ECAPA extraction/k-means generation, correction-vector averaging, trained vectors, or checkpoints. It also contains conflicts that prevent literal reproduction:

1. `train_afris_custom.py` hard-overrides the YAML’s 40,000 updates with 70,000 and sets early-stopping patience 20 ([lines 244–247](https://github.com/nukkk97/pseudo2real/blob/f1fcfffd5600984257e5c9ce27b3b744da05d5b7/train_afris_custom.py#L244-L247), [callback](https://github.com/nukkk97/pseudo2real/blob/f1fcfffd5600984257e5c9ce27b3b744da05d5b7/train_afris_custom.py#L366-L374)). The SMALL YAML uses LR \(10^{-5}\) and omits weight decay, unlike the paper ([config](https://github.com/nukkk97/pseudo2real/blob/f1fcfffd5600984257e5c9ce27b3b744da05d5b7/configs/whisper_small.yaml)).
2. In the `syn == "False"` path, the script filters by real-transcript length but passes `args.synth_text` to preprocessing, so the purported real endpoint is actually tokenized from the pseudo-label column ([lines 308–321](https://github.com/nukkk97/pseudo2real/blob/f1fcfffd5600984257e5c9ce27b3b744da05d5b7/train_afris_custom.py#L308-L321)). Do not copy this path.
3. The parser allows clusters 1, 4, or 8, although the paper and supplied shell script use 2; the training code always reads `fold{n}_clusters.csv` rather than the supplied `ecapa_kmeans2/4` files ([parser](https://github.com/nukkk97/pseudo2real/blob/f1fcfffd5600984257e5c9ce27b3b744da05d5b7/train_afris_custom.py#L139-L144), [assignment loading](https://github.com/nukkk97/pseudo2real/blob/f1fcfffd5600984257e5c9ce27b3b744da05d5b7/train_afris_custom.py#L201-L219)). The released cluster assignments are useful evidence, but the absent generation script and missing seed make regeneration non-exact.
4. The precomputed-vector branch adds the vector once explicitly and then calls `merge`, which adds it a second time ([`test_afris.py`, lines 65–80](https://github.com/nukkk97/pseudo2real/blob/f1fcfffd5600984257e5c9ce27b3b744da05d5b7/test_afris.py#L65-L80)). Evaluation requests `task="translate"` although training uses `task="transcribe"`, and normalizes predictions but computes final WER against the raw transcript field ([lines 8–18 and 82–86](https://github.com/nukkk97/pseudo2real/blob/f1fcfffd5600984257e5c9ce27b3b744da05d5b7/test_afris.py#L8-L18)).
5. The paper says source-dev tuning for \(\lambda\), but the distributed [`test_afris.sh`](https://github.com/nukkk97/pseudo2real/blob/f1fcfffd5600984257e5c9ce27b3b744da05d5b7/sh_scripts/test_afris.sh) sweeps target models on same-accent target dev labels through [`test_afris_val.py`](https://github.com/nukkk97/pseudo2real/blob/f1fcfffd5600984257e5c9ce27b3b744da05d5b7/test_afris_val.py#L46-L50). Follow the paper’s source-only rule to preserve the unlabeled-target claim.

## Actionable protocol for the SPIRE study

1. Keep the global baseline exactly \(\theta^{pseudo}_t+\lambda(\theta^{real}_s-\theta^{pseudo}_s)\), with compatible named-parameter keys/shapes and target buffers unchanged. Record the hash of the identical initialization used for both source endpoints and the target endpoint.
2. Replace the fixed \(\lambda=1\) assumption with a preregistered source-dev grid. Include \(\lambda=0\) as the uncorrected control and search at least 0.1–1.0 as published. Because the paper finds 0.2–0.3 most stable, a result only at 1.0 is not an adequate test.
3. Implement published Pseudo2Real-SC as a uniform-average baseline before claiming a target-conditioned bank. Freeze the ECAPA revision, preprocessing, cluster count, `random_state`, `n_init`, and membership manifest. With \(C\) clusters this requires \(2C\) matched source endpoints.
4. Treat similarity weighting as the proposed contribution: estimate source cluster centroids, compute target ECAPA representations without labels, preregister the similarity-to-weight mapping, and compare it against both global Pseudo2Real and uniform SC. Published SC provides no target-conditioned routing.
5. Tune all method choices and \(\lambda\) on source development only. Keep SPIRE validation/test references sealed until the complete comparison is frozen, then evaluate every frozen system on the same one-time target split.
6. Use identical data exposure, optimizer schedule, precision, normalization, and decoding for every real/pseudo endpoint. Save predictions and insertion/deletion/substitution counts: the published qualitative mechanism is largely reduced insertions/hallucinations, so aggregate WER alone is insufficient.
7. Describe the current 600-update run as a budgeted feasibility protocol, not an exact replication of the published 40,000-update recipe. If target training pools the six SPIRE accents, disclose that departure and retain accent-stratified metrics.
