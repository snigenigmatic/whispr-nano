# Cloud decoder and repetition study — frozen 13 October 2026

Prepared for C. Kaustubh. This extends the Lahaja investigation; it does not yet establish an Indian-English result or a publication-ready contribution.

## Question and decisions

Does a repetition intervention improve recognition beyond a properly configured decoder, and what does it delete when speech truly repeats? A small WER reduction against a weak decoding recipe is insufficient evidence of a new method. The standard decoder and a stronger checkpoint are required controls.

Before reading the new predictions, freeze these comparisons:

1. Transformers Whisper-small, five beams, versus OpenAI Whisper-small's official transcribe defaults with beam size five at temperature zero and best-of-five temperature fallback. This changes the complete decoder recipe, including timestamp handling; it is not a one-parameter ablation.
2. The already frozen rule collapsing at least five identical adjacent normalized hypothesis words, applied separately to each decoder's output.
3. On the same small-model loop trigger, replace the entire hypothesis with the official small-model decoder's output, or with Whisper Turbo's output. These are prediction-only routing rules. References never determine whether to route a recording. All candidate outputs are precomputed to make the paired audit possible; this is a simulation of selective execution, not a measured production cascade.
4. Full Whisper Turbo as a quality reference. Its larger model size must remain explicit; selective use is not a fully small-model on-device solution.

Primary evidence for any *new* repair gain is the 396-recording fresh-audio Lahaja cohort. For a claimed safe improvement, require a negative paired speaker-bootstrap 95% WER-difference interval against the official small-model baseline, and no increase in aggregate word errors on either five- or eight-repeat stress set. These are necessary pilot checks, not proof of safety or publication novelty. The 405 previously evaluated clips are replication evidence. Report every prespecified condition, including failures.

## Data

- Lahaja revision `d4ffd2ecbdd933e37c917ddcf620eef159ceb3a7`. Exclude all 26 first-shard development speakers. Reuse the pinned 405-recording cohort, then choose the next two available 2–25-second clips per speaker/scenario using the original deterministic ordering: **396 additional clips, with no recording overlap**. These recordings share speakers with prior evaluation, so they are not an independent speaker holdout.
- Speech Commands v0.02 official test archive: ten labelled words, ten deterministically selected distinct speakers per word. Use each original human recording once, five times and eight times, inserting 150 ms of silence between copies. **300 clips from 100 source recordings**. This is a controlled concatenation stress test, not naturally occurring conversational repetition or an Indian-accent dataset. Preserve all source silence; no model-dependent filtering.
- Total: **1,101 recordings; 2.3134 hours of audio** per decoder.
- All input audio is decoded and asserted to be mono 16 kHz, finite float32. Lahaja duration is checked against metadata. Selection manifests include exact float32 audio hashes, transcript references and source identity, retained privately.
- Lahaja subset manifest hash: `18dfa27387197bc02b29740d6be96ea266019823e72b27eb592e5450a84854b6`.
- Commands subset manifest hash: `4d2eca6827649275174c7ea18031a92eff0002e0de702967dbb2ae0a2d4bea37`.
- Combined manifest file hash: `0bd487ed2058439515b8d44255e3169573ed900ada9eceb1c7c59f6fe9929d8c`.
- Commands archive SHA-256: `cc2a00c1147c2254e9be3fa0f779d8c17421dc349b86366567a8edfa9acd51df`.

The earlier Scicom oracle check remains a structural counterexample, but its 1,440 items include synthetic syllables, vowels, laughter and clicks. Its deletion rate must not be described as an observed error rate on natural repeated speech. The current Commands test adds actual model inference over repeated human word recordings.

## Compute, evidence and spending

All model inference runs on CUDA in HF Jobs or Modal, tracked by ORX. The Mac handles preparation, launch and independent scoring only. The previous Modal failure was caused by a missing Python 3.14 CA trust bundle; installing the existing certifi bundle repaired default TLS verification. Verification was not disabled.

| Condition | Commit | ORX run | Provider |
|---|---|---|---|
| Small HF beam5 | `aa6cbff` | `0720c354-f39b-400a-957c-d659dbb94348` | HF A100 80 GB |
| Small official fallback | `5a6f27a` | `f4bdc853-6c29-465b-b4aa-22f286973384` | Modal H100 80 GB |
| Turbo HF beam5 | `5d8f689` | `1db19e10-e1e4-4597-b0f5-8aec5a2078b4` | HF A100 80 GB |

Both providers have a separate new $20 cap. `scripts/termeval/launch_budgeted_orx.py` reserves the entire timeout cost plus $0.25 before launching. HF A100 is $2.50/hour; the Modal H100 allowance is deliberately conservative at $8/hour including host resources. Three initial one-hour reservations total **$5.50 HF and $8.25 Modal**; reservations are not bills. No reservation is released on the strength of a runtime estimate. Jobs have a one-hour provider timeout and 45-minute inference guard.

Every run emits append-only compressed prediction chunks and a complete hash-checked artifact bundle. The independent validator verifies manifest, prediction and policy hashes; rescoring; paired IDs; references and audio; pooled counts; and speaker-cluster bootstrap intervals. Gated Lahaja references and predictions are not published to GitHub. Runtime across different GPUs is descriptive and cannot establish a decoder speedup.

## Relevant prior work and positioning

[How to Reduce Whisper Hallucination](https://www.alphaxiv.org/pdf/2609.32560) explicitly measures suppression against recovery of genuinely supported text; its future-work section lists a decoder-side ablation as unrun. This motivates measuring decoder choices but does not establish novelty for our study. Its limitations also distinguish synthetic positives from natural speech.

[From Text Metrics to Model Internals](https://www.alphaxiv.org/pdf/2606.23060) studies reference-free detection and model-internal signals. A simple repetition trigger must be assessed as a heuristic, not a new hallucination detector.

[Hallucination Space Projection](https://www.alphaxiv.org/pdf/2609.04561) evaluates hallucination reduction alongside WER and false rejection. The tradeoff itself is established research; this experiment must earn a narrower empirical claim.

Implementation and data sources: [official Whisper decoder](https://github.com/openai/whisper/blob/main/whisper/transcribe.py), [Speech Commands description](https://www.tensorflow.org/datasets/catalog/speech_commands), [HF Jobs prices](https://huggingface.co/docs/hub/jobs-pricing), [Modal prices](https://modal.com/pricing). The installed Whisper release is pinned to `20250625`; current upstream pages are context, not a substitute for recorded package versions.

## Addendum before unmasking outcomes: acoustic alignment gate

A second round will test a more restrictive repair, using the same Whisper-small weights to align its own candidate transcript to the audio. Trigger on the frozen five-word run. Remove only later copies in that run that receive exactly zero duration from Whisper's published cross-attention/DTW aligner; preserve the first copy. If alignment and transcript word boundaries do not map exactly, abstain. No tuning of the duration threshold: it is zero, with numerical tolerance `1e-8`. A zero estimated span is not proof that a word was not spoken. The stress test must expose alignment-induced false deletions.

Also report the duration-based truncation control from [Timestamped Hindi speech transcription using Whisper](https://www.alphaxiv.org/pdf/2610.08847): at most one character per 20 ms audio frame, cut at a word boundary. Our implementation counts Unicode characters including spaces. This is an implementation of a published baseline, not a new method. That paper already studies Hindi hallucination correction using CTC confidence and length, so neither acoustic verification nor Hindi-specific correction is automatically novel here. [AURA](https://www.alphaxiv.org/pdf/2609.23979) also uses cross-attention uncertainty to route edits; the proposed zero-duration rule is a much simpler diagnostic with unproven effectiveness.

This addendum is recorded while runs are in progress, before inspecting the new prediction content or aggregate quality scores. Compute progress and input checks have been inspected. Every added comparison must be reported; 95% intervals are exploratory and are not adjusted for this expanded set of comparisons.
