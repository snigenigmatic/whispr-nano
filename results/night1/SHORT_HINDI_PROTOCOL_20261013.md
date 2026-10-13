# Short Hindi speech versus silence: decoder ablation

This follow-up was specified after inspecting the long-clip decoder and repetition results, before inspecting the short-stratum predictions. It is exploratory. It tests an existing decoder option, not a claimed new architecture. C. Kaustubh's October 16 review is the delivery target.

## Hypothesis and controls

Whisper's no-speech skip may discard short real utterances. Disable only `no_speech_threshold` (`0.6` → `None`) in OpenAI Whisper Small's standard transcribe recipe. Keep the checkpoint, package versions, seed derivation, temperature fallback, beam/best-of settings, language, audio and HF A100 hardware identical. Positive recovery must be judged alongside new emissions on silence. A lower WER alone cannot establish a safe intervention.

Report pooled WER, exact normalized transcript accuracy, empty outputs, substitutions/deletions/insertions, clip-level helped/harmed counts, and paired speaker-bootstrap 95% intervals. For silence, WER is undefined: report emitted-word totals and the count of inputs that elicit words. Do not pool silence with speech or describe the constructed silence sample as a population false-positive rate. Report the frozen text-loop repair and published duration-cap baseline on all completed outputs, including failures. There is no threshold tuning on these outcomes.

## Frozen data

Lahaja revision `d4ffd2ecbdd933e37c917ddcf620eef159ceb3a7`; exclude the same 26 development speakers. Select all remaining clips with duration strictly between zero and two seconds and exactly one normalized reference word. This gives **414 original recordings, 102 speakers, 17 states**. Durations range from 0.313 to 1.940 seconds. These recording IDs do not overlap the previous 801 long clips; speakers do overlap. Selection uses released transcript/duration metadata, not model outputs.

Choose one short source recording per speaker by minimum SHA-256 of `short-hindi-20261013|<id>`. Concatenate that unchanged waveform five and eight times, inserting 150 ms of zero audio between copies: **102 five-repeat and 102 eight-repeat controls**. These are repeated human recordings, not natural conversational repetitions.

Add Scicom's 42 silence controls from revision `8126be4a4b83afefeef9394ff1312b635a499d82`. Explicitly crop each input at 25 seconds before decoding. Cropping makes six controls duplicate another exact float32 waveform; deduplicate by audio hash and retain the equivalent source IDs. This yields **36 unique silence inputs** across digital zero, dither, hiss, hum and room-tone-labelled floors. They are constructed/calibration controls, not 36 independently sampled natural environments.

Total: **654 inputs; 36.0746 audio minutes**. Audio is mono, finite, float32, 16 kHz; source duration agrees within 5 ms; speech sources have nonzero amplitude. Combined manifest SHA-256:

`ab7da06da199001f450d4958a25cffda2a5b452579f4261b07a08bac82db000a`

## Runs and evidence

| Setting | Commit | ORX run | HF job |
|---|---|---|---|
| Standard no-speech skip | `ba9386e` | `e0553e70-2973-486e-9553-befe8b2ded6c` | [A100 job](https://huggingface.co/jobs/snigenigmatic/6ace1874c66288145222a61d) |
| Skip disabled | `dd67a0f` | `8182a293-4ea8-41ae-9b1a-bca3532caef0` | [A100 job](https://huggingface.co/jobs/snigenigmatic/6ace1a035ccb47ba0f787b9d) |

Both runs use the fixed ORX command `bash experiments/cloud_entrypoint.sh`, a 40-minute provider timeout and 35-minute inference guard. Each reserves $1.92 within the separate $20 HF limit. Reservations are upper allowances, not measured charges. Full prediction/artifact bundles are hash-checked and independently rescored by `scripts/termeval/package_short_hindi_study.py`. No gated audio or reference transcripts are added to public Git.

The original study also adds an official Turbo decoder control (`32e3fc8`, run `aa0ebb98-c8e5-48df-a79f-f723cec9105e`) after inspecting the other three cells. It completes the two-model/two-decoder-recipe comparison on the original 1,101 inputs. It is explicitly an outcome-motivated control. It runs on Modal H100 with a 45-minute timeout and $6.25 reservation. This comparison changes complete decoding recipes, so it cannot isolate temperature fallback from timestamp segmentation or implementation differences.

## Interpretation rule

If the no-speech ablation improves speech but increases silence emissions, report the tradeoff; do not call it safe. If it changes nothing or degrades results, retain that negative result and stop that intervention. No successful publication, fairness improvement or Indian-English transfer is inferred from this Hindi experiment.
