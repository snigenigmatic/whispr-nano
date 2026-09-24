# Hindi audio/reference pairing audit

Checked 2026-09-22 for the three baseline Hindi rows (`humyn-13-00`,
`humyn-13-01`, `humyn-14-00`). This is a structural/audio-file audit only. It
does not assess transcription accuracy and no audio was human-listened.

## Checks performed

The local frozen manifest points to:

| conversation | source file | source shard | output duration | segments |
|---|---|---|---:|---:|
| `humyn-13-00` | `Hindi_1.wav` | `train-00013-of-00022.parquet` | 863.125 s | 148 |
| `humyn-13-01` | `Hindi_2.wav` | `train-00013-of-00022.parquet` | 865.833375 s | 150 |
| `humyn-14-00` | `Hindi_3.wav` | `train-00014-of-00022.parquet` | 906.0416875 s | 77 |

For each derived WAV, `soundfile` reported exactly 16,000 Hz, one channel, and
the frame count implied by the manifest duration:

- `humyn-13-00`: 13,810,000 frames, finite samples, peak abs `0.8055115`.
- `humyn-13-01`: 13,853,334 frames, finite samples, peak abs `0.7988586`.
- `humyn-14-00`: 14,496,667 frames, finite samples, peak abs `0.7943726`.

All three derived-WAV SHA-256 values match their manifest `audio_sha256`. The
manifest's three `source_audio_sha256` values are distinct, and all 37 corpus
rows have distinct source-audio and derived-audio hashes; there is no duplicate
audio collision in the frozen corpus. The `source_file` values are also unique.

Each reference JSON's `raw_transcript_json` parses to the recorded segment
count, and joining nonempty segment text after chronological sorting exactly
reproduces `reference_text`. Segment starts are nondecreasing. The paired
source names are `Hindi_1.wav`, `Hindi_2.wav`, and `Hindi_3.wav`, matching the
three Hindi conversation IDs in order. Speakers are file-local `Speaker A`
and `Speaker B` labels.

`acquire.py` reads `row['audio']['bytes']`, decodes with `soundfile`, checks
nonempty finite samples, averages channels arithmetically, resamples only when
the source rate differs from 16 kHz, and writes float32 WAV at 16 kHz. These
three manifest rows already report 16 kHz mono, so no resampling or channel
mixing was applied to them; the output frame counts and source-rate metadata
are internally consistent.

## Direct upstream byte comparison

The pinned local Hugging Face snapshot was subsequently found at
`/Users/prateekraj/.cache/huggingface/hub/datasets--humyn-labs--Indic-High-Fidelity-MultiSpeaker-ASR/snapshots/2e4879d569bd57f675c227a41fecf4a2e56f467c/data/`.
I reopened both Parquet files with PyArrow and matched rows by their original
`file_name` values:

- Shard 13 SHA-256: `a7764af8b300c266fb04ebfb175869ac276683a74811991809f90adc9d1e55eb`.
  `Hindi_1.wav` is row 0 and `Hindi_2.wav` is row 1.
- Shard 14 SHA-256: `dd201fe1357945f0687e83ada4e49b2e75cefd7b972927727e4aeb072ff1118b`.
  `Hindi_3.wav` is row 0.

For all three rows, the embedded `audio.bytes` SHA-256 equals the manifest's
`source_audio_sha256`, and the Parquet `transcript_json` bytes exactly equal the
retained reference JSON `raw_transcript_json`. Decoding each embedded payload
with `soundfile` and applying the exact `acquire.py` policy (float32 decode,
arithmetic channel mean, conditional `resample_poly`, float32 16 kHz output)
produced the retained WAV byte-for-byte at the sample-array level: frame counts
match and maximum absolute sample difference was `0.0` for all three.

## Findings and limitations

There is no local evidence that these three derived audio files are corrupt,
cross-paired, duplicated, truncated, or silently resampled. The reference
join and hash checks also pass. `humyn-14-00` has one adjacent chronological
segment overlap; `acquire.py` intentionally keeps overlapping speech in start
time order and concatenates it into one reference. The first two have no
adjacent overlap. Total uncovered inter-segment gaps are approximately 126.671,
148.018, and 111.546 seconds respectively, which is expected for conversational
audio with silence and is not evidence of corruption.

The upstream Parquet payloads are now independently verified from the pinned
local snapshot; no cloud fetch was needed. This confirms the source bytes,
transcript bytes, decode, channel policy, resampling policy, and derived WAVs,
but does not assess transcription quality. No human listening was performed.

Nothing here explains transcription quality or the observed WER. A remaining
structural sensitivity is the long 14–15 minute duration of each file and the
single chronological concatenation of two-speaker reference segments; judging
whether those conventions affect accuracy requires reviewing model outputs and,
for audio quality, human listening.
