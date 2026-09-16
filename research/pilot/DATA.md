# L2-ARCTIC controlled pilot data

This corpus supports a controlled **pilot** of training-data selection with
verified speaker codes and repeated elicitation prompts. It does not establish
an Indian-accent fairness result: only four speakers have Hindi as their L1,
and each pilot test L1 has one held-out speaker.

## Sources and access

The [corpus owners' homepage](https://psi.engr.tamu.edu/l2-arctic-corpus/)
identifies 24 speakers across Arabic, Mandarin, Hindi, Korean, Spanish, and
Vietnamese, four per L1. Its [release documentation](https://psi.engr.tamu.edu/l2-arctic-corpus-docs/)
reports 26,867 scripted recordings and 3,599 manually annotated recordings.
The current documented release is v5.0 (12 March 2020). The data are licensed
CC BY-NC 4.0. Cite Zhao et al., Interspeech 2018,
[doi:10.21437/Interspeech.2018-1110](https://doi.org/10.21437/Interspeech.2018-1110).
The original paper describes an earlier, smaller release; the current speaker
inventory comes from release documentation.

The official download flow requires a name, email, affiliation, and license
agreement, then sends a link by email. We did not submit anyone's identity.
Direct programmatic requests to the official pages returned HTTP403, while
the web reading tool could inspect their text.

For executable access we use the public
[raw v5.0 mirror](https://huggingface.co/datasets/chikingsley/l2-arctic-release-v5.0/tree/cc02fd37197966e5de11ad737f7affd577c61992)
at revision `cc02fd37197966e5de11ad737f7affd577c61992`. Its maintainer describes
speaker ZIPs, original README, prompts, and license as the locally distributed
release. It adds a Hugging Face dataset card; this is not a transformed
Parquet subset. ZIP contents, per-speaker WAV counts, transcript/forced-alignment
pairing, and manual-annotation counts are checked against the official
inventory. **We have not compared all bytes with an independently downloaded
official release.** The pinned mirror and downloaded archive hashes provide
reproducibility, not proof of chain of custody back to the original archive.

The mirror's original README calls the Mandarin speakers “Chinese”; we use the
more specific “Mandarin” label printed by the owners' homepage. Do not interpret
speaker-code characters as demographics beyond the supplied mapping.

## Acquisition and integrity

`data_setup.py` reads the ZIP central directories and one contiguous metadata
range per speaker. It obtains actual per-recording orthographic transcripts and
forced-alignment durations, rather than assigning every speaker the canonical
prompt automatically. It refuses non-206 range responses and unexpectedly large
ranges. ZIP CRCs are checked when each transcript and TextGrid is decompressed.
No bulk audio or final-test predictions are downloaded by this script.

`data/speakers.json` supplies the immutable archive URLs, expected bytes, and
whole-archive SHA256 values from Hugging Face LFS. Verify these hashes on the
compute host before extraction. The locally cached metadata ranges each have
their own hashes in `data/provenance.json`; range acquisition alone cannot verify
a whole-archive SHA256.

The `duration_s` field initially comes from each recording's forced-alignment
`xmax` and is labeled accordingly. Actual WAV headers must be checked on the
compute host before matching audio duration between experimental arms. ZIP file
size under an assumed 44-byte PCM16 header is only a consistency diagnostic.
No duration claim is based on pretending that an uninspected header was decoded.

## Frozen splits

The integer seed is `20260909`. Within each official L1, order the four speaker
codes by SHA256 of `20260909:speaker:<code>`. Assign the first two to training,
the third to development, and the fourth to the sealed test. This rule was
fixed before any L2-ARCTIC teacher or student predictions were obtained.

| L1 | Train speakers | Dev speaker | Test speaker |
|---|---|---|---|
| Arabic | ABA, YBAA | SKA | ZHAA |
| Mandarin | NCC, TXHC | BWC | LXC |
| Hindi | SVBI, RRBI | ASI | TNI |
| Korean | HJK, HKK | YDCK | YKWK |
| Spanish | ERMS, MBMPS | EBVS | NJS |
| Vietnamese | PNV, HQTV | TLV | THV |

Content is also partitioned. First join prompt IDs if any supplied transcript
for them shares exactly the same minimal-normalized text. A content family
therefore contains every rendition of a prompt and any duplicate prompt text.
Order families by SHA256 of `20260909:content:<sorted prompt IDs joined by |>`;
assign floor(70%) of families to training, floor(15%) to development, and the
remainder to test. This is an allocation by content family, not by recording.

The **primary** manifest uses a recording only when its speaker and content
partitions agree. It checks that speaker IDs, prompt IDs, and exact normalized
reference strings have zero pairwise overlap between train/dev/test. These
checks cannot establish semantic independence of near-duplicate sentences or
remove possible pretraining exposure to public ARCTIC prompts.

The **shared-content diagnostic** is a separate file containing dev/test
speakers reading training-content prompts. Its `dev_shared` and `test_shared`
labels prevent accidental use as the primary unseen-content evaluation. The
test diagnostic remains sealed alongside the primary test. Remaining crossed
cells are recorded as excluded; no files disappear silently.

Minimal normalization matches the existing audited implementation: lowercase,
replace characters outside `\w`, whitespace and apostrophes with spaces, then
collapse whitespace. English-normalized evaluation may be reported in addition;
it must not redefine the already frozen content partition after seeing results.

## Acquired inventory and checks

The frozen primary manifest contains 9,495 training recordings from 12 speakers
(9.598 hours), 974 development recordings from six speakers (1.032 hours), and
1,026 sealed test recordings from six speakers (0.992 hours). Each test L1 has
171 recordings. Hours here use the alignment metadata, pending the separately
recorded audio-header verification. There are 1,132 distinct prompt families:
792 train, 169 development, and 171 test. The shared-content files contain
4,559 development and 4,748 test recordings; 6,065 remaining crossed cells are
explicitly excluded.

The 24 raw archives total 7.412 GB; the 18 train/development archives total
5.618 GB. Metadata acquisition used 34.912 MB of cached ranges. All per-speaker
recording counts and the 3,599 manual-annotation total match the release
inventory. Actual orthographic transcripts differ from the canonical PROMPTS
file after minimal normalization for 822 recordings, underscoring why the
per-speaker transcripts are used. Offline reconstruction reproduced the primary
manifest SHA256 `9849d5d0d1214f7129c42fe8006a4ca0b79b36c84e8c80406e228077438931e7`.

Eight adversarial implementation checks reject speaker, prompt and normalized
text leakage, empty/missing splits and diagnostic split labels, and verify
reference-family merging and independence from input row ordering. These are
engineering checks on the split logic, not evidence for the training hypothesis.

The seeded split is not gender-stratified: train has seven male and five female
speakers, development four male and two female, and test one male and five
female. This restricts interpretation of between-split differences. All
experimental arms share the same splits. Confirmation should use more speakers
and/or prespecified speaker rotations rather than treating one selected
six-person test as population evidence.

After the compute host supplies all 18 train/dev WAV reports:

```sh
.venv/bin/python research/pilot/data_setup.py --check-audio
```

This checks archive identity, complete recording inventories, sample rate,
channels, and duration agreement within 10 ms. It writes
`data/audio_header_checks.json` and leaves the frozen split manifest unchanged.
Use decoded durations from `data/audio_validation/<speaker>.json` for exact
matching of exposure between training arms. Test WAVs are not required for this
stage. The metadata range cache is local/rebuildable and excluded from Git.

## Files and interface

- `data/manifest.jsonl`: primary train/dev/test recording rows.
- `data/shared_content_diagnostic.jsonl`: separate shared-prompt diagnostics.
- `data/corpus_metadata.jsonl`: all 26,867 source recordings, before cell selection.
- `data/excluded.jsonl`: crossed cells excluded from these evaluations.
- `data/content_partitions.json`: every prompt family and its frozen allocation.
- `data/speakers.json`: speaker mapping, partition, archive URLs and hashes.
- `data/provenance.json`: source/output/range hashes, validation and limitations.
- `data/sources/`: original README, mirror README, PROMPTS and LICENSE.

Every primary row has `utt_id`, `speaker_id`, `native_language`, `prompt_id`,
`reference_text`, `audio_path`, `duration_s`, and `split`. `audio_path` is relative
to the extraction root, for example `ABA/wav/arctic_a0001.wav`. The row also
records transcript and TextGrid hashes, audio ZIP CRC, source revision, manual
annotation availability, and the separate speaker/content partition labels.

```sh
.venv/bin/python research/pilot/data_setup.py
.venv/bin/python research/pilot/data_setup.py --offline
```

The second command rebuilds the exact manifests from cached metadata without
network access. It does not launch compute, run inference, train a model, or
unseal test predictions.

## Label quality and interpretation

The full corpus provides orthographic transcripts; this project has not
independently listened to and adjudicated all of them. Only the manually
annotated subset has the release's detailed human pronunciation annotations.
Those annotations target pronunciation relative to an expected canonical form;
they are not an independent certification that every reference word was spoken
as written. We retain the manual-annotation flag for auditing, not as evidence
that unflagged utterances are bad or should be filtered out.

The source release already removes some recordings for quality, and two
speakers did not complete every ordinary prompt. Thus any downstream filtering
experiment estimates effects within the released corpus, not the effects of
all selection from the original recording population. Group-specific recording
conditions and the small number of speakers remain limitations. A publishable
confirmation requires independent data and more speaker support than this
pilot supplies.
