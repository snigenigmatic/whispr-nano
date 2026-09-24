> **Audio QC correction (2026-09-22):** Five public files contain transcript text over exactly silent audio. Affected Assamese/Bengali quality comparisons and aggregates containing them are withdrawn pending adjudication. Original metrics below are preserved as historical records. See the [source-verified audit](artifacts/audio_content_audit/RESULTS.md).

# Public Humyn corpus structural QC — 22 September 2026

Pinned source: `2e4879d569bd57f675c227a41fecf4a2e56f467c`. Downloaded and decoded all 37 conversations (8.345 hours). Audio is converted to mono 16 kHz float WAV, with source and derived hashes retained. No SPIRE material was accessed.

30 recordings passed structural checks. Seven have reference segment end times more than 0.25 seconds beyond the decoded audio length. This flags a potential reference/alignment problem; it does not establish transcript inaccuracy or BRIDGE membership. Do not trim references to repair these automatically.

| File | Audio seconds | Last reference end | Excess seconds |
|---|---:|---:|---:|
| Bhojpuri_2.wav | 894.70 | 915.36 | 20.66 |
| Marathi_2.wav | 689.54 | 708.22 | 18.68 |
| Marathi_3.wav | 610.28 | 805.58 | 195.30 |
| Punjabi_1.wav | 499.32 | 871.84 | 372.52 |
| Punjabi_2.wav | 372.51 | 1054.71 | 682.20 |
| Punjabi_3.wav | 285.33 | 1109.30 | 823.97 |
| Tamil_2.wav | 901.97 | 902.39 | 0.42 |

All three Punjabi files are currently flagged. Therefore Punjabi cannot be included in the QC-passing primary cohort until the discrepancy is resolved. The Tamil excess is only 0.42 seconds and may be rounding/alignment imprecision; the same predeclared 0.25-second tolerance was applied to every file.

The three Hindi recordings pass structural QC and total 2,635 seconds (43.92 minutes). They are a public development pilot, not a held-out benchmark. File-local speaker labels do not establish independent speakers.

Frozen label mapping: released `Chattisgarhi` maps to `hne`; the original spelling is retained in every manifest row. References are concatenated in stable start-time order; overlapped speech ordering has not been manually adjudicated.

Receipts: `artifacts/corpus_receipt_v1.json`. Audio and references stay under ignored `data/bridge_open/`; the acquisition manifest records hashes and source filenames.
