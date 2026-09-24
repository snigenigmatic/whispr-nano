> **Audio QC correction (2026-09-22):** Five public files contain transcript text over exactly silent audio. Affected Assamese/Bengali quality comparisons and aggregates containing them are withdrawn pending adjudication. Original metrics below are preserved as historical records. See the [source-verified audit](artifacts/audio_content_audit/RESULTS.md).

# Decoder repair results: Hindi turbo pilot

This is a bounded engineering comparison on the same three public development
conversations (`humyn-13-00`, `humyn-13-01`, `humyn-14-00`). It compares the
completed MPS baseline (`hindi_turbo_v1`) with CUDA greedy, CUDA beam-5, and
CUDA timestamp-enabled artifacts. All four use the same model revision
`openai/whisper-large-v3-turbo@41f01f3fe87f28c78e2fbf8b568835947dd65ed9`, the
same frozen manifest (`e6992a043c9448a9974d1707186ee8cbfda87154b1d4aad940c8546c4054aa78`),
30-second non-overlapping chunks, and the same frozen NFC/casefold/punctuation/
whitespace scorer. No lexical normalization or reference text changed.

## Aggregate result

| condition | WER errors / ref words | micro WER | CER errors / ref chars | micro CER | mean conversation WER | capped chunks |
|---|---:|---:|---:|---:|---:|---:|
| MPS baseline | 4356 / 6503 | 0.669845 | 12408 / 22557 | 0.550073 | 0.671475 | 15 / 89 |
| CUDA greedy | 4356 / 6503 | 0.669845 | 12408 / 22557 | 0.550073 | 0.671475 | 15 / 89 |
| CUDA beam-5 | 3383 / 6503 | 0.520221 | 9751 / 22557 | 0.432283 | 0.519259 | 4 / 89 |
| CUDA timestamps | 4168 / 6503 | 0.640935 | 11371 / 22557 | 0.504101 | 0.642378 | 21 / 89 |

Beam-5 reduces pooled WER by 0.149623 (14.962 percentage points) and pooled
CER by 0.117790 points versus the MPS baseline. Timestamp mode reduces pooled
WER by 0.028910 points and pooled CER by 0.045973 points. CUDA greedy is
transcript-identical to the MPS baseline for every conversation, so its metric
identity is expected rather than a normalization artifact.

## Per-conversation comparison

Values are WER / CER; parenthesized values are absolute changes from the MPS
baseline in percentage points.

| conversation | MPS baseline | CUDA greedy | CUDA beam-5 | CUDA timestamps |
|---|---:|---:|---:|---:|
| `humyn-13-00` | 0.719898 / 0.578420 | 0.719898 / 0.578420 (+0.000 / +0.000) | 0.531122 / 0.397785 (-18.878 / -18.063) | 0.669898 / 0.509129 (-5.000 / -6.929) |
| `humyn-13-01` | 0.633476 / 0.458310 | 0.633476 / 0.458310 (+0.000 / +0.000) | 0.482693 / 0.375070 (-15.078 / -8.324) | 0.631105 / 0.481114 (-0.237 / +2.280) |
| `humyn-14-00` | 0.661052 / 0.603529 | 0.661052 / 0.603529 (+0.000 / +0.000) | 0.543961 / 0.505557 (-11.709 / -9.797) | 0.626130 / 0.519079 (-3.492 / -8.445) |

Beam-5 has lower WER and CER on all three conversations. Timestamp mode has
lower WER on all three, but its CER worsens on `humyn-13-01` by 2.280 points.

## Capping and evidence checks

The baseline and CUDA greedy each have 7, 5, and 3 capped chunks respectively
for the three conversations. Beam-5 has 2, 2, and 0; the third conversation
therefore reaches clean `success`, while the first two remain
`generation_limit`. Timestamp mode has 7, 9, and 5 capped chunks. Capped
outputs remain scored under the registered estimand; they are not counted as
clean completions.

All artifacts contain exactly the same three conversation IDs. Their per-row
audio and reference hashes are the same across conditions, and each summary
reports 6503 reference words and 22557 reference characters. The retained
reference cohort is therefore common across all decoder comparisons. The
MPS/CUDA greedy raw transcript strings are byte-for-byte identical for each
conversation.

## Resource accounting

The CUDA `cloud_receipt.json` estimates are:

- greedy: `$0.0857836101` estimated function resource cost, 72.396 seconds;
- beam-5: `$0.0674644777`, 56.936 seconds;
- timestamps: `$0.0679531532`, 57.348 seconds.

Their summed estimated resource cost is `$0.2212012409`. These are estimates,
not provider bills (`provider_bill_verified` is false in each receipt). The
durable local budget ledger confirms a conservative `$1.50` reservation for
each of the three completed CUDA jobs, or `$4.50` total, including startup and
overhead allowance. Reservations are upper bounds retained after completion;
they are not claims about actual provider charges and are not reduced to the
small post hoc resource estimates. These three completed reservations are the scope of this report; the later
Qwen reservation is accounted for separately in the shared ledger.

## Limits

This is evidence from three public development conversations, not a held-out
benchmark or population estimate. It demonstrates that beam search changed
these outputs under the fixed pilot setup, but does not establish a novel
algorithm, causal generalization, or publication-quality result. Timestamp
mode also changes the decoding path and adds alignment-related behavior, so it
is not an isolated lexical repair. No lexical normalization changes were made,
and no claim is made about human-perceived transcription quality beyond these
automated metrics.
