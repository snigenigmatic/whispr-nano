# BRIDGE ASR 2.0: feasibility for a six-model open study

Checked 2026-09-22. This is a bounded, read-only review of first-party Humyn and Hugging Face pages. No audio, Parquet shard, or cloud/API run was downloaded.

## Decision

The public material is sufficient to design an open study on Humyn-labelled sample datasets, but it is not sufficient for an **exact BRIDGE ASR 2.0 replication**. A six-model study may be run on the public sample cards after freezing a local manifest and open scoring protocol, but its numbers must be labelled an independent open-sample study. They cannot be called BRIDGE-matched until Humyn publishes an authoritative evaluation-set manifest (or file-to-BRIDGE mapping), reference/normalization version, executable scorer and inference configuration.

## Verified on the official BRIDGE page

- Humyn describes BRIDGE ASR 2.0 as 23 ASR models evaluated on 21 languages with a six-metric stack. It says recordings are genuine spontaneous dual-speaker conversations, tagged across seven cohort dimensions, passed through the same three-step text cleanup, and evaluated on the same set for every provider. [Official BRIDGE page](https://humynlabs.ai/bridge/ASR/2.0) (page text around lines 29–63).
- The page names the six displayed metrics as WER, CER, SemSim, CS F1, PIER, and WIL, and says the live evaluation set is inspectable. [Official BRIDGE page](https://humynlabs.ai/bridge/ASR/2.0) (around lines 55–75).
- The dataset section says the **sample BRIDGE corpus** contains Indic, Latin-American Spanish, Brazilian Portuguese, and Vietnamese audio, golden transcripts, speaker metadata, cohort labels, and evaluation scripts “under the IndicBench dataset card”; it says additional languages and an overlap-focused corpus are in preparation. The linked page resolves to Humyn’s generic dataset catalog, whose server-rendered response does not expose an IndicBench repository, file manifest, checksum, license, or scripts. [Dataset section](https://humynlabs.ai/bridge/ASR/2.0) (around lines 671–684); [linked catalog](https://humynlabs.ai/datasets?sense=sound&group=Bridge).
- The page’s linked [BRIDGE Metrics Reference](https://sideways-vertebra-975.notion.site/BRIDGE-Metrics-Reference-20-Measures-in-Six-Groups-3d527e4b975081f49d8dd0af32e26408?source=copy_link) and [Humyn Score Methodology](https://sideways-vertebra-975.notion.site/Humyn-Score-Methodology-3d527e4b9750811285acccae76bf2358) both returned HTTP 404 on this check. This is a current access result, not an inference that the documents never existed.

## Exact seven cohort dimensions

The official page’s serialized first-party data contains this exact ordered list:

| Dimension | Public category order visible in page data |
|---|---|
| Region | Same State; Cross State |
| Gender | Dimension name visible; category order not exposed in the checked page data |
| Age | Dimension name visible; category order not exposed |
| Density | Dimension name visible; category order not exposed |
| Duration | Short; Medium; Long; Very Long |
| Overlap | No Overlap; Has Overlap |
| Gap | No Gap; Low; Medium; High |

The page also gives examples of cohort analyses and says categories with fewer than two audio samples for a model are excluded, but it does not publish the row-level cohort table or a frozen manifest. [Official BRIDGE page](https://humynlabs.ai/bridge/ASR/2.0) (cohort section around lines 243–535; serialized page data includes `cohortDims` and schemas).

## Public Hugging Face inventory

The current Hugging Face API lists 12 public datasets under the `humyn-labs` organization. Three cards are relevant to the language scope claimed for the sample corpus:

| Repository (revision checked) | Public metadata | What is actually exposed |
|---|---|---|
| [`humyn-labs/Indic-High-Fidelity-MultiSpeaker-ASR`](https://huggingface.co/datasets/humyn-labs/Indic-High-Fidelity-MultiSpeaker-ASR), `2e4879d569bd57f675c227a41fecf4a2e56f467c` | Ungated, public, CC BY 4.0; train split 37 examples, 1,065,664,379 bytes; 22 Parquet shards | Features are `language`, `file_name`, `audio`, and `transcript_json`; card describes natural dual-speaker conversations and timestamped transcripts across Indic languages. |
| [`humyn-labs/LATAM-High-Fidelity-ASR`](https://huggingface.co/datasets/humyn-labs/LATAM-High-Fidelity-ASR), `84b7e64ea1a449f3538dd3e0d3bb0aef7f449dc9` | Ungated, public, CC BY 4.0; train split 28 examples, 1,393,845,879 bytes; 7 Parquet shards | Features are `language`, `file_name`, `audio`, `transcript_json`, and `type`; README lists Peruvian, Venezuelan, and Argentinian Spanish plus Brazilian Portuguese, with 10–30 minute conversational recordings. |
| [`humyn-labs/Asian-High-Fidelity-ASR-Dataset`](https://huggingface.co/datasets/humyn-labs/Asian-High-Fidelity-ASR-Dataset), `9d640c7c6a94cd21641d0934591da7870e00c9b5` | Ungated, public, CC BY 4.0; train split 242 examples, 2,099,204,670 bytes; 11 Parquet shards | Features are `language`, `file_name`, `audio`, `transcript_json`, and `type`; card languages are Arabic, Korean, and Vietnamese. Vietnamese is present, but the card is broader than the BRIDGE sample claim and includes single- and multi-speaker data. |

Evidence: the read-only [organization dataset API](https://huggingface.co/api/datasets?author=humyn-labs&limit=100), each repository’s [metadata API](https://huggingface.co/api/datasets/humyn-labs/Indic-High-Fidelity-MultiSpeaker-ASR), [Indic tree](https://huggingface.co/api/datasets/humyn-labs/Indic-High-Fidelity-MultiSpeaker-ASR/tree/main?recursive=true), [LATAM README](https://huggingface.co/datasets/humyn-labs/LATAM-High-Fidelity-ASR/raw/main/README.md), and [Asian README](https://huggingface.co/datasets/humyn-labs/Asian-High-Fidelity-ASR-Dataset/raw/main/README.md). The three repository trees contain README/metadata plus Parquet shards; no public BRIDGE manifest, scorer, normalizer, model-output files, or benchmark config was found in the inspected trees. The embedded transcript field is `transcript_json`; no separate golden-reference file is listed.

## Replication boundary for the six open models

**Verified available:** public audio-bearing cards, embedded transcript fields, basic language/file metadata, CC BY 4.0 card licenses, and the seven dimension names/category orders listed above.

**Not verified as BRIDGE-authoritative:** exact membership of any card row in the scored BRIDGE set; row-level cohort labels for all seven dimensions; transcript/reference version; the three-step cleanup implementation; tokenization and all six metric implementations; HumynScore formula/weights/aggregation; per-file provider predictions; provider model/API versions and decoding/preprocessing configuration; and a common BRIDGE manifest/checksum.

**Inference:** the six open models can be compared on a newly frozen intersection of public rows if their inference and failure handling are documented, and WER/CER (plus any independently frozen open metrics) can be computed under the repository’s metrics contract. Such results are an open protocol study, not an exact reproduction of the BRIDGE leaderboard or HumynScore. The public card contents alone do not prove that the three cards are the complete BRIDGE sample or that their rows share BRIDGE’s seven cohort annotations.

