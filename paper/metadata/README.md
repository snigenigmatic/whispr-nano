# Svarah metadata provenance

The audit freezes the Hugging Face dataset `ai4bharat/Svarah` at commit
`ebbf7777fe771490696a3f7b007097606fa8c924` (last modified 2025-03-10). Its test
split contains 6,656 rows across three Parquet files. The schema exposes audio,
transcript, duration, and demographic fields, but no speaker identifier.
[Dataset API](https://huggingface.co/api/datasets/ai4bharat/Svarah)

The original authors document `meta_speaker_stats.csv` with a `speaker_id`
field, but the current GitHub tree does not contain that CSV. A historical
README links the original archive; that endpoint returned HTTP 403 during
this audit. An unresolved issue in the authors' repository also reports the
missing speaker column. Two cached copies of the original metadata CSV were
subsequently recovered from the project's existing Modal volume. They are
byte-identical and also lack a speaker column. The original manifest contains
only filepath, text, and duration. All 6,656 original filenames and transcripts
match the pinned Hub rows; durations are identical. These checks establish
that the mapping was not recovered in this sprint, rather than proving that
no mapping exists anywhere.
[Original metadata documentation](https://github.com/AI4Bharat/Svarah/blob/2c644dc4f67da2aae63a0bd9ceb2c6f8679ff05c/README.md),
[historical download reference](https://github.com/AI4Bharat/Svarah/blob/074065ee3addaadfe07739a2bb150f61eb763a28/README.md),
[missing-ID issue](https://github.com/AI4Bharat/Svarah/issues/4)

The dataset card states CC BY 4.0. The Hub requires the account's existing
authorized access to retrieve files. This work preserves attribution and
exports no audio bytes into the metadata files.
[Dataset card](https://huggingface.co/datasets/ai4bharat/Svarah/blob/main/README.md)

The corpus paper reports 117 speakers and explains that collected recordings
were manually segmented. It does not document the filename-to-speaker
mapping. Consequently, an `_fN_chunk_N` filename tag may support a recording
group as a stated inference, but is not a speaker identifier. No identity is
inferred from demographic combinations, transcripts, or voices.
[Original paper, Section 3](https://www.isca-archive.org/interspeech_2023/javed23_interspeech.pdf)

## Reproduction

1. In an environment with `datasets`, `pyarrow`, and authorized Hugging Face
   access, run `extract_svarah_metadata.py`. It selects the path field directly
   from Arrow storage, avoiding materialization of embedded audio byte arrays.
2. Run `validate_metadata.py` after the extraction. It checks every frozen CSV
   hash and every existing utterance's reference, language, gender, and duration
   against the pinned row index before exporting the analysis table.
3. Inspect `../data/metadata_validation.json` for validation results and
   limitations, and `../data/cohort_metadata_counts.csv` for sample counts.

The recording grouping is a sensitivity analysis for known filename groups;
it does not eliminate dependence between multiple recordings by one speaker.
True speaker resampling remains unavailable until an author-provided mapping
is recovered and its row identities are verified.

There are two nested filename groupings: 3,232 distinct `fN` tags and 2,938
numeric prefixes. Every `fN` tag belongs to one prefix; 294 prefixes contain two
tags. Each grouping has consistent values for all eight demographic fields.
Their semantics remain an inference from filenames. The numeric prefix is
available as `coarse_recording_cluster_id` for an additional sensitivity.

| Cohort | Utterances | `fN` tags | Numeric prefixes | Demographic profiles |
|---|---:|---:|---:|---:|
| Full test split | 6,656 | 3,232 | 2,938 | 115 |
| Open-system panel | 2,500 | 1,473 | 1,393 | 115 |
| Matched Saaras panel | 450 | 326 | 324 | 92 |
| Union of existing evaluations | 2,794 | 1,600 | 1,513 | 115 |

The last column counts exact equality across gender, age group, native
language, state, district, qualification, job category, and occupation domain.
It does **not** count verified speakers. Distinct people may share a profile,
and inaccurate metadata could split a person's recordings across profiles.
No value is missing under the explicit null/blank/NA check. Profile IDs are
provided solely for an explicitly labeled exploratory blocking sensitivity;
this analysis was chosen after the metadata were inspected. Bodo has five
profiles in each panel, making profile-based uncertainty particularly
sensitive to individual blocks.

The family column preserves the declared audit grouping. It includes 14
Indo-Aryan language labels, four Dravidian labels, and Bodo as the only
Sino-Tibetan label. A difference involving that last group describes the Bodo
sample and cannot establish a general Sino-Tibetan family effect. The corpus
paper's wording about four language families is not used to invent an
additional category or to reclassify Kashmiri without evidence.
