# Replacement multilingual ASR evaluation dataset check (2026-09-22)

Scope: first-party AI4Bharat repository and Hugging Face cards only. No large archive or private SPIRE split was downloaded.

## Recommendation

Use **AI4Bharat/IndicSUPERB (Kathbath)** as the immediately usable public evaluation source for Hindi, Bengali and the other covered languages (Gujarati, Kannada, Malayalam, Marathi, Odia, Punjabi, Sanskrit, Tamil, Telugu and Urdu). Its official repository gives direct download links, clean `Valid`, `Test Known` and `Test Unknown` splits, separate clean/noisy test material, and human-created transcriptions/annotations. The clean test artifacts are still large (2 GB each), so pilot with a small selected set only after downloading the transcript/index and a bounded object-store range or a few files if the hosting layout permits. The repository does not expose an Assamese split.

Use **ai4bharat/IndicVoices** only when Assamese is required, or when conversational/extempore coverage matters more than a pre-defined held-out ASR test. It has Assamese, Bengali, Hindi and 19 other language configs, a `valid` split, and streaming support, but the current card exposes only `train` and `valid`, is gated behind sharing contact information, and does not document speaker-disjoint train/valid/test construction. Treat `valid` as a provisional evaluation slice after checking speaker overlap yourself; do not call it a clean held-out test without that check.

## Evidence by candidate

### 1. IndicSUPERB / Kathbath

- Exact code/release repository: [AI4Bharat/IndicSUPERB](https://github.com/AI4Bharat/indicSUPERB).
- Coverage and size: 1,684 hours across 12 languages. Per-language hours are Bengali 115.8, Gujarati 129.3, Hindi 150.2, Kannada 65.8, Malayalam 147.3, Marathi 185.2, Odia 111.6, Punjabi 136.9, Sanskrit 115.5, Tamil 185.1, Telugu 154.9 and Urdu 86.7. The card reports male/female speaker counts per language; e.g. Bengali 18/28 and Hindi 58/63.
- Evaluation structure: clean `Train` 85 GB, `Valid` 3 GB, `Test Known` 2 GB and `Test Unknown` 2 GB; noisy `Test Known` 2 GB and `Test Unknown` 1.4 GB. Audio is organized by language/split, with speaker and gender encoded in filenames. The repository’s ASV documentation says testing speakers may be unseen for the unknown-speaker test condition; the ASR README itself does not provide a full speaker-overlap table, so verify speaker IDs before making a speaker-disjoint claim.
- Transcription/quality: the repository says the audio transcriptions and labelled annotations were created by AI4Bharat; clean and noisy transcript bundles are linked. The audio is collected prompted/read speech (the card describes raw text from IndicCorp and provides vocabulary counts), so it is a stronger reproducible read-speech benchmark than a conversational benchmark. It is not suitable as the Assamese leg.
- Access/license: direct object-store links, no Hugging Face contact gate observed. Dataset packaging is stated as CC0; AI4Bharat notes it does not own the underlying IndicCorp raw text, while code/models are MIT. Record the underlying-text caveat in any redistribution/commercial review.

### 2. IndicVoices

- Exact dataset repository: [ai4bharat/IndicVoices](https://huggingface.co/datasets/ai4bharat/IndicVoices); first-party code link is [AI4Bharat/IndicVoices](https://github.com/AI4Bharat/IndicVoices).
- Coverage/mixture: the current card (updated 23 Dec 2025) reports 23.7K hours from 51K speakers, 22 languages and 400+ districts; 11.2K hours are transcribed. Audio mix is 8% read, 76% extempore and 15% conversational. This is the best candidate for realistic conversational evaluation, subject to access and split validation.
- Languages/configs: the card lists configs including Assamese, Bengali, Hindi, Bodo, Dogri, Gujarati, Kannada, Kashmiri, Konkani, Maithili, Malayalam, Manipuri, Marathi, Nepali, Odia, Punjabi, Sanskrit, Santali, Sindhi, Tamil, Telugu and Urdu.
- Available split/size evidence: the card’s dataset metadata exposes only `train` and `valid` for each config. Examples: Assamese valid 3,873 rows; Bengali valid 3,082 rows; Hindi valid 2,792 rows (the exact row counts and byte sizes are in the card’s generated metadata). No `test` split or speaker-disjoint guarantee is stated. The schema includes `speaker_id`, scenario, gender, age group, occupation, area, district and state, so an overlap audit is possible once access is granted.
- Access/license: Hugging Face marks it gated: users must agree to share contact information. The card says an HF token is needed. License is CC BY 4.0. The card reports total file size 745 GB; full-length source audio is separately distributed as five incremental `v1`–`v5` archives per language, so avoid those archives for a pilot.
- Cheap pilot: after accepting the gate, use `datasets.load_dataset("ai4bharat/IndicVoices", "assamese", split="valid", streaming=True)` (similarly `bengali`/`hindi`) and inspect only the first N rows, selecting audio duration and speaker IDs. Streaming is explicitly documented by AI4Bharat and avoids materializing the full archive. Because the card does not promise test isolation, hash or list all speaker IDs in any train slice used by the benchmark and remove collisions from the evaluation slice.
- Metadata limitation: the rich demographic fields enable subgroup reporting, but the first-party card provides no population-representativeness guarantee beyond its collection description and does not publish per-language/per-split demographic counts in the card. Report observed coverage, missingness and self-reported categories rather than claiming national representativeness.

### 3. IndicVoices-R (not the primary ASR choice)

- Exact release repository: [AI4Bharat/IndicVoices-R](https://github.com/AI4Bharat/IndicVoices-R); HF mirror [ai4bharat/indicvoices_r](https://huggingface.co/datasets/ai4bharat/indicvoices_r).
- It contains 1,704 hours, 10,496 speakers and 22 languages, with transcripts and rich speaker/acoustic metadata. However, it is explicitly an ASR-enhanced **TTS** corpus: denoising/speech enhancement was applied to natural-conversation source data. The official README gives tar downloads (for example Assamese) and a CC-BY-4.0 repository license.
- The HF mirror is gated and exposes `train` plus `test` parquet files, but the card does not provide a documented ASR benchmark protocol, test duration, or speaker-disjoint guarantee. Use only as a secondary robustness/TTS source after inspecting its `test` speaker IDs; do not substitute it for Kathbath’s documented ASR benchmark.

## Operational choice

1. For the publication’s core Hindi/Bengali/other-12-language score table, freeze **Kathbath clean Test Unknown** (and optionally Test Known as a separate seen-speaker condition), cite the exact repository commit/date, and retain the official transcript bundle.
2. For Assamese, request IndicVoices access and stream a bounded `valid` pilot. Before publication, perform an explicit speaker-disjoint audit against every training source and report the resulting duration, number of speakers, scenario mix and demographic missingness.
3. If one unified 22-language source is mandatory, IndicVoices is the practical candidate, but the paper must describe it as a curated validation slice rather than an official held-out test until a speaker split is verified.

## First-party citations

- [IndicSUPERB/Kathbath README](https://github.com/AI4Bharat/indicSUPERB) (coverage, split download sizes, transcript claim, license, speaker/file layout).
- [IndicVoices dataset card](https://huggingface.co/datasets/ai4bharat/IndicVoices) (current hours/speakers, speech-type mix, 22-language coverage, gating, streaming, archive size, license).
- [IndicVoices-R README](https://github.com/AI4Bharat/IndicVoices-R) and [HF mirror](https://huggingface.co/datasets/ai4bharat/indicvoices_r) (TTS-oriented derivation, metadata, tar access, gate and visible train/test files).
