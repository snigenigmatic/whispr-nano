# Open ASR candidate roster — 22 September 2026

27 verified model repositories, with immutable revisions and card hashes in `model_roster.json`. No weights downloaded and no inference performed for this selection. “Best” is a hypothesis to test on Indian speech, not a ranking inferred from downloads or vendor tables.

## Broad Indian-language comparisons

| Model | Public Humyn languages | Stage | Access |
|---|---:|---|---|
| [openai/whisper-tiny](https://huggingface.co/openai/whisper-tiny) | 10/13 | expanded | public |
| [openai/whisper-small](https://huggingface.co/openai/whisper-small) | 10/13 | first_batch | public |
| [openai/whisper-large-v3](https://huggingface.co/openai/whisper-large-v3) | 10/13 | first_batch | public |
| [openai/whisper-large-v3-turbo](https://huggingface.co/openai/whisper-large-v3-turbo) | 10/13 | first_batch | public |
| [ai4bharat/indic-conformer-600m-multilingual](https://huggingface.co/ai4bharat/indic-conformer-600m-multilingual) | 10/13 | first_batch_after_access | gated; account access unverified |
| [bodhan-ai/indic-transcribe-core](https://huggingface.co/bodhan-ai/indic-transcribe-core) | 11/13 | first_batch_after_access | gated; account access unverified |
| [bodhan-ai/indic-transcribe-flex](https://huggingface.co/bodhan-ai/indic-transcribe-flex) | 13/13 | first_batch_after_access | gated; account access unverified |
| [facebook/omniASR-CTC-1B](https://huggingface.co/facebook/omniASR-CTC-1B) | 13/13 | first_batch | public |
| [facebook/omniASR-LLM-7B](https://huggingface.co/facebook/omniASR-LLM-7B) | 13/13 | expanded | public |

## Hindi and Indian-English comparisons

| Model | Public Humyn languages | Stage | Access |
|---|---:|---|---|
| [Qwen/Qwen3-ASR-0.6B](https://huggingface.co/Qwen/Qwen3-ASR-0.6B) | 1/13 | first_batch | public |
| [Qwen/Qwen3-ASR-1.7B](https://huggingface.co/Qwen/Qwen3-ASR-1.7B) | 1/13 | first_batch | public |
| [microsoft/VibeVoice-ASR](https://huggingface.co/microsoft/VibeVoice-ASR) | 1/13 | expanded | public |
| [nvidia/nemotron-3.5-asr-streaming-0.6b](https://huggingface.co/nvidia/nemotron-3.5-asr-streaming-0.6b) | 1/13 | first_batch | public |
| [mistralai/Voxtral-Mini-4B-Realtime-2602](https://huggingface.co/mistralai/Voxtral-Mini-4B-Realtime-2602) | 1/13 | first_batch | public |
| [mistralai/Voxtral-Mini-3B-2507](https://huggingface.co/mistralai/Voxtral-Mini-3B-2507) | 1/13 | expanded | public |
| [FunAudioLLM/Fun-ASR-MLT-Nano-2512](https://huggingface.co/FunAudioLLM/Fun-ASR-MLT-Nano-2512) | 1/13 | expanded | public |

## English supplementary track

| Model | Public Humyn languages | Stage | Access |
|---|---:|---|---|
| [CohereLabs/cohere-transcribe-03-2026](https://huggingface.co/CohereLabs/cohere-transcribe-03-2026) | 0/13 | expanded | gated; account access unverified |
| [ibm-granite/granite-speech-5.0-470m-turboctc](https://huggingface.co/ibm-granite/granite-speech-5.0-470m-turboctc) | 0/13 | english_first_batch | public |
| [ibm-granite/granite-speech-4.1-2b](https://huggingface.co/ibm-granite/granite-speech-4.1-2b) | 0/13 | expanded | public |
| [ibm-granite/granite-speech-4.1-2b-plus](https://huggingface.co/ibm-granite/granite-speech-4.1-2b-plus) | 0/13 | expanded | public |
| [nvidia/parakeet-tdt-0.6b-v3](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3) | 0/13 | english_first_batch | public |
| [nvidia/canary-1b-v2](https://huggingface.co/nvidia/canary-1b-v2) | 0/13 | expanded | public |
| [nvidia/canary-qwen-2.5b](https://huggingface.co/nvidia/canary-qwen-2.5b) | 0/13 | expanded | public |
| [FunAudioLLM/SenseVoiceSmall](https://huggingface.co/FunAudioLLM/SenseVoiceSmall) | 0/13 | expanded | public |
| [zai-org/GLM-ASR-Nano-2512](https://huggingface.co/zai-org/GLM-ASR-Nano-2512) | 0/13 | expanded | public |
| [moonshine-ai/moonshine-streaming-medium](https://huggingface.co/moonshine-ai/moonshine-streaming-medium) | 0/13 | english_first_batch | public |
| [kyutai/stt-2.6b-en](https://huggingface.co/kyutai/stt-2.6b-en) | 0/13 | expanded | public |

## Interpretation and execution order

Start the public, broad-coverage comparison with Whisper small, large-v3, turbo, and OmniASR-CTC-1B. Add IndicConformer and Bodhan Core/Flex when account access is verified. Compare Hindi-supported newcomers on exactly the same Hindi conversations. Three public Hindi conversations provide only a pilot; they cannot establish population-level superiority.

Keep English experiments separate. SPIRE provides an Indian-English track, but its reserved validation/test material remains unopened until the protocol is frozen. The current roster authorizes no release of that data.

Then expand to larger models, streaming conditions, and the compact English models. “First batch” is an order of priority, not a GPU job authorization or cost estimate. The existing compute cap has not been raised.

## Release, license, and scoring qualifications

- The snapshot records repository creation/update dates, not verified public release dates. Model family names are not reliable total parameter counts; raw HF tensor metadata is retained separately.
- Bodhan Core/Flex have custom Indic Open Model License terms and gated access; do not call every candidate permissively open-source. Nemotron uses OpenMDW; several NVIDIA/Kyutai weights use CC BY. Check redistribution terms before packaging weights.
- Whisper HF license tags and the official [Whisper repository](https://github.com/openai/whisper) differ for some versions; retain provenance and resolve the release notice.
- Qwen HF runtime conversions and other quantizations are implementation variants, not extra independent base models.
- Bodhan output mode changes script conventions. Compare raw and frozen-normalized results; a script mismatch is not automatically an acoustic recognition error.
- Omni all-13 coverage was checked against the pinned official registry linked in the JSON supplement, not inferred merely from the advertised 1,600+ languages.
- `ai4bharat/indic-asr-nemotron-600m` returned 401 and remains unverified. The initial Kyutai candidate name was corrected to `kyutai/stt-2.6b-en`; preserve the failed discovery record.

## Benchmark contract

Use `PROTOCOL.md` and `metrics_contract.json`. Public Humyn data membership in the BRIDGE scored corpus is unverified; do not place these scores in its leaderboard. Score per language and matched file set. Record inference failures and coverage as well as WER/CER/WIL, runtime, and peak memory. Any native-long-form versus standardized-segmentation experiment must state the distinct pipelines and supported input limits.

A useful publication question is whether model rankings, script handling, and transcription failures change on long Indian conversations under documented segmentation and resource constraints. This is a proposed study, not an established novel finding. Running more models alone does not establish a publishable contribution.
