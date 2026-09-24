# OmniASR CTC-1B runtime contract

This adapter targets `facebook/omniASR-CTC-1B` at HF revision `8c22e3ffdaa4aab6431b128b84b991a7d9c2515c`. It is a bounded public-data pilot: 30-second absolute audio-clock chunks, batch size 1, no reference text opened during inference, and the existing scorer applied only after the conversation has been fully transcribed.

## Official runtime and artifact pin

The implementation follows Meta's official [Omnilingual ASR repository](https://github.com/facebookresearch/omnilingual-asr/tree/81f51e224ce9e74b02cc2a3eaf21b2d91d743455) and its [inference pipeline](https://github.com/facebookresearch/omnilingual-asr/blob/81f51e224ce9e74b02cc2a3eaf21b2d91d743455/src/omnilingual_asr/models/inference/pipeline.py). The public model card's [pinned HF revision](https://huggingface.co/facebook/omniASR-CTC-1B/tree/8c22e3ffdaa4aab6431b128b84b991a7d9c2515c) contains `omniASR-CTC-1B.pt` and `omniASR_tokenizer.model`; the model artifact's exposed HF linked etag is `e8564fa59dab7caedbcdb54ab7fb9bd6c96989f4d19add2ad81ddd969716952c` (3,900,517,028 bytes), and the tokenizer etag is `b954cc166b0c9e0271b953fa226fa27ca706a25b7029e84579fe2c60a2b451fe` (87,562 bytes). The official [model card](https://huggingface.co/facebook/omniASR-CTC-1B/blob/8c22e3ffdaa4aab6431b128b84b991a7d9c2515c/README.md) identifies the runtime card as `omniASR_CTC_1B` and the license as Apache-2.0.

The Modal image pins `omnilingual-asr` to source commit `81f51e224ce9e74b02cc2a3eaf21b2d91d743455`, `fairseq2==0.6.0` and its matching `fairseq2n==0.6` (the official package declares `>=0.5.2,<=0.6.0`), `torch==2.8.0`, `torchaudio==2.8.0`, `numpy==1.26.4` (fairseq2 0.6.0 declares NumPy `~=1.23`), `scipy==1.15.3`, `soundfile==0.13.1`, `jiwer==4.0.0`, and `huggingface-hub==0.36.2`. The image installs `git`, `build-essential`, and `cmake` for the pinned source/kenlm install, plus `libsndfile1` because the official README calls out libsndfile as required for audio support.

The official pipeline currently rejects audio longer than 40 seconds. The adapter therefore downloads both files with `hf_hub_download(..., revision=8c22e3ff...)`, verifies their content SHA-256 values before loading, overrides the official fairseq2 model/tokenizer cards with those local paths, and passes the loaded model/tokenizer to `ASRInferencePipeline(model_card=None, ...)`. It invokes the official preprocessing path and CTC model forward path rather than custom repository inference code. Before processing the cohort, it also runs the public `pipeline.transcribe()` API and asserts exact text equality with the retained-token path on the first chunk.

## CTC evidence and termination semantics

CTC emits framewise vocabulary logits and decodes with argmax plus consecutive-token collapse. There is no autoregressive EOS token and no `max_new_tokens` termination condition. Each chunk is complete when its forward pass returns; a nonempty transcript is `success`, and an empty transcript is `empty_output`. The adapter records `ctc_raw_token_ids` and `ctc_collapsed_token_ids` when available, and sets `eos_present: null` with `termination_semantics: ctc_no_eos`.

## Required validator change before acceptance

`validate_run.py` currently assumes autoregressive evidence: it requires `generation_sequences`, interprets missing EOS as `generation_limit`, and rejects a chunk whose `eos_present` is not a Boolean. Parent integration must branch on `termination_semantics == "ctc_no_eos"` and instead:

1. require `ctc_raw_token_ids` and `ctc_collapsed_token_ids` to be integer lists;
2. require `eos_present` to be `null` and reject any `generation_limit` status for CTC;
3. check chunk count and contiguous absolute ranges exactly as for Whisper;
4. recompute `text_raw`, normalization, and metrics exactly as today; and
5. keep runtime failures separate from quality outcomes while accepting `success`/`empty_output` only after every configured conversation is scored.

Do not infer truncation from CTC sequence length. Any model forward exception, malformed token evidence, or missing chunk remains a runtime failure.

## Reproducibility boundaries

The worker is configured for one H100, 4 CPUs, 16 GiB memory, 900-second timeout, 180-second startup timeout, and an adapter wall limit of 780 seconds, checked during chunk processing. It records CUDA version, GPU name, peak allocation, package versions, config hash, code hash, and asset hashes. It does not reserve funds, launch jobs, or upload data. The existing launcher remains responsible for those actions and must explicitly select `modal_omni_worker.evaluate`.
