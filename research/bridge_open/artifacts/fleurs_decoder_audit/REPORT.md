# FLEURS Large-v3 short-form decoder audit

## Scope and checkout identity

The requested worktree is currently at `5d94067ae35badbd4c1e2fc10d610f7ba398efac` (`Freeze hf-fleurs-test-large-as-short-768-end-v2 on 212 official test recordings`). Commit `33a032ec2c79cfeeb9d4217cd3425c9f3f70729c` is the deliberate frozen harness baseline; its baseline config names Turbo, while each Large-v3 shard commit supplies the Large-v3 model settings. `fleurs_dispatch_coverage_audit.json` reports that every Large-v3 shard’s inference/scoring source hashes match that baseline, and the per-shard `orx_source_receipt.json` files bind each dispatch to its immutable commit. This is therefore not a provenance defect. The Large-v3 artifacts’ effective config names `openai/whisper-large-v3`, revision `06f233fe06e710322aca913c1bc4249a0d71fce1`.

## What the fixed short runner actually does

`research/bridge_open/run_whisper_batched.py` groups rows by the two-letter manifest language code, so a generation call never mixes language hints (`run_whisper_batched.py:64-69,254-274`). The manifest maps Assamese→`as`, Bengali→`bn`, and Hindi→`hi` in `acquire.py:20-22`; these are valid Transformers Whisper language codes. Transformers 4.57.6 maps `as` to Assamese, `bn` to Bengali, and `hi` to Hindi (`transformers/models/whisper/tokenization_whisper.py:105,131,179`). The saved predictions confirm the forced prompt IDs: Assamese `50350`, Bengali `50302`, Hindi `50276`, after decoder start `50258`.

The runner verifies each WAV hash, requires mono 16 kHz finite nonzero audio, and rejects clips over 30 seconds before model load (`run_whisper_batched.py:72-87,194-210`). It requests `padding="max_length"`, 480,000 samples, no truncation, and an attention mask (`:268-270`), then uses float32 `WhisperForConditionalGeneration(...).to(device).eval()` with SDPA (`:230-233`). CUDA TF32 is disabled (`:122-129`). Generation is deterministic beam 5, `do_sample=False`, `return_timestamps=False`, `max_new_tokens=440` (`:272-274`). Raw sequences are retained and decoded with `skip_special_tokens=True` (`:275-276`); EOS is checked from the raw sequence, so removing the prompt/EOS for displayed text does not hide generation truncation.

These choices are consistent with the HF Transformers Whisper API for a short, batched, same-language condition. The model is not silently using the old checkpoint prompt: `forced_decoder_ids` is explicitly cleared (`:233`), while `language=...` and `task="transcribe"` are passed to `generate`. Transformers 4.57.6’s `_retrieve_init_tokens` resolves two-letter codes through its `lang_to_id` table and inserts the language and task tokens; this matches the observed prompt IDs.

## Explanation of the extreme Assamese/Bengali outputs

The observed Assamese outputs are overwhelmingly Devanagari while references are Assamese script. A CPU inspection of the completed artifacts found:

| language | utterances | output alphabet signal | status |
|---|---:|---|---|
| Hindi | 208 | 20,697 Devanagari letters, 107 other alphabetic | 208 success |
| Bengali | 304 | 29,851 Bengali letters, 185 Devanagari, 961 other alphabetic | 297 success, 7 generation-limit |
| Assamese | 192 | 18,877 Devanagari letters, 2,694 other alphabetic | 186 success, 6 generation-limit |

The first saved Assamese raw sequence begins `[50258, 50350, 50360, 50364, ...]`; Bengali begins `[50258, 50302, 50360, 50364, ...]`; Hindi begins `[50258, 50276, 50360, 50364, ...]`. Thus the reported `50350`/`50302` prompts are the correct language tokens, not a Hindi/Bengali prompt mix-up. The Assamese Devanagari behavior is consistent with a model output/script failure or model-language limitation, and is not evidence that the runner forced Hindi. Bengali outputs are in the Bengali script but contain spelling/substitution errors, compatible with a Large-v3 quality failure on this cohort. These are hypotheses from static/artifact inspection, not proof of an inherent model limitation; canonical native OpenAI inference is still pending.

The scorer preserves Unicode scripts: it NFC-normalizes, case-folds, replaces only Unicode punctuation, and retains combining marks/symbols (`scoring.py:17-32`). It does not transliterate Assamese, Bengali, or Hindi, so cross-script Assamese output correctly incurs very high WER. Reference text is read only after generation and scored with JiWER (`run_whisper_batched.py:148-159`); it cannot affect decoding.

## Falsifiable bug checks and limits

1. **No obvious decoder-language bug found in static review.** Prompt IDs and the installed 4.57.6 language table agree for all three languages. A replay that logs `generation_config.language`, `generation_config.task`, and the first four decoder IDs would make this check explicit, but the existing raw sequences already provide that evidence. This does not prove that no runtime/model interaction bug exists.
2. **No obvious EOS/token extraction bug found.** In timestamp-free mode `generation_rows` stores `output["sequences"]` unchanged as both `token_ids` and the single raw generation sequence, then the processor removes special tokens only for displayed text (`generation_output.py:10-31`). The validator recomputes metrics from that displayed text and checks EOS status. The 440-token cap explains the seven Bengali and six Assamese `generation_limit` records, but cannot by itself explain the successful records’ script distribution.
3. **No obvious batching/mask mix-up found.** Groups are same-language and the output count is checked against the input group (`run_whisper_batched.py:64-69,254-278`). Each clip is padded to the Whisper 30-second feature window with an attention mask. The model is evaluated in inference mode, and no previous-token context or fallback is enabled. This remains a static review finding, not an exhaustive runtime proof.
4. **128-mel preprocessing is model-controlled.** The runner does not override the feature extractor; it loads the processor from the pinned model snapshot (`:225-232`). Whisper large-v3’s processor configuration is therefore the authority for its 128-bin frontend. The artifact records the pinned model revision and auxiliary-file hashes, but does not copy `preprocessor_config.json`; the exact frontend can be independently bound only by those hashes plus the remote snapshot or a preserved model cache.
5. **Provenance is covered by the study receipts.** The baseline harness commit and per-shard commits are intentionally distinct; dispatch coverage verifies that the inference/scoring files used by Large-v3 match the baseline hashes, while each shard receipt records its immutable source commit and model revision.

## Bottom-line interpretation

The extreme FLEURS WER is not explained by any obvious incorrect language map, wrong decoder prompt, EOS extraction error, batch cross-talk, or padding/mask error found in this static review. The strongest concrete observation is that Assamese is generated mainly in Devanagari despite the correct Assamese prompt token; this is a falsifiable model-output hypothesis, not proof of an inherent model failure. The baseline/shard provenance is covered by the dispatch audit. Before treating the result as a final model-quality claim, complete canonical native OpenAI inference or an equivalent independent decoder check on the same frozen WAVs and prompts.
