# FLEURS scoring sensitivity audit

This CPU-only audit reread the preserved FLEURS full-test prediction JSONL files for both `openai/whisper-large-v3` and `openai/whisper-large-v3-turbo` (2,322 unique utterances per model). It checked each stored WER against the repository scorer, without changing any prediction, metric, or registry artifact. The script and machine-readable output are [audit.py](./audit.py) and [results.json](./results.json).

WER above 100% is valid here. JiWER counts substitutions, deletions, and insertions in the Levenshtein alignment; insertions can make `(S+D+I)/N > 1`. The preserved outputs include extreme examples: Large-v3 Assamese reaches WER 20.81 with 417 insertions, Bengali 12.22 with 202 insertions, and Hindi 7.65 with 153 insertions. These are valid error rates, not scorer overflow or clipping.

| model / language | n | generation-limit | WER > 1 | mean WER | micro WER | mean CER | script mismatch | repeated trigram excess |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Large-v3 / Assamese | 984 | 29 | 520 | 1.136 | 1.142 | 1.092 | 984/984 | 2,083 |
| Large-v3 / Bengali | 920 | 26 | 29 | 0.752 | 0.748 | 0.278 | 7/920 | 85 |
| Large-v3 / Hindi | 418 | 1 | 2 | 0.289 | 0.282 | 0.117 | 0/418 | 205 |
| Turbo / Assamese | 984 | 65 | 311 | 1.337 | 1.368 | 1.172 | 972/984 | 3,257 |
| Turbo / Bengali | 920 | 53 | 107 | 0.866 | 0.860 | 0.357 | 36/920 | 668 |
| Turbo / Hindi | 418 | 3 | 2 | 0.324 | 0.326 | 0.141 | 1/418 | 377 |

“Script mismatch” compares the dominant Unicode script of reference and hypothesis. Assamese references use the Bengali script block, while the bad hypotheses are predominantly Devanagari; this is why the Large-v3 Assamese mismatch rate is exactly 100% and Turbo is 98.8%. Bengali and Hindi mostly remain in their expected scripts. Repeated-trigram excess counts within-utterance repeated 3-word sequences beyond the first occurrence; it catches repetitive hypotheses that exact duplicate utterance counting misses. Exact duplicate hypothesis excess was only 1 for each Indic language/model grouping, so the dominant pathology is within-utterance repetition, not identical outputs across recordings.

The stored study scorer is the NFC/casefold plus punctuation-to-space convention in `research/bridge_open/scoring.py`; all stored WERs recomputed exactly. Its high Assamese WER is accompanied by high CER (1.09–1.17) and near-universal script mismatch, so the result is not plausibly caused only by word tokenization. Bengali has much lower CER than WER (Large-v3 micro CER 0.278 vs micro WER 0.748), consistent with same-script spelling/word-boundary errors rather than a script mismatch. Hindi is lower on both measures.

As a secondary sensitivity check, the audit applied OpenAI Whisper’s `BasicTextNormalizer` with `remove_diacritics=False` and `split_letters=False`, copied and preserved as [openai_whisper_basic.py](./openai_whisper_basic.py) from the official source at commit `86098128c0b4f24f0e2aa2994de830614b474227` (`basic.py` SHA-256 `4742eaa040e0657fa1247a1361e0d856c62317a43326ea59a40c2e9edd8d2c38`). Mean WER changed as follows:

| model / language | study mean WER | BasicTextNormalizer mean WER | change |
|---|---:|---:|---:|
| Large-v3 / Assamese | 1.136 | 1.093 | −0.043 |
| Large-v3 / Bengali | 0.752 | 0.429 | −0.323 |
| Large-v3 / Hindi | 0.289 | 0.171 | −0.118 |
| Turbo / Assamese | 1.337 | 1.140 | −0.197 |
| Turbo / Bengali | 0.866 | 0.528 | −0.338 |
| Turbo / Hindi | 0.324 | 0.199 | −0.125 |

The apparent reduction is not evidence of improved Indic ASR. OpenAI’s `remove_symbols` path replaces every Unicode category beginning with `M`, `S`, or `P` with a space even when `remove_diacritics=False`; this removes Indic vowel signs and other combining marks. For example, the exact Basic normalizer maps Hindi `किताबें` to `क त ब `, Bengali `বাংলা ভালো` to `ব ল ভ ল `, and `हिंदी में` to `ह द म `. The study scorer intentionally preserves those marks and scripts. BasicTextNormalizer is distinct from Whisper’s English normalizer, but it is still unsafe as an Indic metric without language-specific validation.

The scoring implementation therefore does not plausibly explain the extreme results by itself. It permits WER above 1 correctly, reproduces all stored metrics, and shows that Assamese errors are dominated by cross-script output and repetition. Normalization choices can affect Bengali and Hindi substantially, so reported comparisons must keep the scorer fixed; an independent native decoder check remains the right way to test whether the model outputs themselves are at fault.
