# Qwen3-ASR-0.6B local feasibility

Checked 2026-09-22 for the cached [`Qwen/Qwen3-ASR-0.6B` model card](https://huggingface.co/Qwen/Qwen3-ASR-0.6B) and the [official Qwen3-ASR repository](https://github.com/QwenLM/Qwen3-ASR). This is a source review only; no package was installed and no model was loaded.

## Documented runtime

The cached card recommends the `qwen-asr` package and gives this transformers API:

```python
import torch
from qwen_asr import Qwen3ASRModel

model = Qwen3ASRModel.from_pretrained(
    "Qwen/Qwen3-ASR-0.6B",
    dtype=torch.bfloat16,
    device_map="cuda:0",
    max_inference_batch_size=32,
    max_new_tokens=256,
)
results = model.transcribe(audio="path-or-url.wav", language=None)
text = results[0].text
```

The official repository currently declares `qwen-asr==0.0.6` and pins its transformers backend to `transformers==4.57.6`, with `accelerate==1.12.0`, `qwen-omni-utils`, `librosa`, `soundfile`, `sox`, and other utility/UI dependencies. The optional vLLM extra is `vllm==0.14.0`. These versions are from the repository's `pyproject.toml`, not from an installed environment.

The source implementation is `qwen_asr/inference/qwen3_asr.py`. `Qwen3ASRModel.from_pretrained()` forwards arbitrary keyword arguments to `AutoModel.from_pretrained()` and loads an `AutoProcessor`; `transcribe(audio, language, return_time_stamps=False)` accepts a local path, URL, base64 data URL, or `(numpy_array, sample_rate)` tuple. Its transformers path calls the model's normal `generate()` and moves processor inputs to `self.model.device`. The package exports `Qwen3ASRModel` from `qwen_asr`.

## CPU and MPS assessment

CPU is technically plausible through the transformers backend: the source does not force a CUDA device, and `from_pretrained()` forwards device/dtype choices to Transformers. The source even falls back to `torch.device("cpu")` when it cannot infer a model device. However, the official examples only document `device_map="cuda:0"`; CPU support is therefore an inference from the generic Transformers path, not a Qwen-tested support claim. A CPU attempt should use float32 and an explicit CPU placement, for example `device_map="cpu", dtype=torch.float32`, subject to memory and speed checks.

MPS is also not documented by Qwen. The transformers wrapper has no literal `cuda` requirement in its load or inference path, so a manually placed PyTorch model may be able to run on MPS. This remains unverified: Accelerate device-map handling, model generation operators, and the Qwen audio/model kernels need a real smoke test on the target macOS/PyTorch stack. The documented examples' `torch.bfloat16` and FlashAttention 2 settings should not be copied to MPS; use float32 first. No official source reviewed here promises MPS compatibility.

## Actual CUDA blockers

The optional vLLM backend is a CUDA-oriented route. The official examples initialize it with `Qwen3ASRModel.LLM(...)`, GPU memory settings, and CUDA device selection; the repository's local demo also exposes `--cuda-visible-devices`. vLLM should therefore be treated as unavailable for this macOS MPS/CPU pilot.

FlashAttention 2 is optional, explicitly recommended for compatible hardware, and is not required by the transformers backend. It is a CUDA-oriented optimization and should be omitted locally. The standard transformers backend itself contains no hardcoded `torch.cuda`, `cuda:0`, or mandatory CUDA assertion in the reviewed inference source; the `cuda:0` values are example arguments in the documentation and demo defaults.

## Practical decision

Qwen3-ASR-0.6B is a reasonable post-Whisper local candidate only via the transformers backend, with CPU as the safer documented-by-generic-Transformers inference and MPS as an unverified experiment. It is not a drop-in vLLM candidate on this Mac. A future bounded smoke test should pin `qwen-asr==0.0.6`/`transformers==4.57.6`, use float32, avoid FlashAttention and vLLM, load the exact 0.6B checkpoint, and record whether model loading and one Hindi transcription complete on CPU and MPS.

Primary source links:

- [Qwen/Qwen3-ASR-0.6B model card](https://huggingface.co/Qwen/Qwen3-ASR-0.6B)
- [Qwen3-ASR README and quickstart](https://github.com/QwenLM/Qwen3-ASR/blob/main/README.md)
- [Official package metadata and pinned dependencies](https://github.com/QwenLM/Qwen3-ASR/blob/main/pyproject.toml)
- [Transformers inference wrapper](https://github.com/QwenLM/Qwen3-ASR/blob/main/qwen_asr/inference/qwen3_asr.py)
- [Transformers model implementation](https://github.com/QwenLM/Qwen3-ASR/blob/main/qwen_asr/core/transformers_backend/modeling_qwen3_asr.py)

