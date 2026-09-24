"""Small deterministic decoding diagnostic, not a benchmark result.

Exit 1 reproduces missing raw EOS on a fixed public 30s development clip.
Compression is descriptive only: even the human Hindi reference exceeds 2.4.
A passing termination diagnostic does not establish ASR accuracy.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import time
import zlib
from pathlib import Path

import soundfile as sf
import torch
import transformers
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from generation_output import generation_rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--beams', type=int, default=1)
    parser.add_argument('--timestamps', action='store_true')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        raise SystemExit('Refuse to overwrite diagnostic evidence')
    assert transformers.__version__ == '4.57.6'
    torch.set_num_threads(4)
    torch.manual_seed(12)
    model_id = 'openai/whisper-large-v3-turbo'
    revision = '41f01f3fe87f28c78e2fbf8b568835947dd65ed9'
    audio_path = Path(__file__).resolve().parents[2] / 'data/bridge_open/humyn_public_v1/humyn-13-00.wav'
    audio, sr = sf.read(audio_path, frames=480000, dtype='float32')
    assert sr == 16000 and len(audio) == 480000
    processor = WhisperProcessor.from_pretrained(model_id, revision=revision, local_files_only=True)
    model = WhisperForConditionalGeneration.from_pretrained(model_id, revision=revision,
                local_files_only=True, torch_dtype=torch.float32,
                attn_implementation='sdpa').to('mps').eval()
    model.generation_config.forced_decoder_ids = None
    inputs = processor(audio, sampling_rate=sr, return_tensors='pt', return_attention_mask=True)
    started = time.monotonic()
    with torch.inference_mode():
        result = model.generate(input_features=inputs.input_features.to('mps'),
                    attention_mask=inputs.attention_mask.to('mps'), language='hi', task='transcribe',
                    do_sample=False, num_beams=args.beams, return_timestamps=args.timestamps,
                    return_dict_in_generate=True, max_new_tokens=440)
    torch.mps.synchronize()
    decoded = generation_rows(result, model.config.eos_token_id)[0]
    tokens = decoded['token_ids']
    text = processor.decode(tokens, skip_special_tokens=True)
    encoded = text.encode('utf-8')
    compression_ratio = len(encoded) / len(zlib.compress(encoded))
    eos = decoded['eos_present']
    capped = not eos
    passed = eos
    report = dict(scope='fixed public 30-second engineering diagnostic, not held-out evaluation',
        model_id=model_id, model_revision=revision, clip='humyn-13-00:0-30',
        clip_float32_sha256=hashlib.sha256(audio.tobytes()).hexdigest(),
        code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        transformers=transformers.__version__, torch=torch.__version__,
        beams=args.beams, timestamps=args.timestamps, tokens=tokens, text=text,
        generation_sequences=decoded['generation_sequences'],
        pass_criterion='every raw generation ended in EOS; compression is descriptive',
        compression_ratio=compression_ratio, eos_present=eos, capped=capped,
        elapsed_seconds=time.monotonic()-started, diagnostic_pass=passed)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='tokens'}, ensure_ascii=False), flush=True)
    return 0 if passed else 1


if __name__ == '__main__':
    raise SystemExit(main())
