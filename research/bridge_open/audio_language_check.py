"""Reference-free CPU diagnostic of audio scale and language-ID predictions."""
import hashlib
import json
from pathlib import Path
import numpy as np
import soundfile as sf
import torch
from huggingface_hub import snapshot_download
from transformers import WhisperForConditionalGeneration, WhisperProcessor

ROOT=Path(__file__).resolve().parents[2]


def digest(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    cfg=json.loads((ROOT/'research/bridge_open/run_config.json').read_text())
    manifest=Path(cfg['manifest_path'])
    assert digest(manifest)==cfg['manifest_sha256']
    by_id={r['conversation_id']:r for r in map(json.loads,manifest.read_text().splitlines())}
    torch.set_num_threads(4);torch.manual_seed(12)
    snapshot=Path(snapshot_download(cfg['model_id'],revision=cfg['model_revision'],token=False,
        allow_patterns=['*.json','*.txt','*.safetensors']))
    processor=WhisperProcessor.from_pretrained(snapshot,local_files_only=True)
    model=WhisperForConditionalGeneration.from_pretrained(snapshot,local_files_only=True,
        torch_dtype=torch.float32,attn_implementation='sdpa').cpu().eval()
    langs=model.generation_config.lang_to_id
    names=list(langs);ids=[langs[k] for k in names]
    rows=[]
    for cid in cfg['conversation_ids']:
        source=by_id[cid];assert digest(source['audio_path'])==source['audio_sha256']
        audio,sr=sf.read(source['audio_path'],dtype='float32');assert sr==16000 and audio.ndim==1
        assert np.isfinite(audio).all()
        for label,offset in [('first',0),('middle',len(audio)//2)]:
            samples=audio[offset:offset+30*sr]
            features=processor(samples,sampling_rate=sr,return_tensors='pt').input_features
            with torch.inference_mode():
                output=model(input_features=features,decoder_input_ids=torch.tensor([[model.config.decoder_start_token_id]]))
            prob=torch.softmax(output.logits[0,-1,ids],dim=-1)
            order=torch.argsort(prob,descending=True)[:5].tolist()
            row=dict(conversation_id=cid,released_language=source['language'],window=label,start_seconds=offset/sr,
                duration_seconds=len(samples)/sr,rms=float(np.sqrt(np.mean(samples.astype(np.float64)**2))),
                absolute_peak=float(np.abs(samples).max()),clipped_fraction=float(np.mean(np.abs(samples)>=0.999)),
                top5_language_token_probabilities=[dict(language=names[i],probability=float(prob[i])) for i in order],
                audio_sha256=source['audio_sha256'])
            rows.append(row);print(json.dumps(dict(event='language_id_diagnostic',**row)),flush=True)
    report=dict(scope='Reference-free diagnostic; model language probabilities are not calibrated human adjudication.',
        config=cfg,model_weight_sha256={p.name:digest(p) for p in snapshot.glob('*.safetensors')},
        code_sha256=digest(__file__),torch_version=torch.__version__,cloud_cost_usd=0,rows=rows)
    out=ROOT/cfg['analysis_output'];out.mkdir(parents=True,exist_ok=False)
    (out/'evidence.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(event='final_audio_language_check',**report)),flush=True)
    return 0
