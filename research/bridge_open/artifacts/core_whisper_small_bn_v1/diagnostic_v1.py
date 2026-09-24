"""Independent Whisper implementation audit; diagnostic examples, not a benchmark."""
import copy
import hashlib
import importlib.metadata
import inspect
import json
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from research.bridge_open.artifact_transport import emit
from research.bridge_open.scoring import score, normalize_v1


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def hf_key(native_key):
    replacements = [
        ('.blocks.', '.layers.'), ('.cross_attn_ln.', '.encoder_attn_layer_norm.'),
        ('.attn_ln.', '.self_attn_layer_norm.'), ('.mlp_ln.', '.final_layer_norm.'),
        ('.cross_attn.query.', '.encoder_attn.q_proj.'),
        ('.cross_attn.key.', '.encoder_attn.k_proj.'),
        ('.cross_attn.value.', '.encoder_attn.v_proj.'),
        ('.cross_attn.out.', '.encoder_attn.out_proj.'),
        ('.attn.query.', '.self_attn.q_proj.'), ('.attn.key.', '.self_attn.k_proj.'),
        ('.attn.value.', '.self_attn.v_proj.'), ('.attn.out.', '.self_attn.out_proj.'),
        ('.mlp.0.', '.fc1.'), ('.mlp.2.', '.fc2.'),
        ('.ln_post.', '.layer_norm.'), ('decoder.ln.', 'decoder.layer_norm.'),
        ('.token_embedding.', '.embed_tokens.'),
    ]
    for old, new in replacements:
        native_key = native_key.replace(old, new)
    if native_key.endswith('.positional_embedding'):
        native_key = native_key.replace('.positional_embedding', '.embed_positions.weight')
    return native_key


def main():
    import numpy as np
    import soundfile as sf
    import torch
    import whisper
    from huggingface_hub import snapshot_download
    from transformers import WhisperProcessor, WhisperForConditionalGeneration, pipeline

    start = time.monotonic()
    cfg = json.loads((ROOT/'experiments/config.json').read_text())
    rows = json.loads((ROOT/'experiments/cohort.json').read_text())
    old = {r['conversation_id']: r for r in json.loads((ROOT/'experiments/original_predictions.json').read_text())}
    out = ROOT/'artifacts/audit'
    out.mkdir(parents=True, exist_ok=False)
    for name in ['config.json', 'cohort.json', 'native_source_receipt.json']:
        (out/name).write_bytes((ROOT/'experiments'/name).read_bytes())
    torch.manual_seed(cfg['seed']);np.random.seed(cfg['seed']);torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    assert torch.cuda.is_available() and torch.__version__.split('+')[0] == '2.8.0'
    audio = {}
    for row in rows:
        path = ROOT/row['staged_audio']
        assert sha(path) == row['audio_sha256']
        assert sha(ROOT/row['staged_reference']) == row['reference_sha256']
        samples, sr = sf.read(path, dtype='float32')
        assert sr == 16000 and samples.ndim == 1 and np.isfinite(samples).all()
        assert 0 < np.abs(samples).max() <= 1
        audio[row['conversation_id']] = samples
    snapshot = Path(snapshot_download(cfg['model_id'], revision=cfg['model_revision'],
        allow_patterns=['*.json','*.safetensors','*.txt'], token=False))
    processor = WhisperProcessor.from_pretrained(snapshot, local_files_only=True)
    hf = WhisperForConditionalGeneration.from_pretrained(snapshot, local_files_only=True,
        torch_dtype=torch.float32, attn_implementation='sdpa').to('cuda').eval()
    hf.generation_config.forced_decoder_ids = None
    baseline = copy.deepcopy(hf.generation_config)
    native = whisper.load_model(cfg['native_model'], device='cuda', download_root=str(ROOT/'model_cache')).float().eval()
    native_checkpoint = ROOT/'model_cache'/f"{cfg['native_model']}.pt"
    expected_checkpoint_sha = whisper._MODELS[cfg['native_model']].split('/')[-2]
    actual_checkpoint_sha = sha(native_checkpoint)
    assert actual_checkpoint_sha == expected_checkpoint_sha
    assert not hf.training and not native.training
    assert processor.feature_extractor.feature_size == native.dims.n_mels == 128
    assert processor.feature_extractor.sampling_rate == 16000
    # Compare all corresponding tensors, allowing only numerical tolerance for
    # stored positional buffers; no tolerance is needed for learned parameters.
    hf_state = hf.model.state_dict(); native_state = native.state_dict()
    learned = set(dict(native.named_parameters()))
    weight_comparison = []
    for key, tensor in native_state.items():
        if tensor.is_sparse:
            continue
        mapped = hf_key(key)
        if mapped not in hf_state:
            raise ValueError(f'Native tensor has no HF counterpart: {key} -> {mapped}')
        other = hf_state[mapped]
        assert tensor.shape == other.shape, (key, tensor.shape, other.shape)
        delta = float((tensor-other).abs().max())
        tolerance = 0.0 if key in learned else 1e-5
        weight_comparison.append(dict(native=key, hf=mapped, learned=key in learned, max_abs_delta=delta))
        assert delta <= tolerance, (key, delta)
    (out/'weight_comparison.json').write_text(json.dumps(weight_comparison, indent=2)+'\n')
    assert set(hf_state) == {r['hf'] for r in weight_comparison}, 'Unaccounted HF model tensors'
    assert torch.equal(hf.proj_out.weight, hf.model.decoder.embed_tokens.weight), 'HF output projection is not tied'
    environment = dict(gpu=torch.cuda.get_device_name(0), hf_training=hf.training,
        native_training=native.training, dtype=str(next(hf.parameters()).dtype),
        native_dtype=str(next(native.parameters()).dtype), native_checkpoint_sha256=actual_checkpoint_sha,
        hf_weight_sha256={p.name:sha(p) for p in snapshot.glob('*.safetensors')},
        hf_generation_config=baseline.to_dict(), native_transcribe_signature=str(inspect.signature(native.transcribe)),
        packages={n:importlib.metadata.version(n) for n in ['torch','transformers','openai-whisper','soundfile','numpy','jiwer']},
        native_source_sha256={name:sha(Path(whisper.__file__).parent/name) for name in ['audio.py','model.py','decoding.py','transcribe.py','tokenizer.py']},
        script_sha256=sha(__file__),model_load_seconds=time.monotonic()-start)
    (out/'environment.json').write_text(json.dumps(environment,indent=2)+'\n')
    print(json.dumps({'event':'audit_models_loaded','weight_tensors_compared':len(weight_comparison),
        'learned_tensors_equal':True,'elapsed_seconds':environment['model_load_seconds']}),flush=True)

    features = []
    for row in rows:
        arr = audio[row['conversation_id']][:480000]
        hf_mel = processor(arr, sampling_rate=16000, return_tensors='pt', return_attention_mask=True).input_features[0]
        native_mel = whisper.log_mel_spectrogram(whisper.pad_or_trim(arr), n_mels=128)
        delta = float((hf_mel-native_mel).abs().max())
        features.append(dict(conversation_id=row['conversation_id'], max_abs_delta=delta,
            hf_shape=list(hf_mel.shape),native_shape=list(native_mel.shape)))
        assert delta < 1e-4, (row['conversation_id'], delta)
    (out/'feature_comparison.json').write_text(json.dumps(features,indent=2)+'\n')
    records=[]
    def check_time():
        if time.monotonic()-start > cfg['max_wall_seconds']:
            raise TimeoutError('Diagnostic wall-time guard')
    def record(arm,row,text,**extra):
        # Reference text first enters this function after inference.
        ref=json.loads((ROOT/row['staged_reference']).read_text())['reference_text']
        rec=dict(arm=arm,conversation_id=row['conversation_id'],language=row['language'],
            audio_sha256=row['audio_sha256'],reference_sha256=row['reference_sha256'],
            text=text,metrics=score(ref,text),**extra)
        records.append(rec)
        with (out/'predictions.jsonl').open('a') as stream:stream.write(json.dumps(rec,ensure_ascii=False)+'\n')
        print(json.dumps({'event':'diagnostic_prediction','arm':arm,'id':row['conversation_id'],
            'wer':rec['metrics']['wer'],'elapsed_seconds':time.monotonic()-start}),flush=True)
    def hf_decode(arrays,language):
        hf.generation_config=copy.deepcopy(baseline)
        inputs=processor(arrays,sampling_rate=16000,padding='max_length',max_length=480000,
            truncation=False,return_tensors='pt',return_attention_mask=True).to('cuda')
        with torch.inference_mode():
            result=hf.generate(**inputs,language=language,task='transcribe',num_beams=5,
                do_sample=False,return_timestamps=False,max_new_tokens=440,return_dict_in_generate=True)
        sequences=result.sequences.cpu().tolist()
        return processor.batch_decode(sequences,skip_special_tokens=True),sequences

    short=[r for r in rows if r['duration_seconds']<=30]
    for language in ['hi','bn','as']:
        check_time();group=[r for r in short if r['language']==language]
        texts,tokens=hf_decode([audio[r['conversation_id']] for r in group],language)
        for row,text,ids in zip(group,texts,tokens):
            record('hf_original_batch',row,text,tokens=ids,original_text_equal=text.strip()==old[row['conversation_id']]['text_raw'].strip())
    for row in rows:
        if row['duration_seconds']<=30:continue
        check_time();arr=audio[row['conversation_id']]
        # Original long runner groups at batch_size=1; repeat independently.
        all_text=[];all_tokens=[]
        for i in range(0,len(arr),480000):
            text,seq=hf_decode([arr[i:i+480000]],row['language']);all_text+=text;all_tokens+=seq
        record('hf_original_fixed30',row,' '.join(t.strip() for t in all_text),tokens=all_tokens)
    for row in short:
        check_time();text,tokens=hf_decode([audio[row['conversation_id']]],row['language'])
        record('hf_single',row,text[0],tokens=tokens[0])
    pipe=pipeline('automatic-speech-recognition',model=hf,tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,device=0,torch_dtype=torch.float32)
    for row in short:
        check_time();hf.generation_config=copy.deepcopy(baseline)
        with torch.inference_mode():
            result=pipe({'array':audio[row['conversation_id']].copy(),'sampling_rate':16000},
                generate_kwargs=dict(language=row['language'],task='transcribe',num_beams=5,
                    do_sample=False,return_timestamps=False,max_new_tokens=440))
        record('hf_pipeline',row,result['text'])
    for row in short:
        check_time()
        mel=whisper.log_mel_spectrogram(whisper.pad_or_trim(audio[row['conversation_id']]),n_mels=128).to('cuda')
        options=whisper.DecodingOptions(language=row['language'],task='transcribe',fp16=False,
            beam_size=5,temperature=0.0,without_timestamps=True,sample_len=440)
        with torch.inference_mode(): result=native.decode(mel,options)
        record('native_decode_matched',row,result.text,tokens=result.tokens,
            avg_logprob=result.avg_logprob,no_speech_prob=result.no_speech_prob,
            compression_ratio=result.compression_ratio,options=vars(options))
    for row in rows:
        check_time();torch.manual_seed(cfg['seed']);np.random.seed(cfg['seed'])
        with torch.inference_mode():
            result=native.transcribe(audio[row['conversation_id']].copy(),language=row['language'],
                task='transcribe',fp16=False,verbose=None)
        record('native_transcribe_default',row,result['text'],segments=result['segments'],
            options=dict(language=row['language'],task='transcribe',fp16=False,verbose=None,
                         note='All other options are official package defaults; signature captured.'))
    summary={}
    for arm in sorted({r['arm'] for r in records}):
        summary[arm]={}
        for lang in ['hi','bn','as']:
            group=[r for r in records if r['arm']==arm and r['language']==lang]
            if not group:continue
            errors=sum(r['metrics']['substitutions']+r['metrics']['deletions']+r['metrics']['insertions'] for r in group)
            words=sum(r['metrics']['ref_words'] for r in group)
            summary[arm][lang]=dict(n=len(group),word_errors=errors,ref_words=words,wer=errors/words)
    (out/'summary.json').write_text(json.dumps(dict(arms=summary,total_wall_seconds=time.monotonic()-start,
        purpose=cfg['purpose'],cost_usd=None),indent=2)+'\n')
    print(json.dumps({'event':'diagnostic_complete','summary':summary}),flush=True)


if __name__=='__main__':
    if (ROOT/'artifacts/audit').exists():
        raise SystemExit('Audit output already exists; refusing to overwrite or re-emit stale evidence')
    code=0
    try:main()
    except Exception as exc:
        code=1;traceback.print_exc();out=ROOT/'artifacts/audit';out.mkdir(parents=True,exist_ok=True)
        (out/'failure.json').write_text(json.dumps({'type':type(exc).__name__,'message':str(exc)},indent=2)+'\n')
    finally:
        out=ROOT/'artifacts/audit'
        if out.exists():emit(out)
    raise SystemExit(code)
