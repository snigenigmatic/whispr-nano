"""Native Whisper long-form decoding with a trace of real input consumption."""
import time


def as_list(value):
    return value.detach().cpu().tolist() if hasattr(value, 'detach') else list(value)


def decode_native(model, processor, audio, sr, cfg, language, started):
    import torch
    inputs = processor(audio, sampling_rate=sr, truncation=False, padding='longest',
                       return_attention_mask=True, return_tensors='pt')
    total_frames = inputs.input_features.shape[-1]
    trace = []
    original_input = model._get_input_segment
    original_segment = model._retrieve_segment
    original_fallback = model._need_fallback

    def capture_input(**kwargs):
        if time.monotonic() - started > cfg['max_wall_seconds']:
            raise TimeoutError('Native decoding wall-time limit reached')
        if kwargs['cur_bsz'] != 1:
            raise ValueError('Native evidence contract requires batch1')
        start = int(kwargs['seek'][0])
        valid = int(kwargs['seek_num_frames'][0])
        trace.append(dict(start_frame=start, input_end_frame=start+valid))
        result = original_input(**kwargs)
        trace[-1]['encoder_frames'] = result.shape[-1]
        return result

    def capture_segment(**kwargs):
        segments, offset = original_segment(**kwargs)
        trace[-1]['next_seek_frame'] = trace[-1]['start_frame'] + int(offset)
        trace[-1]['generation_sequence'] = as_list(kwargs['seek_outputs'][kwargs['idx']]['sequences'])
        return segments, offset

    def capture_fallback(sequence, outputs, index, logits_processor, generation_config, vocab_size, temperature):
        result = original_fallback(sequence, outputs, index, logits_processor, generation_config, vocab_size, temperature)
        trace[-1].setdefault('attempts', []).append(dict(
            temperature=float(temperature), check_token_ids=as_list(sequence),
            effective_num_beams=generation_config.num_beams,
            effective_do_sample=generation_config.do_sample,
            vocab_size=vocab_size,
            compression_ratio=float(model._retrieve_compression_ratio(sequence,vocab_size)),
            needs_fallback=bool(result[0]), should_skip=bool(result[1]),
            generation_sequence=as_list(outputs[index]['sequences'])))
        return result

    model._get_input_segment = capture_input
    model._retrieve_segment = capture_segment
    model._need_fallback = capture_fallback
    try:
        with torch.inference_mode():
            out = model.generate(input_features=inputs.input_features.to(cfg['device']),
                attention_mask=inputs.attention_mask.to(cfg['device']), language=language,
                task='transcribe', return_timestamps=True, return_segments=True,
                return_dict_in_generate=True, do_sample=False, num_beams=cfg['num_beams'],
                max_new_tokens=cfg['max_new_tokens'], temperature=cfg['temperature'],
                condition_on_prev_tokens=False, compression_ratio_threshold=cfg['compression_ratio_threshold'],
                logprob_threshold=None, no_speech_threshold=None)
    finally:
        model._get_input_segment = original_input
        model._retrieve_segment = original_segment
        model._need_fallback = original_fallback
    if not trace or trace[-1].get('next_seek_frame', -1) < total_frames:
        raise ValueError('Native decoder did not consume all feature frames')
    tokens = as_list(out['sequences'][0])
    raw = [step['generation_sequence'] for step in trace]
    return dict(start_seconds=0.0, end_seconds=len(audio)/sr,
        text_raw=processor.decode(tokens, skip_special_tokens=True), token_ids=tokens,
        generation_sequences=raw, eos_present=all(model.config.eos_token_id in seq for seq in raw),
        native_trace=trace, total_feature_frames=total_frames,
        feature_hop_samples=processor.feature_extractor.hop_length,
        native_segments=[dict(start_seconds=float(s['start']),end_seconds=float(s['end']),
                              token_ids=as_list(s['tokens'])) for s in out['segments'][0]])
