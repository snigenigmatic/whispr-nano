"""Termination observations from raw Qwen generation, accounting for batch padding."""
def token_termination(tokens, eos_ids, max_new_tokens):
    first_eos = next((i for i, token in enumerate(tokens) if token in eos_ids), None)
    return dict(new_token_counts=len(tokens) if first_eos is None else first_eos + 1,
                eos_present=first_eos is not None,
                cap_observed=first_eos is None and len(tokens) >= max_new_tokens)


def validate_trace(trace, cfg):
    errors = []
    if not isinstance(trace, list) or len(trace) != 1:
        return ['requires exactly one native generation observation']
    t = trace[0]
    if t.get('trace_schema') != 2 or t.get('observation_status') != 'captured':
        return ['invalid trace schema/status']
    tokens, raw, prompt = t.get('new_token_sequences'), t.get('raw_sequences'), t.get('prompt_tokens')
    if not isinstance(tokens, list) or not tokens or any(type(v) is not int for v in tokens):
        return ['invalid continuation tokens']
    if not isinstance(prompt, int) or prompt < 1 or not isinstance(raw, list) or raw[prompt:] != tokens:
        errors.append('prompt/continuation mismatch')
    expected = token_termination(tokens, cfg['eos_token_ids'], cfg['max_new_tokens'])
    for key, value in expected.items():
        if t.get(key) != value:
            errors.append('termination mismatch: ' + key)
    if t.get('max_new_tokens') != cfg['max_new_tokens'] or len(tokens) > cfg['max_new_tokens']:
        errors.append('generation token limit mismatch')
    if not isinstance(t.get('raw_decoded_text'), str):
        errors.append('raw decoded text missing')
    return errors
