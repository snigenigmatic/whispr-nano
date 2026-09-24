"""Retain true termination evidence for both Whisper return formats."""


def _list(value):
    if hasattr(value, 'detach'):
        value = value.detach().cpu()
    return value.tolist() if hasattr(value, 'tolist') else list(value)


def generation_rows(output, eos_token_id):
    """Separate displayed tokens from each underlying generation's raw tokens.

    Timestamp-mode displayed sequences strip EOS. Segment results retain it.
    Several timestamp segments can reference the same underlying generate call.
    """
    displayed = _list(output['sequences'])
    rows = []
    for index, tokens in enumerate(displayed):
        if 'segments' in output:
            raw, seen = [], set()
            for segment in output['segments'][index]:
                result = segment['result']
                if id(result) not in seen:
                    seen.add(id(result))
                    raw.append(_list(result['sequences']))
            if not raw:
                raise ValueError('Timestamp output has no raw termination evidence')
        else:
            raw = [tokens]
        rows.append({'token_ids':tokens, 'generation_sequences':raw,
                     'eos_present':all(eos_token_id in seq for seq in raw)})
    return rows
