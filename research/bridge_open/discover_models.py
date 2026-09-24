"""Snapshot official ASR model metadata without downloading model weights.

Candidate IDs were identified through live HF catalog, publisher and Exa searches.
Repository creation/update dates are not model release dates or quality rankings.
"""
import concurrent.futures
import datetime
import hashlib
import json
from pathlib import Path

import requests

MODEL_IDS = [
    'openai/whisper-tiny', 'openai/whisper-small', 'openai/whisper-large-v3',
    'openai/whisper-large-v3-turbo',
    'Qwen/Qwen3-ASR-0.6B', 'Qwen/Qwen3-ASR-1.7B',
    'ai4bharat/indic-conformer-600m-multilingual',
    'ai4bharat/indic-asr-nemotron-600m',
    'bodhan-ai/indic-transcribe-core', 'bodhan-ai/indic-transcribe-flex',
    'facebook/omniASR-CTC-1B', 'facebook/omniASR-LLM-7B',
    'microsoft/VibeVoice-ASR',
    'CohereLabs/cohere-transcribe-03-2026',
    'ibm-granite/granite-speech-5.0-470m-turboctc',
    'ibm-granite/granite-speech-4.1-2b',
    'ibm-granite/granite-speech-4.1-2b-plus',
    'nvidia/nemotron-3.5-asr-streaming-0.6b',
    'nvidia/parakeet-tdt-0.6b-v3', 'nvidia/canary-1b-v2',
    'nvidia/canary-qwen-2.5b',
    'mistralai/Voxtral-Mini-4B-Realtime-2602',
    'mistralai/Voxtral-Mini-3B-2507',
    'FunAudioLLM/Fun-ASR-MLT-Nano-2512',
    'FunAudioLLM/SenseVoiceSmall', 'zai-org/GLM-ASR-Nano-2512',
    'moonshine-ai/moonshine-streaming-medium',
    'kyutai/stt-2.6b-en',
]


def fetch(model_id):
    response = requests.get(f'https://huggingface.co/api/models/{model_id}', timeout=35)
    if not response.ok:
        return dict(model_id=model_id, metadata_http_status=response.status_code)
    data = response.json()
    card = data.get('cardData') or {}
    revision = data.get('sha')
    result = dict(model_id=model_id, revision=revision,
                  source=f'https://huggingface.co/{model_id}',
                  metadata_http_status=200, created_at=data.get('createdAt'),
                  last_modified=data.get('lastModified'), gated=data.get('gated'),
                  license_tag=card.get('license'), license_name=card.get('license_name'),
                  declared_language_tags=card.get('language'),
                  pipeline_tag=data.get('pipeline_tag'), library_name=data.get('library_name'),
                  safetensors=data.get('safetensors'), downloads=data.get('downloads'),
                  files=[x['rfilename'] for x in data.get('siblings',[])])
    readme = requests.get(f'https://huggingface.co/{model_id}/resolve/{revision}/README.md', timeout=35)
    result['card_http_status'] = readme.status_code
    if readme.ok:
        cache = Path('.cache/bridge_open/model_cards')
        cache.mkdir(parents=True, exist_ok=True)
        dest = cache / (model_id.replace('/','__') + '.md')
        dest.write_bytes(readme.content)
        result['card_sha256'] = hashlib.sha256(readme.content).hexdigest()
        result['card_cache_path'] = str(dest)
    return result


if __name__ == '__main__':
    out = Path('research/bridge_open/artifacts/model_catalog_20260922.json')
    if out.exists():
        raise SystemExit('Snapshot already exists; preserve it and choose a new dated file.')
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
        models = list(pool.map(fetch, MODEL_IDS))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(dict(checked_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                                  scope='bounded official-release candidate catalog; not every HF upload or verified ranking',
                                  weights_downloaded=False, models=models), indent=2) + '\n')
    for row in models:
        print(row['model_id'], row.get('metadata_http_status'), row.get('license_tag'),
              'gated=' + str(row.get('gated')), 'languages=' + str(row.get('declared_language_tags')))
