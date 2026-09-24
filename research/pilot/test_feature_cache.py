from pathlib import Path
from types import SimpleNamespace
import ast
import json

import numpy as np
import pytest

from research.pilot import feature_cache as cache


class FakeExtractor:
    feature_size = 80
    nb_max_frames = 3000
    def __init__(self):
        self.calls = []
    def to_dict(self):
        return {'feature_size':80, 'sampling_rate':16000}
    def __call__(self, audio, **kwargs):
        self.calls.append((audio.copy(), kwargs))
        return SimpleNamespace(input_features=np.full((1,80,3000), audio.mean(), dtype=np.float32))


@pytest.fixture
def prepared(tmp_path):
    root = tmp_path/'audio'; root.mkdir(); (root/'a.wav').write_bytes(b'test-audio-identity')
    rows = [dict(utt_id='a',audio_path='a.wav',duration_s=1.0,split='train')]
    extractor = FakeExtractor()
    # Use a stereo fixture to exercise identical arithmetic downmix.
    audio = np.stack([np.ones(16000,dtype=np.float32),np.zeros(16000,dtype=np.float32)],axis=1)
    sf = SimpleNamespace(read=lambda *args,**kwargs:(audio,16000))
    scipy = SimpleNamespace(signal=SimpleNamespace(resample_poly=lambda *args:pytest.fail('unexpected resampling')))
    versions = {'torch':'fixture','transformers':'5.16.1','numpy':'fixture','scipy':'fixture','soundfile':'fixture'}
    return dict(rows=rows,audio_root=root,cache_dir=tmp_path/'cache',processor=SimpleNamespace(feature_extractor=extractor),
                np=np,sf=sf,scipy=scipy,versions=versions)


def test_prepares_same_key_and_safely_reuses(prepared):
    first = cache._prepare(**prepared)
    assert first['rows']==1 and first['created_features']==1
    assert prepared['processor'].feature_extractor.calls[0][0].mean()==0.5
    record = first['files'][0]
    assert record['feature_file']==cache.canonical_hash([record['audio_sha256'],first['feature_signature']])+'.npy'
    second = cache._prepare(**prepared)
    assert second['reused_features']==1 and second['files']==first['files']
    assert len(prepared['processor'].feature_extractor.calls)==1
    assert not list(Path(prepared['cache_dir']).glob('*.tmp'))


@pytest.mark.parametrize('split',['test','test_shared','dev_shared'])
def test_forbids_final_test_and_diagnostics_before_model_import(prepared,split):
    prepared['rows'][0]['split']=split
    with pytest.raises(ValueError,match='prohibited'):
        cache.prepare(prepared['rows'],prepared['audio_root'],prepared['cache_dir'])


def test_duplicate_ids_rejected(prepared):
    prepared['rows'].append(dict(prepared['rows'][0]))
    with pytest.raises(ValueError,match='duplicate'):cache._prepare(**prepared)


def test_path_escape_rejected(prepared):
    prepared['rows'][0]['audio_path']='../outside.wav'
    with pytest.raises(ValueError,match='escapes'):cache._prepare(**prepared)


@pytest.mark.parametrize('bad_array',[np.zeros((80,2),dtype=np.float32),np.zeros((80,3000),dtype=np.float64),
                                     np.full((80,3000),np.nan,dtype=np.float32)])
def test_corrupt_existing_cache_fails_without_replacement(prepared,bad_array):
    receipt=cache._prepare(**prepared);path=Path(prepared['cache_dir'])/receipt['files'][0]['feature_file']
    np.save(path,bad_array);before=cache.sha256_file(path)
    with pytest.raises(ValueError,match='Invalid cached'):cache._prepare(**prepared)
    assert cache.sha256_file(path)==before


def test_audio_hash_or_duration_mismatch_rejected(prepared):
    prepared['rows'][0]['audio_sha256']='bad'
    with pytest.raises(ValueError,match='checksum'):cache._prepare(**prepared)
    prepared['rows'][0].pop('audio_sha256');prepared['rows'][0]['duration_s']=2.0
    with pytest.raises(ValueError,match='duration mismatch'):cache._prepare(**prepared)


def test_signature_payload_keys_and_literals_match_frozen_training_source():
    tree=ast.parse((Path(__file__).with_name('training.py')).read_text())
    assignments=[node for node in ast.walk(tree) if isinstance(node,ast.Assign)
                 and any(isinstance(t,ast.Name) and t.id=='feature_signature' for t in node.targets)]
    assert len(assignments)==1
    payload=assignments[0].value.args[0]
    keys=[node.value for node in payload.keys]
    values=dict(zip(keys,payload.values))
    versions={'transformers':'5.16.1','numpy':'n','scipy':'s','soundfile':'f'}
    expected={'revision':cache.MODEL_REVISION,**versions,'downmix':values['downmix'].value,
              'resampling':values['resampling'].value,'feature_extractor':{'test':1}}
    assert set(keys)==set(expected)
    assert cache.feature_signature(versions,{'test':1})==cache.canonical_hash(expected)
