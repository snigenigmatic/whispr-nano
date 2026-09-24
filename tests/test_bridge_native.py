import copy
from research.bridge_open.validate_run import validate_native_trace


def fixture():
    return dict(total_feature_frames=6500,feature_hop_samples=160,
        token_ids=[1,2,3], generation_sequences=[[10,50257],[20,50257],[30,50257]],
        native_trace=[dict(start_frame=0,input_end_frame=3000,next_seek_frame=2700,encoder_frames=3000,generation_sequence=[10,50257]),
                      dict(start_frame=2700,input_end_frame=5700,next_seek_frame=5600,encoder_frames=3000,generation_sequence=[20,50257]),
                      dict(start_frame=5600,input_end_frame=6500,next_seek_frame=6500,encoder_frames=3000,generation_sequence=[30,50257])],
        native_segments=[dict(token_ids=[1,2]),dict(token_ids=[3])])


def test_native_allows_adaptive_overlap_but_requires_full_coverage():
    x=fixture()
    assert validate_native_trace(x,65)==[]
    x['native_trace'][-1]['next_seek_frame']=6499
    assert 'native decoding did not reach end of audio' in validate_native_trace(x,65)


def test_native_rejects_gap_stall_and_missing_raw_evidence():
    for field,value in [('start_frame',2800),('next_seek_frame',2700),('encoder_frames',1500)]:
        x=fixture();x['native_trace'][1][field]=value
        assert validate_native_trace(x,65)
    x=fixture();x['generation_sequences'][0]=[99]
    assert validate_native_trace(x,65)
    x=fixture();x['token_ids'].append(123)
    assert validate_native_trace(x,65)


def test_fallback_decisions_are_recomputed_and_selected_output_verified():
    import math,zlib
    x=fixture();cfg=dict(compression_ratio_threshold=1.35,temperature=[0.0,.2])
    def attempt(tokens,temp,raw):
        data=b''.join(t.to_bytes(2,'little') for t in tokens)
        ratio=len(data)/len(zlib.compress(data))
        return dict(check_token_ids=tokens,vocab_size=51866,compression_ratio=ratio,
                    needs_fallback=ratio>1.35,temperature=temp,should_skip=False,generation_sequence=raw)
    for step in x['native_trace']:
        step['attempts']=[attempt([100]*80,0.0,[100]*80),attempt(list(range(100,110)),.2,step['generation_sequence'])]
    assert validate_native_trace(x,65,cfg)==[]
    bad=copy.deepcopy(x);bad['native_trace'][0]['attempts'][0]['needs_fallback']=False
    assert 'fallback decision mismatch' in validate_native_trace(bad,65,cfg)
    bad=copy.deepcopy(x);bad['native_trace'][0]['attempts'][-1]['generation_sequence']=[999]
    assert 'fallback selected output mismatch' in validate_native_trace(bad,65,cfg)
    bad=copy.deepcopy(x);bad['native_trace'][0]['attempts']=bad['native_trace'][0]['attempts'][:1]
    assert 'fallback schedule terminated early' in validate_native_trace(bad,65,cfg)
