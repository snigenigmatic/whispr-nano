from research.bridge_open.generation_output import generation_rows


def test_timestamp_display_omits_eos_but_raw_generation_terminated():
    result = {'sequences':[50258, 100, 50257]}
    output = {'sequences':[[50364, 100, 50414]],
              'segments':[[{'result':result}, {'result':result}]]}
    row = generation_rows(output, 50257)[0]
    assert row['eos_present'] is True
    assert row['generation_sequences'] == [[50258, 100, 50257]]
    assert 50257 not in row['token_ids']


def test_one_terminated_segment_does_not_hide_later_cap():
    output = {'sequences':[[50364, 100, 50414, 101]],
              'segments':[[{'result':{'sequences':[50258, 100, 50257]}},
                           {'result':{'sequences':[50258, 101, 101]}}]]}
    assert generation_rows(output, 50257)[0]['eos_present'] is False


def test_plain_raw_output_preserves_termination_and_each_batch_row():
    rows = generation_rows({'sequences':[[50258, 100, 50257], [50258, 101, 101]]}, 50257)
    assert [r['eos_present'] for r in rows] == [True, False]
