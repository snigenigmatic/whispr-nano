from publication_study.alignment_repair import repair_zero_duration,duration_character_cap


def test_supported_repetition_is_preserved():
    text='stop stop stop stop stop stop stop stop'
    words=[dict(word='stop',start=i,end=i+.5) for i in range(8)]
    assert repair_zero_duration(text,words)==(text,{'status':'unchanged','removed':0})


def test_only_zero_span_copies_in_long_run_are_removed():
    text='please stop stop stop stop stop now'
    words=[dict(word=w,start=i,end=i+(.3 if i in [0,1,3,6] else 0)) for i,w in enumerate(text.split())]
    repaired,meta=repair_zero_duration(text,words)
    assert repaired=='please stop stop now' and meta['removed']==3


def test_mapping_disagreement_abstains():
    text='go go go go go'
    assert repair_zero_duration(text,[dict(word='gogo',start=0,end=0)])[0]==text


def test_short_legitimate_repeat_never_touched_even_with_bad_alignment():
    text='no no no'
    assert repair_zero_duration(text,[dict(word='no',start=0,end=0) for _ in range(3)])[0]==text


def test_cap_ends_on_word_boundary_without_reading_references():
    assert duration_character_cap('one two three',.15)=='one two'
    assert duration_character_cap('stop stop stop stop stop',2)=='stop stop stop stop stop'
