import pytest
from research.bridge_open.scoring import score
from research.bridge_open.study_panels import aggregate_errors, oracle_panel


def row(cid, ref, hyp, language='hi'):
    return dict(conversation_id=cid,language=language,metrics=score(ref,hyp))


def test_error_rates_reconstruct_wer_and_do_not_clip_insertions():
    rows=[row('a','one','one two three'),row('b','four five','')]
    got=aggregate_errors(rows)
    assert got['micro_wer']==pytest.approx(4/3)
    assert got['substitution_rate']+got['deletion_rate']+got['insertion_rate']==pytest.approx(got['micro_wer'])
    assert got['pooled_wil']==pytest.approx(1-1/9)


def test_oracle_uses_pooled_counts_not_mean_per_file_wer():
    ref='one two three four five six seven eight nine'
    models={'a':[row('short','one','one'),row('long',ref,'')],
            'b':[row('short','one',''),row('long',ref,ref)]}
    got=oracle_panel(models)
    assert got['fixed_model']=='b'
    assert got['fixed_model_micro_wer']==pytest.approx(.1)
    assert got['language_oracle_micro_wer']==pytest.approx(.1)
    assert got['conversation_oracle_micro_wer']==0


def test_oracle_refuses_unequal_cohort_and_reference_lengths():
    with pytest.raises(ValueError,match='conversation sets'):
        oracle_panel({'a':[row('a','one','one')],'b':[row('b','one','one')]})
    with pytest.raises(ValueError,match='denominators'):
        oracle_panel({'a':[row('a','one','one')],'b':[row('a','one two','one')]})
