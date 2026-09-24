"""Real-evidence negative checks; originals are never modified."""
import copy
from unittest.mock import patch
import pytest
from research.bridge_open.core_study import accept_qwen06 as a

RUN=a.ART/'core_qwen06_svarah_v1'

def test_real_completed_svarah():
    result=a.validate(RUN,'svarah')
    assert result['n']==518 and result['word_errors']==883 and result['reference_words']==5389

@pytest.mark.parametrize('kind',['wrong_model','corrupt_summary','missing_prediction','wrong_eos'])
def test_reject_real_evidence_corruption(kind):
    original_read,original_rows=a.read,a.rows
    def changed_read(path):
        obj=original_read(path)
        if str(path)==str(RUN/'config.json') and kind=='wrong_model':obj['model_id']='incorrect/model'
        if str(path)==str(RUN/'summary.json') and kind=='corrupt_summary':obj['per_language']['en']['word_errors']+=1
        return obj
    def changed_rows(path):
        records=original_rows(path)
        if str(path)==str(RUN/'predictions.jsonl'):
            if kind=='missing_prediction': records.pop()
            if kind=='wrong_eos':records[0]['chunks'][0]['generation_trace'][0]['eos_present']=False
        return records
    with patch.object(a,'read',changed_read),patch.object(a,'rows',changed_rows):
        with pytest.raises(ValueError):a.validate(RUN,'svarah')
