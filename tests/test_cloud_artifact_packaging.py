"""Protect against silently accepting partial or mismatched cloud evidence."""
import io
from contextlib import redirect_stdout
import json
from pathlib import Path
import sys
import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
sys.path.insert(0,str(ROOT/'scripts/termeval'))
from publication_study.artifact_transport import emit,recover
from package_short_hindi_study import describe


def test_incomplete_or_conflicting_exports_are_rejected(tmp_path):
    src=tmp_path/'src';src.mkdir()
    (src/'predictions.jsonl').write_text('वास्तविक शब्द\n')
    stdout=io.StringIO()
    with redirect_stdout(stdout):emit(src)
    bundle=json.loads(stdout.getvalue());bundle['count']+=1
    log=tmp_path/'partial.log';log.write_text(json.dumps(bundle))
    with pytest.raises(ValueError,match='Incomplete'):recover(log,tmp_path/'partial')
    log.write_text(stdout.getvalue());recover(log,tmp_path/'valid')
    assert (tmp_path/'valid/predictions.jsonl').read_bytes()==(src/'predictions.jsonl').read_bytes()
    bundle=json.loads(stdout.getvalue());bundle['payload']='corrupt'
    log.write_text(stdout.getvalue()+json.dumps(bundle)+'\n')
    with pytest.raises(ValueError,match='Conflicting'):recover(log,tmp_path/'conflict')


def test_silence_emissions_never_become_zero_wer():
    result=describe([{'reference':'','prediction':'hello there'},
                     {'reference':'','prediction':''}],'prediction')
    assert result['wer_percent'] is None
    assert result['insertions']==2 and result['emitted']==1 and result['empty']==1
