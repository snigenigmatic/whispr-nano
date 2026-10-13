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
from package_short_hindi_study import describe,validate_resume,sha
from package_vocabulary_study import verify_choices


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


def test_resume_cannot_silently_replace_original_predictions(tmp_path):
    source=tmp_path/'source';source.mkdir();target=tmp_path/'resume';target.mkdir()
    old=[dict(id='one',prediction='original')]
    (source/'predictions.jsonl').write_text(json.dumps(old[0])+'\n')
    digest=sha(source/'predictions.jsonl')
    env=dict(decoder={'model':'small'},packages={'whisper':'pinned'},manifest_sha256='same',
             gpu='A100',config={'seed':1})
    summary=dict(completed=1,predictions_sha256=digest)
    (source/'environment.json').write_text(json.dumps(env))
    (source/'summary.json').write_text(json.dumps(summary))
    resumed=dict(env,resume=dict(source_environment=env,source_summary=summary,preserved=1,
                               source_predictions_sha256=digest))
    rows=old+[dict(id='two',prediction='new')]
    assert validate_resume(target,rows,resumed)['preserved']==1
    rows[0]=dict(id='one',prediction='silently replaced')
    with pytest.raises(AssertionError,match='changed the original prefix'):
        validate_resume(target,rows,resumed)


def test_candidate_receipt_rejects_injected_reference_or_changed_winner():
    row=dict(baseline_prediction='beta',candidates=['','alpha','beta'],
        log_likelihoods=[-1.,-2.,-3.],null_log_likelihoods=[-.5,-4.,-1.],token_counts=[1,2,2],
        selected_indices=dict(sum_ll=0,mean_ll=0,sum_prior_corrected=1,mean_prior_corrected=1),
        predictions=dict(sum_ll='',mean_ll='',sum_prior_corrected='alpha',mean_prior_corrected='alpha'),
        scoring_seconds=.1)
    verify_choices(row,['alpha'])
    row['selected_indices']['sum_ll']=1
    with pytest.raises(AssertionError,match='decisions disagree'):verify_choices(row,['alpha'])
    row['candidates'].append('gold answer')
    with pytest.raises(AssertionError,match='Candidate list differs'):verify_choices(row,['alpha'])
