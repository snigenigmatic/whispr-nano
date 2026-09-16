"""Integration guard tests using the independently scored synthetic ASR fixture."""

import copy
import json
from pathlib import Path
import shutil

import pytest

from research.pilot import data_setup, evaluate_runs, loop, workflow
from research.pilot.test_evaluate_runs import study


def save(path, value):
    Path(path).write_text(json.dumps(value, allow_nan=False))


@pytest.fixture
def integrated(tmp_path, monkeypatch, study):
    real_pilot = Path(workflow.__file__).parent
    root = tmp_path / 'workspace'
    pilot = root / 'research/pilot'
    pilot.mkdir(parents=True)
    for directory in ('data', 'teacher_predictions', 'runs', 'selections'):
        (pilot / directory).mkdir()
    shutil.copy(study['manifest_path'], pilot / 'data/manifest.jsonl')
    for source in study['selections_dir'].glob('*.jsonl'):
        shutil.copy(source, pilot / 'selections' / source.name)
    save(pilot / 'selections/selection_report.json', {'stratum_balance': [{'duration_match_within_tolerance': True}]})
    save(pilot / 'feature_cache_receipt.json', {'status': 'completed', 'rows': 10469,
                                              'test_access': False, 'feature_signature': 'f' * 64})
    (pilot / 'teacher_predictions/all_train.jsonl').write_text('{}\n')
    for name in ('training.py', 'frozen_english_normalizer.py', 'selection.py', 'evaluate_runs.py', 'workflow.py', 'loop.py', 'modal_runner.py', 'feature_cache.py'):
        shutil.copy(real_pilot / name, pilot / name)
    (root / 'paper/data').mkdir(parents=True)
    shutil.copy(real_pilot.parents[1] / 'paper/data/whisper_normalizer.json', root / 'paper/data/whisper_normalizer.json')
    schedule = copy.deepcopy(study['schedule'])
    schedule.update(experiment_id='integration-fixture', frozen_inputs={},
                    pilot_positive_gate={'mean_primary_contrast_at_least': 0.005,
                                         'positive_paired_seeds_at_least': 2,
                                         'mean_minimal_contrast_at_least': 0})
    save(pilot / 'schedule.json', schedule)
    for directory in study['run_dirs']:
        summary = loop.read_json(directory / 'summary.json')
        name = workflow.run_name(summary['config']['arm_name'], summary['config']['seed'])
        destination = pilot / 'runs' / name
        shutil.copytree(directory, destination)
        summary['checkpoint_model_sha256'] = 'c' * 64
        save(destination / 'summary.json', summary)
    smoke = pilot / 'runs/train-smoke-1'
    smoke.mkdir()
    save(smoke / 'summary.json', {'completed_updates': 3, 'dev_utterances': 1})
    for label in ('baseline', 'final'):
        prediction = json.loads((study['run_dirs'][0] / f'{label}_dev_predictions.jsonl').read_text().splitlines()[0])
        (smoke / f'{label}_dev_predictions.jsonl').write_text(json.dumps(prediction) + '\n')
    monkeypatch.setattr(workflow, 'ROOT', root)
    monkeypatch.setattr(workflow, 'PILOT', pilot)
    monkeypatch.setattr(workflow, 'STORE', pilot / 'ledger')

    def validated_headers(path):
        save(path / 'audio_header_checks.json', {'status': 'fixture-verified'})
    monkeypatch.setattr(data_setup, 'validate_audio_headers', validated_headers)
    real_evaluate = evaluate_runs.evaluate

    def small_fixture_audit(*args, **kwargs):
        # The production entry point retains974; only the fixture has2 dev rows.
        return real_evaluate(*args, expected_dev_count=2, **kwargs)
    monkeypatch.setattr(evaluate_runs, 'evaluate', small_fixture_audit)
    workflow.initialize()
    return {'root': root, 'pilot': pilot, 'store': workflow.STORE, 'schedule': schedule}


def config(arm='full_gold', seed=0):
    return {'run_name': workflow.run_name(arm, seed), 'job_id': f'fixture/{arm}/{seed}',
            'training': workflow.training_config(arm, seed)}


def test_initialization_discloses_smoke_and_freezes_its_actual_subset(integrated):
    current = loop.status(integrated['store'])
    assert current['dev_queries_consumed'] == current['dev_queries_actual_reported'] == 2
    assert current['jobs']['smoke_prior'] == 'passed'
    assert all(value == 'queued' for key, value in current['jobs'].items() if key != 'smoke_prior')
    contract = loop.read_json(integrated['store'] / 'contract.json')
    queries = contract['jobs']['smoke_prior']['dev_queries']
    assert all(query['dataset'] == 'smoke_development' for query in queries.values())
    smoke_rows = (integrated['pilot'] / 'data/smoke_dev_ledger_manifest.jsonl').read_text().splitlines()
    primary_rows = (integrated['pilot'] / 'data/dev_ledger_manifest.jsonl').read_text().splitlines()
    assert len(smoke_rows) == 1 and len(primary_rows) == 2
    assert workflow.initialize()['ledger_head'] == current['ledger_head']


@pytest.mark.parametrize('mutation', ['updates', 'arm_mapping', 'unregistered_seed'])
def test_parameter_or_identity_drift_is_rejected_before_dispatch(integrated, mutation):
    job = config()
    if mutation == 'updates':
        job['training']['num_updates'] += 1
    elif mutation == 'arm_mapping':
        job['run_name'] = workflow.run_name('filtered10_gold', 0)
    else:
        with pytest.raises(ValueError, match='absent from the frozen schedule'):
            workflow.training_config('full_gold', 99)
        return
    with pytest.raises(ValueError):
        workflow.before_dispatch(job)
    assert loop.status(integrated['store'])['dev_queries_consumed'] == 2


def test_code_tampering_blocks_dispatch_without_starting_compute(integrated):
    path = integrated['pilot'] / 'selection.py'
    path.write_text(path.read_text() + '\n# Changed after registration\n')
    with pytest.raises(ValueError, match='Artifact mutation'):
        workflow.before_dispatch(config())


def test_independent_run_audit_precedes_success_and_binds_without_double_charge(integrated):
    job = config()
    workflow.before_dispatch(job)
    before = loop.status(integrated['store'])
    assert before['dev_queries_consumed'] == 4 and before['dev_reservations_charged'] == 2
    assert before['dev_queries_actual_reported'] == 2
    directory = integrated['pilot'] / 'runs' / job['run_name']
    workflow.after_result(job, directory)
    after = loop.status(integrated['store'])
    assert after['jobs'][job['run_name']] == 'passed'
    assert after['dev_queries_consumed'] == after['dev_queries_actual_reported'] == 4
    assert after['dev_execution_receipts'] == 2
    receipt = loop.read_json(directory / 'workflow_audit.json')
    assert receipt['matching_registered_run'] is True
    assert receipt['audit']['status'] == 'incomplete'
    assert receipt['audit']['runs'][0]['dev_evaluation_queries'] == 2


@pytest.mark.parametrize('corruption', ['score', 'arm', 'extra_eval'])
def test_bad_worker_result_cannot_be_recorded_as_passed(integrated, corruption):
    job = config()
    workflow.before_dispatch(job)
    directory = integrated['pilot'] / 'runs' / job['run_name']
    if corruption == 'score':
        path = directory / 'final_dev_predictions.jsonl'
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        rows[0]['english']['errors'] = 1234
        path.write_text(''.join(json.dumps(row) + '\n' for row in rows))
    elif corruption == 'arm':
        path = directory / 'summary.json'
        summary = loop.read_json(path)
        summary['config']['arm_name'] = 'filtered10_gold'
        save(path, summary)
    else:
        path = directory / 'history.jsonl'
        with path.open('a') as stream:
            stream.write(json.dumps({'event': 'dev_eval', 'label': 'extra_search', 'update': 1}) + '\n')
    with pytest.raises(ValueError):
        workflow.after_result(job, directory)
    state = loop.status(integrated['store'])
    assert state['jobs'][job['run_name']] == 'running'
    assert state['dev_queries_consumed'] == 4
    assert state['dev_queries_actual_reported'] == 2
    assert not (directory / 'workflow_audit.json').exists()


def test_unknown_worker_failure_does_not_assert_zero_actual_queries(integrated):
    job = config()
    workflow.before_dispatch(job)
    path = workflow.record_failure(job, RuntimeError('Worker connection lost'))
    evidence = loop.read_json(path)
    assert 'Unknown' in evidence['actual_development_evaluations']
    assert 'actual_dev_query_ids' not in evidence
    state = loop.status(integrated['store'])
    assert state['jobs'][job['run_name']] == 'running'
    assert state['dev_queries_consumed'] == 4
    assert workflow.finalize is not None
    with pytest.raises(ValueError, match='every registered run passes'):
        workflow.finalize()
    with pytest.raises(ValueError, match='already exists'):
        workflow.record_failure(job, RuntimeError('Repeated unknown status'))


def test_training_feature_signature_must_match_frozen_precomputed_cache(integrated):
    job = config()
    workflow.before_dispatch(job)
    directory = integrated['pilot'] / 'runs' / job['run_name']
    summary_path = directory / 'summary.json'
    summary = loop.read_json(summary_path)
    summary['feature_signature'] = 'e' * 64
    save(summary_path, summary)
    with pytest.raises(ValueError, match='feature signature differs from the precomputed cache'):
        workflow.after_result(job, directory)
    assert not (directory / 'workflow_audit.json').exists()
    current = loop.status(integrated['store'])
    assert current['jobs'][job['run_name']] == 'running'
    assert current['dev_reservations_charged'] == 2
    assert current['dev_execution_receipts'] == 0


def test_complete_null_pilot_remains_inconclusive_and_cannot_promote(integrated):
    for planned in integrated['schedule']['runs']:
        job = config(planned['arm_name'], planned['seed'])
        workflow.before_dispatch(job)
        workflow.after_result(job, integrated['pilot'] / 'runs' / job['run_name'])
    result = workflow.finalize()
    assert result['decision'] == 'blocked'
    assert result['final_test'] == 'sealed'
    current = loop.status(integrated['store'])
    assert current['jobs']['scientific_assessment'] == 'inconclusive'
    decision = loop.read_json(integrated['pilot'] / 'results/decision.json')
    assert decision['pilot_positive'] is False
    assert decision['mean_english_contrast_pp'] == 0
    assert decision['mean_minimal_contrast_pp'] == 0
    assert workflow.finalize()['decision'] == 'blocked'
