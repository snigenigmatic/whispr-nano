"""Bind the ASR pilot's fixed schedule to local evidence and evaluation accounting."""
from __future__ import annotations

import json
from pathlib import Path
import statistics

from research.pilot import loop
from research.pilot.training import canonical_hash

ROOT = Path(__file__).resolve().parents[2]
PILOT = ROOT / 'research/pilot'
STORE = PILOT / 'ledger'


def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')


def run_name(arm, seed):
    return f'{arm}-seed{seed}'


def training_config(arm, seed):
    schedule = loop.read_json(PILOT / 'schedule.json')
    if dict(arm_name=arm, seed=seed) not in schedule['runs']:
        raise ValueError('Arm/seed is absent from the frozen schedule')
    rows = [json.loads(line) for line in (PILOT / 'data/manifest.jsonl').read_text().splitlines()]
    dev = sorted((r for r in rows if r['split'] == 'dev'), key=lambda r: r['utt_id'])
    return {**schedule['fixed_config'], 'arm_name': arm, 'seed': seed,
            'expected_initial_state_sha256': schedule['expected_initial_state_sha256'],
            'expected_dev_manifest_sha256': canonical_hash(dev)}


def check(source, pointer, value, op='eq'):
    return dict(source=source, pointer=pointer, op=op, value=value)


def initialize():
    """Freeze selected inputs before any proper training; disclose earlier work."""
    if STORE.exists():
        return loop.status(STORE)
    from research.pilot.data_setup import validate_audio_headers
    validate_audio_headers(PILOT / 'data')
    feature_receipt = loop.read_json(PILOT / 'feature_cache_receipt.json')
    if feature_receipt.get('status') != 'completed' or feature_receipt.get('rows') != 10469 or feature_receipt.get('test_access') is not False:
        raise ValueError('Complete frozen train/dev feature cache is required before GPU training')
    schedule = loop.read_json(PILOT / 'schedule.json')
    for path, expected in schedule['frozen_inputs'].items():
        if loop.file_sha(ROOT / path) != expected:
            raise ValueError(f'Frozen schedule input changed: {path}')
    rows = [json.loads(line) for line in (PILOT / 'data/manifest.jsonl').read_text().splitlines()]
    for split in ('dev', 'test'):
        path = PILOT / 'data' / f'{split}_ledger_manifest.jsonl'
        path.write_text(''.join(json.dumps(r, sort_keys=True) + '\n' for r in rows if r['split'] == split))
    smoke_dir = PILOT / 'runs/train-smoke-1'
    smoke_summary = loop.read_json(smoke_dir / 'summary.json')
    smoke_ids = []
    for label in ('baseline', 'final'):
        predictions = [json.loads(line) for line in (smoke_dir / f'{label}_dev_predictions.jsonl').read_text().splitlines() if line.strip()]
        ids = [r['utt_id'] for r in predictions]
        if len(ids) != len(set(ids)) or len(ids) != smoke_summary['dev_utterances']:
            raise ValueError('Prior smoke prediction identities do not match its recorded development count')
        smoke_ids.append(set(ids))
    if smoke_ids[0] != smoke_ids[1]:
        raise ValueError('Prior smoke baseline/final used different development sets')
    dev_by_id = {r['utt_id']: r for r in rows if r['split'] == 'dev'}
    if smoke_ids[0] - set(dev_by_id):
        raise ValueError('Prior smoke includes IDs outside the registered development set')
    smoke_manifest = PILOT / 'data/smoke_dev_ledger_manifest.jsonl'
    smoke_manifest.write_text(''.join(json.dumps(dev_by_id[uid], sort_keys=True) + '\n' for uid in sorted(smoke_ids[0])))
    report = loop.read_json(PILOT / 'selections/selection_report.json')
    if not all(r['duration_match_within_tolerance'] for r in report['stratum_balance']):
        raise ValueError('Selected control fails the frozen duration balance tolerance')
    artifacts = {}
    for name, path, role in [
        ('schedule', 'schedule.json', 'input'), ('manifest', 'data/manifest.jsonl', 'input'),
        ('development', 'data/dev_ledger_manifest.jsonl', 'development'),
        ('smoke_development', 'data/smoke_dev_ledger_manifest.jsonl', 'development'),
        ('sealed_test', 'data/test_ledger_manifest.jsonl', 'sealed_test'),
        ('training', 'training.py', 'evaluator'), ('normalizer', 'frozen_english_normalizer.py', 'evaluator'),
        ('selection', 'selection.py', 'code'), ('auditor', 'evaluate_runs.py', 'code'),
        ('controller', 'workflow.py', 'code'), ('loop', 'loop.py', 'code'),
        ('runner', 'modal_runner.py', 'code'), ('selection_report', 'selections/selection_report.json', 'input'),
        ('audio_headers', 'data/audio_header_checks.json', 'input'),
        ('teacher_predictions', 'teacher_predictions/all_train.jsonl', 'input'),
        ('feature_cache', 'feature_cache.py', 'code'),
        ('feature_receipt', 'feature_cache_receipt.json', 'input'),
    ]:
        artifacts[name] = dict(path=str(PILOT / path), role=role)
    for arm in ('full_gold', 'filtered10_gold', 'random_matched_gold'):
        artifacts[arm] = dict(path=str(PILOT / 'selections' / f'{arm}.jsonl'), role='input')
    for path in (PILOT / 'teacher_predictions').glob('*.runtime.json'):
        artifacts['teacher_' + path.stem] = dict(path=str(path), role='input')
    for relative in schedule.get('protocol_amendments', []):
        path = ROOT / relative
        artifacts['amendment_' + path.stem] = dict(path=str(path), role='input')
    artifacts['normalizer_map'] = dict(path=str(ROOT / 'paper/data/whisper_normalizer.json'), role='evaluator')
    query = lambda label: dict(dataset='development', evaluator='training', purpose=label)
    jobs = {
        'smoke_prior': dict(kind='infrastructure', dependencies=[], parameters={},
            required_evidence=['summary', 'baseline', 'final'],
            success_checks=[check('summary', '/completed_updates', 3),
                            check('summary', '/dev_utterances', len(smoke_ids[0]))],
            dev_queries={k: dict(dataset='smoke_development', evaluator='training',
                                purpose='Disclosed prior smoke on its exact recorded development IDs') for k in ('baseline', 'final')}),
    }
    names = []
    for run in schedule['runs']:
        name = run_name(run['arm_name'], run['seed'])
        names.append(name)
        jobs[name] = dict(kind='infrastructure', dependencies=[],
            parameters=training_config(**dict(arm=run['arm_name'], seed=run['seed'])),
            required_evidence=['summary', 'baseline', 'final', 'manifest', 'history', 'audit'],
            success_checks=[check('summary', '/status', 'completed'),
                            check('summary', '/completed_updates', schedule['fixed_config']['num_updates']),
                            check('summary', '/total_training_exposures', schedule['fixed_config']['num_updates'] *
                                  schedule['fixed_config']['batch_size'] * schedule['fixed_config'].get('grad_accum_steps', 1)),
                            check('summary', '/dev_utterances', len(dev_by_id)),
                            check('audit', '/audited_run_count', 1),
                            check('audit', '/matching_registered_run', True),
                            check('audit', '/dev_evaluation_queries', 2)],
            dev_queries={k: query('Fixed initial weights' if k == 'baseline' else 'Fixed final update')
                         for k in ('baseline', 'final')})
    jobs['scientific_assessment'] = dict(kind='scientific', dependencies=names, parameters={},
        required_evidence=['audit', 'decision'], dev_queries={},
        success_checks=[check('audit', '/status', 'valid_complete'),
                        check('decision', '/pilot_positive', True)])
    contract = dict(schema_version=1, experiment_id=schedule['experiment_id'],
        max_jobs=len(jobs), max_running=3, max_dev_queries=20, final_test_policy='sealed',
        artifacts=artifacts, jobs=jobs,
        registration_note='Proper training prospective. Data preparation, teacher inference and smoke began before this ledger; preserved separately in budget.json and source receipts.',
        gates={'replication_ready': dict(required_jobs=['scientific_assessment'], checks=[])})
    dump(PILOT / 'contract.json', contract)
    loop.initialize(STORE, PILOT / 'contract.json')
    loop.apply(STORE, 'register', dict(job_id='smoke_prior', origin='retrospective',
               reason='Infrastructure smoke completed before this controller existed. It is excluded from scientific outcomes.'))
    loop.apply(STORE, 'start', dict(job_id='smoke_prior'))
    smoke = PILOT / 'runs/train-smoke-1/summary.json'
    for label in ('baseline', 'final'):
        loop.apply(STORE, 'query', dict(job_id='smoke_prior', query_id=label,
            accounting='retrospective_import', reason='Completed 12-clip smoke evaluation; no full final checkpoint fingerprint imported.',
            receipt={'path': str(smoke)}))
    loop.apply(STORE, 'record', dict(job_id='smoke_prior', outcome='passed',
        evidence={'summary': {'path': str(smoke)},
                  'baseline': {'path': str(smoke_dir / 'baseline_dev_predictions.jsonl')},
                  'final': {'path': str(smoke_dir / 'final_dev_predictions.jsonl')}}, dev_query_ids=['baseline', 'final'],
        actual_dev_query_ids=['baseline', 'final']))
    for name in names + ['scientific_assessment']:
        loop.apply(STORE, 'register', dict(job_id=name, origin='prospective'))
    return loop.status(STORE)


def before_dispatch(config):
    name = config['run_name']
    contract = loop.read_json(STORE / 'contract.json')
    if name not in contract['jobs'] or name != run_name(config['training']['arm_name'], config['training']['seed']):
        raise ValueError('Dispatch run name does not identify its registered arm and seed')
    if config['training'] != contract['jobs'][name]['parameters']:
        raise ValueError('Dispatch configuration differs from registered parameters')
    loop.apply(STORE, 'start', dict(job_id=name, external_job_id=config['job_id']))
    for label in ('baseline', 'final'):
        loop.apply(STORE, 'reserve_query', dict(job_id=name, query_id=label,
            reason='Precharge both fixed development evaluations before the Modal call; reservations are never refunded.'))


def after_result(config, directory):
    from research.pilot.evaluate_runs import evaluate
    directory = Path(directory)
    summary = loop.read_json(directory / 'summary.json')
    feature_receipt = loop.read_json(PILOT / 'feature_cache_receipt.json')
    if summary.get('feature_signature') != feature_receipt['feature_signature']:
        raise ValueError('Training feature signature differs from the precomputed cache')
    name = config['run_name']
    if summary['config']['arm_name'] != config['training']['arm_name'] or summary['config']['seed'] != config['training']['seed']:
        raise ValueError('Worker returned a different arm/seed than the dispatched job')
    schedule = loop.read_json(PILOT / 'schedule.json')
    audit = evaluate([directory], PILOT / 'data/manifest.jsonl', PILOT / 'selections', schedule)
    matching = (len(audit['runs']) == 1 and audit['runs'][0]['arm_name'] == config['training']['arm_name']
                and audit['runs'][0]['seed'] == config['training']['seed'])
    if not matching or audit['runs'][0]['dev_evaluation_queries'] != 2:
        raise ValueError('Independent per-run audit does not match the fixed two-query dispatched experiment')
    receipt = dict(audited_run_count=len(audit['runs']), matching_registered_run=matching,
                   dev_evaluation_queries=audit['runs'][0]['dev_evaluation_queries'],
                   audit=audit,
                   scope='Independent integrity audit of one run; the full nine-run evidence remains incomplete.')
    audit_path = directory / 'workflow_audit.json'
    if audit_path.exists():
        raise ValueError('Refusing to overwrite a previously recorded per-run audit')
    dump(audit_path, receipt)
    for label, fingerprint in [('baseline', summary['initial_state_sha256']),
                               ('final', summary['checkpoint_model_sha256'])]:
        loop.apply(STORE, 'query', dict(job_id=name, query_id=label,
            accounting='receipt_after_run', checkpoint_sha256=fingerprint,
            reason='Verified worker receipt after fixed decoding; baseline state hash or final safetensors hash.',
            receipt={'path': str(audit_path)}))
    evidence = {key: {'path': str(directory / filename)} for key, filename in
                [('summary', 'summary.json'), ('baseline', 'baseline_dev_predictions.jsonl'),
                 ('final', 'final_dev_predictions.jsonl'), ('manifest', 'manifest.json'), ('history', 'history.jsonl')]}
    evidence['audit'] = {'path': str(audit_path)}
    loop.apply(STORE, 'record', dict(job_id=name, outcome='passed', evidence=evidence,
        dev_query_ids=['baseline', 'final'], actual_dev_query_ids=['baseline', 'final']))


def record_failure(config, error):
    directory = PILOT / 'runs' / config['run_name']
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / 'failure.json'
    if path.exists():
        raise ValueError('Failure evidence already exists; do not overwrite unresolved execution history')
    dump(path, dict(error=str(error), config=config,
                    actual_development_evaluations='Unknown until remote artifacts are recovered; both reserved slots remain charged.'))
    # Do not assert zero evaluations on a worker failure: an unknown partial
    # execution stays running/needs inspection, with its query reservations intact.
    return path


def finalize():
    from research.pilot.evaluate_runs import evaluate, evidence_markdown
    current = loop.status(STORE)
    assessment_state = current['jobs']['scientific_assessment']
    if assessment_state in loop.OUTCOMES:
        return loop.gate(STORE, 'replication_ready')
    missing = [name for name, state in current['jobs'].items()
               if name not in ('scientific_assessment', 'smoke_prior') and state != 'passed']
    if missing:
        raise ValueError(f'Cannot assess the scientific contrast before every registered run passes: {missing}')
    schedule = loop.read_json(PILOT / 'schedule.json')
    directories = [PILOT / 'runs' / run_name(r['arm_name'], r['seed']) for r in schedule['runs']]
    audit = evaluate(directories, PILOT / 'data/manifest.jsonl', PILOT / 'selections', schedule)
    output = PILOT / 'results'
    output.mkdir(exist_ok=True)
    dump(output / 'audit.json', audit)
    (output / 'evidence.md').write_text(evidence_markdown(audit))
    english = [r['filtered_minus_random_macro_wer_pp'] for r in audit['paired_contrasts'] if r['normalization'] == 'english']
    minimal = [r['filtered_minus_random_macro_wer_pp'] for r in audit['paired_contrasts'] if r['normalization'] == 'minimal']
    rules = schedule['pilot_positive_gate']
    decision = dict(valid_complete=audit['status'] == 'valid_complete',
        mean_english_contrast_pp=statistics.mean(english) if english else None,
        positive_seed_count=sum(x > 0 for x in english),
        mean_minimal_contrast_pp=statistics.mean(minimal) if minimal else None,
        scope='Exploratory pilot only; a positive result permits independent replication, not a paper claim.')
    decision['pilot_positive'] = (decision['valid_complete'] and len(english) == 3 and len(minimal) == 3
        and decision['mean_english_contrast_pp'] >= 100 * rules['mean_primary_contrast_at_least']
        and decision['positive_seed_count'] >= rules['positive_paired_seeds_at_least']
        and decision['mean_minimal_contrast_pp'] >= 100 * rules['mean_minimal_contrast_at_least'])
    dump(output / 'decision.json', decision)
    loop.apply(STORE, 'start', dict(job_id='scientific_assessment'))
    loop.apply(STORE, 'record', dict(job_id='scientific_assessment',
        outcome='passed' if decision['pilot_positive'] else 'inconclusive',
        evidence={'audit': {'path': str(output / 'audit.json')}, 'decision': {'path': str(output / 'decision.json')}},
        reason='Registered pilot gate met; replication required.' if decision['pilot_positive'] else 'Registered pilot support criterion not met; preserve the result without changing the metric.',
        dev_query_ids=[], actual_dev_query_ids=[]))
    return loop.gate(STORE, 'replication_ready')
