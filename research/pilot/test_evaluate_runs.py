"""Adversarial evidence fixtures; no model or network use."""
from collections import Counter
from dataclasses import asdict
import hashlib
import inspect
import json
from pathlib import Path
import random

import pytest

from research.pilot import evaluate_runs as audit
from research.pilot import training


def write_json(path, value):
    path.write_text(json.dumps(value, sort_keys=True, allow_nan=False))


def write_jsonl(path, rows):
    path.write_text(''.join(json.dumps(row, sort_keys=True, allow_nan=False)+'\n' for row in rows))


def source(uid, speaker, language, split, reference):
    return dict(utt_id=uid, speaker_id=speaker, native_language=language, split=split,
                prompt_id='p_'+uid, reference_text=reference, audio_path=uid+'.wav', duration_s=1.0)


@pytest.fixture
def study(tmp_path):
    rows = [source('t1', 'trainA', 'Arabic', 'train', 'first train example'),
            source('t2', 'trainB', 'Hindi', 'train', 'second train example'),
            source('t3', 'trainA', 'Arabic', 'train', 'third train example'),
            source('d1', 'devA', 'Arabic', 'dev', 'We have 3.'),
            source('d2', 'devB', 'Hindi', 'dev', "I don't go."),
            source('x', 'testC', 'Hindi', 'test', 'sealed final example')]
    manifest = tmp_path/'main.jsonl'; write_jsonl(manifest, rows)
    selections = tmp_path/'selections'; selections.mkdir()
    arms = {'full_gold': rows[:3], 'filtered10_gold': rows[:2], 'random_matched_gold': rows[1:3]}
    for arm, selected in arms.items():
        write_jsonl(selections/f'{arm}.jsonl', selected)
    schedule = {'runs': [{'arm_name': arm, 'seed': seed} for arm in arms for seed in (0,1)],
                'fixed_config': {'num_updates':2, 'batch_size':2},
                'expected_initial_state_sha256':'a'*64}
    normalizer_path = audit.ROOT/'paper/data/whisper_normalizer.json'
    english = audit.EnglishTextNormalizer(audit.load_json(normalizer_path))
    directories = []
    for planned in schedule['runs']:
        cfg = asdict(training.TrainConfig(**schedule['fixed_config'], **planned))
        selected, dev = training.validate_rows(arms[planned['arm_name']] + rows[3:5])
        directory = tmp_path/f"{planned['arm_name']}_{planned['seed']}"; directory.mkdir(); directories.append(directory)
        provenance = dict(config=cfg, train_manifest_sha256=training.canonical_hash(selected),
            dev_manifest_sha256=training.canonical_hash(dev), training_code_sha256=audit.sha(Path(training.__file__)),
            normalizer_sha256=audit.sha(normalizer_path),
            normalizer_implementation_sha256=audit.sha(Path(inspect.getfile(audit.EnglishTextNormalizer))),
            initial_state_sha256='a'*64, initial_encoder_sha256='b'*64,
            trainable_parameters=100, encoder_trainable_parameters=40,
            generation_kwargs=dict(language='en',task='transcribe',return_timestamps=False,max_new_tokens=224,
                                   do_sample=False,num_beams=1,use_cache=True),
            decoder_prefix_ids=[1,2,3], generation_config={'num_beams':1},
            runtime={'torch':'fixture','transformers':'fixture','dtype':'float32'}, feature_signature='f'*64)
        history = []
        def prediction_stage(label, update):
            predictions = []
            for row in dev:
                hypothesis = row['reference_text'] if label == 'baseline' else ('We have three' if row['utt_id']=='d1' else 'I go')
                prediction = {key:row[key] for key in ('utt_id','speaker_id','native_language','prompt_id','split','reference_text')}
                prediction.update(prediction=hypothesis,update=update,generated_token_ids=[1,2,3,4],
                    generation_reached_token_limit=False,
                    minimal=training.word_counts(training.minimal_normalize(row['reference_text']),training.minimal_normalize(hypothesis)),
                    english=training.word_counts(english(row['reference_text']),english(hypothesis)))
                predictions.append(prediction)
            metrics = training.summarize_predictions(predictions)
            write_jsonl(directory/f'{label}_dev_predictions.jsonl',predictions)
            write_json(directory/f'{label}_dev_metrics.json',metrics)
            history.append(dict(event='dev_eval', update=update, label=label, metrics=metrics,
                                elapsed_s=1.0,prediction_file=f'{label}_dev_predictions.jsonl'))
            return metrics
        baseline=prediction_stage('baseline',0)
        cycler=training.IndexCycler(len(selected),cfg['seed']); counts=Counter()
        for update in range(1,cfg['num_updates']+1):
            ids=[selected[i]['utt_id'] for i in cycler.take(cfg['batch_size'])]
            counts.update(ids)
            history.append(dict(event='train_update',update=update,loss=0.7,pre_clip_grad_norm=1.5,
                learning_rate=cfg['learning_rate'],supervised_tokens=7,audio_seconds=float(len(ids)),
                utterance_ids=ids,unique_seen=len(counts),update_seconds=0.1,elapsed_s=2.0))
        final=prediction_stage('final',cfg['num_updates'])
        audio={uid:dict(audio_sha256=hashlib.sha256(uid.encode()).hexdigest(),duration_s=1.0,
                       original_sample_rate=44100,channels=1) for uid in set(counts)|{'d1','d2'}}
        summary=dict(provenance, status='completed', baseline_dev_metrics=baseline,final_dev_metrics=final,
            final_encoder_sha256='c'*64,encoder_parameter_l2_change=0.1,encoder_relative_l2_change=0.01,
            completed_updates=cfg['num_updates'],train_pool_utterances=len(selected),dev_utterances=len(dev),
            unique_training_utterances_seen=len(counts),total_training_exposures=sum(counts.values()),
            total_supervised_tokens=14,total_training_audio_seconds=4.0,exposures_by_utterance=dict(counts),
            actual_audio=audio,history=history,
            prediction_files={'baseline':'baseline_dev_predictions.jsonl','final':'final_dev_predictions.jsonl'})
        write_json(directory/'summary.json',summary);write_json(directory/'provenance.json',provenance)
        write_json(directory/'manifest.json',selected+dev);write_jsonl(directory/'history.jsonl',history)
    return dict(run_dirs=directories,manifest_path=manifest,selections_dir=selections,schedule=schedule,expected_dev_count=2)


def mutate_json(path, mutation):
    value=audit.load_json(path);mutation(value);write_json(path,value)


def mutate_predictions(study, mutation):
    path=study['run_dirs'][0]/'final_dev_predictions.jsonl'
    rows=audit.load_jsonl(path);mutation(rows);write_jsonl(path,rows)


def test_complete_fixture_and_neutral_packet(study):
    result=audit.evaluate(**study)
    assert result['status']=='valid_complete' and result['scientific_promotion'] is False
    assert len(result['runs'])==6 and len(result['paired_contrasts'])==4
    assert result['selected_train_overlap_counts']['filtered10_gold:random_matched_gold']==1
    assert result['paired_contrasts'][0]['filtered_minus_random_macro_wer_pp']==0
    assert 'Every L1 and seed' in audit.evidence_markdown(result)
    scores=result['runs'][0]['metrics']['final']
    assert scores['english']['micro_wer']==pytest.approx(2/7)
    assert scores['english']['macro_language_wer']==pytest.approx(0.25)
    assert scores['minimal']['micro_wer']==pytest.approx(3/7)


def test_missing_scheduled_seed_is_incomplete(study):
    study['run_dirs']=study['run_dirs'][:-1]
    result=audit.evaluate(**study)
    assert result['status']=='incomplete'
    assert result['missing_runs']==[{'arm_name':'random_matched_gold','seed':1}]


@pytest.mark.parametrize('mutation',[
    lambda rows:rows.pop(),
    lambda rows:rows.append(dict(rows[0])),
    lambda rows:rows[0].update(utt_id='x'),
    lambda rows:rows[0].update(reference_text='edited easy reference'),
    lambda rows:rows[0].update(speaker_id='testC'),
    lambda rows:rows[0].update(split='test'),
    lambda rows:rows[0]['english'].update(errors=99),
    lambda rows:rows[0]['minimal'].update(reference_words=99),
])
def test_bad_prediction_rows_fail_closed(study,mutation):
    mutate_predictions(study,mutation)
    with pytest.raises(audit.EvidenceError):audit.evaluate(**study)


@pytest.mark.parametrize('field,new_value',[
    ('normalizer_implementation_sha256','d'*64),('normalizer_sha256','d'*64),
    ('training_code_sha256','d'*64),('initial_state_sha256','d'*64),
    ('dev_manifest_sha256','d'*64),('final_encoder_sha256','b'*64),
    ('completed_updates',1),('total_training_exposures',3),
])
def test_mixed_identity_or_fake_exposure_rejected(study,field,new_value):
    directory=study['run_dirs'][1]
    mutate_json(directory/'summary.json',lambda value:value.update({field:new_value}))
    with pytest.raises(audit.EvidenceError):audit.evaluate(**study)


def test_edited_aggregate_metric_rejected(study):
    mutate_json(study['run_dirs'][0]/'summary.json',lambda value:value['final_dev_metrics']['english'].update(micro_wer=0.0))
    with pytest.raises(audit.EvidenceError):audit.evaluate(**study)


def test_training_reference_cannot_change_in_both_selection_and_run(study):
    path=study['selections_dir']/'full_gold.jsonl';rows=audit.load_jsonl(path)
    rows[0]['reference_text']='a convenient new label';write_jsonl(path,rows)
    with pytest.raises(audit.EvidenceError,match='reference_text'):audit.evaluate(**study)


def test_changed_fixed_hyperparameter_rejected(study):
    mutate_json(study['run_dirs'][0]/'summary.json',lambda value:value['config'].update(learning_rate=0.1))
    with pytest.raises(audit.EvidenceError,match='learning_rate'):audit.evaluate(**study)


def test_smoke_cannot_enter_expected_schedule(study):
    study['schedule']['runs'][0]['arm_name']='smoke_not_scientific_result'
    with pytest.raises(audit.EvidenceError,match='Smoke'):audit.evaluate(**study)


def test_default_production_count_is_974(study):
    study.pop('expected_dev_count')
    with pytest.raises(audit.EvidenceError,match='974'):audit.evaluate(**study)


def test_independent_counts_match_producer_across_alignment_ties():
    rng=random.Random(124)
    for _ in range(100):
        ref=' '.join(rng.choice(['a','b','c']) for _ in range(rng.randint(1,8)))
        hyp=' '.join(rng.choice(['a','b','c']) for _ in range(rng.randint(0,8)))
        assert audit.edit_counts(ref,hyp)==training.word_counts(ref,hyp)
    for text in ["I don't go.", '“ＦＯＯ”—bar!', 'café №5', 'Hindi_English']:
        assert audit.minimal(text)==training.minimal_normalize(text)


def test_same_fake_initialization_across_all_runs_still_fails_registered_identity(study):
    for directory in study['run_dirs']:
        for name in ('summary.json','provenance.json'):
            mutate_json(directory/name,lambda value:value.update(initial_state_sha256='d'*64))
    with pytest.raises(audit.EvidenceError,match='Registered initial state'):audit.evaluate(**study)


def test_missing_control_in_schedule_is_not_a_complete_experiment(study):
    study['schedule']['runs']=[r for r in study['schedule']['runs'] if r['arm_name']!='random_matched_gold']
    with pytest.raises(audit.EvidenceError,match='matched-random'):audit.evaluate(**study)


def test_full_arm_cannot_silently_be_a_subset(study):
    path=study['selections_dir']/'full_gold.jsonl';rows=audit.load_jsonl(path);write_jsonl(path,rows[:2])
    with pytest.raises(audit.EvidenceError,match='entire frozen training pool'):audit.evaluate(**study)


def test_supplied_manifest_must_match_registered_hash(study):
    study['schedule']['frozen_inputs']={'research/pilot/data/manifest.jsonl':audit.sha(audit.ROOT/'research/pilot/data/manifest.jsonl')}
    with pytest.raises(audit.EvidenceError,match='Supplied frozen input'):audit.evaluate(**study)
