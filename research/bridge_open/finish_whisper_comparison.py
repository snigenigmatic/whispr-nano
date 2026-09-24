"""Compose a fully covered, explicitly duration-routed Whisper benchmark."""
import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

from research.bridge_open.analyze_fleurs import load_model,cluster_bootstrap_difference
from research.bridge_open.scoring import score,normalize_v1

ROOT=Path(__file__).resolve().parents[2]
ART=ROOT/'research/bridge_open/artifacts'
MODELS={'Turbo':('openai/whisper-large-v3-turbo','41f01f3fe87f28c78e2fbf8b568835947dd65ed9'),
        'Large-v3':('openai/whisper-large-v3','06f233fe06e710322aca913c1bc4249a0d71fce1')}


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def jsonl(p):return [json.loads(s) for s in Path(p).read_text().splitlines() if s.strip()]
def aggregate(rows):
    errors=sum(r['metrics'][k] for r in rows for k in ['substitutions','deletions','insertions'])
    words=sum(r['metrics']['ref_words'] for r in rows)
    chars=sum(r['metrics']['ref_chars'] for r in rows)
    return dict(n=len(rows),word_errors=errors,ref_words=words,micro_wer=errors/words,
        micro_cer=sum(r['metrics']['character_errors'] for r in rows)/chars,
        known_short_generation_limits=sum(r.get('status')=='generation_limit' for r in rows),
        native_termination_not_observed=sum(r['route']=='native_longform' for r in rows),
        normalized_empty=sum(not normalize_v1(r['text_raw']) for r in rows))


def analyze(correction,out):
    correction=Path(correction);out=Path(out)
    assert not (correction/'failure.json').exists()
    cfg=json.loads((correction/'config.json').read_text());env=json.loads((correction/'environment.json').read_text())
    records=jsonl(correction/'predictions.jsonl')
    assert len(records)==124
    assert json.loads((correction/'summary.json').read_text())['status']=='completed'
    receipt=json.loads((correction/'orx_source_receipt.json').read_text())
    assert sha(receipt['sourcePath'])==receipt['sourceDigest']
    manifest_path=ROOT/'data/bridge_open/fleurs_test_v1/corpus-v2.jsonl'
    assert sha(manifest_path)=='967d8dca02f9f729c1ed7548dce7283cf9dfc62623815a7c131f71acb13ce2c7'
    manifest=jsonl(manifest_path);byid={r['conversation_id']:r for r in manifest};assert len(byid)==len(manifest)==2322
    longids={k for k,r in byid.items() if r['duration_seconds']>30};assert len(longids)==14
    cohort=json.loads((correction/'cohort.json').read_text())
    shortids={r['conversation_id'] for r in cohort if r['duration_seconds']<=30};assert len(shortids)==24
    assert {r['conversation_id'] for r in cohort}-shortids==longids
    keyed={}
    for r in records:
        k=(r['model_id'],r['arm'],r['conversation_id']);assert k not in keyed
        keyed[k]=r;s=byid[r['conversation_id']]
        assert r['audio_sha256']==s['audio_sha256'] and r['reference_sha256']==s['reference_sha256']
        assert sha(s['reference_path'])==r['reference_sha256']
        ref=json.loads(Path(s['reference_path']).read_text())['reference_text']
        assert score(ref,r['text'])==r['metrics']
        assert r['sample_rate']==16000 and abs(r['input_sample_count']/16000-s['duration_seconds'])<1e-4
        if r['arm']=='native_longform':
            assert r['effective_options']==dict(language=s['language'],**cfg['native_transcribe'])
            assert r['decode_attempts'] and r['native_language']==s['language']
    for e in env:
        assert (e['model_id'],e['model_revision']) in MODELS.values()
        assert e['dtype']==e['native_dtype']=='torch.float32' and not e['hf_training'] and not e['native_training']
        assert e['script_sha256']==sha(correction/'executed_audit.py')
        weights=json.loads((correction/f"{e['native_model']}_weights.json").read_text())
        assert weights and all(w['max_abs_delta']<=(0 if w['learned'] else 1e-5) for w in weights)
        features=json.loads((correction/f"{e['native_model']}_features.json").read_text())
        assert {f['conversation_id'] for f in features}=={r['conversation_id'] for r in cohort}
        assert all(f['max_abs_delta']<1e-4 for f in features)
    assert len(env)==2
    registry=json.loads((ART/'fleurs_whisper_full_registry.json').read_text())
    composed={};old={};parity={};sensitivity=[]
    for label,(mid,rev) in MODELS.items():
        dirs=[Path(e['output_dir']) for e in registry if e['model_id']==mid]
        baseline=load_model(manifest,mid,rev,dirs);old[label]=baseline
        oldby={r['conversation_id']:r for r in baseline}
        native={r['conversation_id']:r for r in records if r['model_id']==mid and r['arm']=='native_longform'}
        assert set(native)==longids
        if label=='Turbo':
            expected_arms=['hf_batch','hf_single','hf_pipeline','native_decode_matched']
            for arm in expected_arms:
                group=[r for r in records if r['model_id']==mid and r['arm']==arm]
                assert {r['conversation_id'] for r in group}==shortids
                equal=sum(r['text'].strip()==oldby[r['conversation_id']]['text_raw'].strip() for r in group)
                parity[arm]=dict(n=len(group),texts_matching_original=equal,
                    wers_matching_original=sum(r['metrics']['wer']==oldby[r['conversation_id']]['metrics']['wer'] for r in group))
                if arm in ['hf_batch','hf_single','hf_pipeline']:assert equal==24,(arm,equal)
        result=[]
        for source in manifest:
            cid=source['conversation_id'];before=oldby[cid]
            if cid in longids:
                r=native[cid]
                rec=dict(conversation_id=cid,model_id=mid,model_revision=rev,text_raw=r['text'],metrics=r['metrics'],
                    audio_sha256=r['audio_sha256'],reference_sha256=r['reference_sha256'],status='native_api_completed',
                    evidence_path=str(correction/'predictions.jsonl'),route='native_longform',
                    native_segment_bounds_valid=r['segment_bounds_valid'],termination_evidence='public native API strips EOS; see decode attempts')
                sensitivity.append(dict(model=label,id=cid,language=source['language'],duration=source['duration_seconds'],
                    old_wer=before['metrics']['wer'],native_wer=r['metrics']['wer'],
                    old_insertions=before['metrics']['insertions'],native_insertions=r['metrics']['insertions']))
            else:
                rec={**before,'route':'hf_short_beam5','evidence_path':'original full-test shard registry'}
            rec.update(language=source['language'],duration_seconds=source['duration_seconds'],source_sentence_id=str(source['source_sentence_id']))
            result.append(rec)
        assert len(result)==len({r['conversation_id'] for r in result})==2322
        composed[label]=result
    report=dict(scope='Descriptive duration-routed Whisper systems on public FLEURS official test. Post-result protocol correction.',
        inference_acceptance='accepted for the declared routes; no universal best-model, held-out improvement, population or fairness claim',
        models={},turbo_parity=parity,paired_differences=[],long_sensitivity=sensitivity,
        limitations=['Small native parity cohorts cannot prove all possible decoder behavior.',
            'Native long-form has stochastic fallback, fixed seed 12, different context/segmentation from short form.',
            'No human semantic transcript validation; public source integrity was checked.',
            'Native EOS/cap termination is not directly exposed; do not interpret unknown caps as zero.'])
    for label,rows in composed.items():
        report['models'][label]=dict(model_id=MODELS[label][0],revision=MODELS[label][1],languages={})
        for lang in ['hi','bn','as']:
            group=[r for r in rows if r['language']==lang]
            report['models'][label]['languages'][lang]=dict(pooled_duration_routed=aggregate(group),
                strata={route:aggregate([r for r in group if r['route']==route]) for route in ['hf_short_beam5','native_longform']},
                old_fixed_recipe=aggregate([{**r,'route':'hf_short_beam5' if r['conversation_id'] not in longids else 'old_fixed_split'} for r in old[label] if r['language']==lang]))
    for lang in ['hi','bn','as']:
        report['paired_differences'].append(dict(language=lang,left='Turbo',right='Large-v3',
            **cluster_bootstrap_difference([r for r in composed['Turbo'] if r['language']==lang],[r for r in composed['Large-v3'] if r['language']==lang])))
    out.mkdir(parents=True,exist_ok=False)
    (out/'comparison.json').write_text(json.dumps(report,indent=2)+'\n')
    for label,rows in composed.items():
        (out/f'{label.lower()}_predictions.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
    with (out/'long_sensitivity.tsv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(sensitivity[0]),delimiter='\t');w.writeheader();w.writerows(sensitivity)
    receipts=dict(manifest_sha256=sha(manifest_path),correction_transport_sha256=json.loads((correction/'transport_receipt.json').read_text())['sha256'],
        correction_run=receipt,analysis_sha256=sha(__file__),scorer_sha256=sha(ROOT/'research/bridge_open/scoring.py'),
        original_registry_sha256=sha(ART/'fleurs_whisper_full_registry.json'),scored_predictions=4644)
    (out/'analysis_receipt.json').write_text(json.dumps(receipts,indent=2)+'\n')
    lines=['# Corrected Whisper comparison','','All 2,322 recordings per model are covered. Short clips retain the audited HF beam-5 route; every >30-second clip uses pinned native transcription. This is a post-result correction, not a held-out improvement.','','| Model | Language | n | Short WER | Long WER (n) | Pooled routed WER | Pooled CER |','|---|---|---:|---:|---:|---:|---:|']
    for label,model in report['models'].items():
        for lang,x in model['languages'].items():
            a=x['pooled_duration_routed'];s=x['strata']['hf_short_beam5'];l=x['strata']['native_longform']
            lines.append(f"| {label} | {lang} | {a['n']} | {100*s['micro_wer']:.2f}% | {100*l['micro_wer']:.2f}% ({l['n']}) | {100*a['micro_wer']:.2f}% | {100*a['micro_cer']:.2f}% |")
    lines+=['','The pooled column describes the declared duration-based routing rule. It is not a homogeneous decoder condition. Long-stratum sample counts are small. Preserve the previous fixed-split results as sensitivity analysis; do not substitute only favorable cases.','','Known short-form caps remain in the denominator. Native EOS/cap evidence is unavailable; detailed decode attempts and timestamp flags remain in the source artifacts. Sentence-cluster bootstrap intervals in comparison.json do not support speaker/population fairness claims.','','No human semantic transcript review was performed. Input/source fidelity, model state, native parity samples and metric arithmetic were verified.']
    (out/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('correction');p.add_argument('output');a=p.parse_args()
    r=analyze(a.correction,a.output);print(json.dumps(dict(models=r['models'],turbo_parity=r['turbo_parity']),indent=2))
