"""Prepare one frozen ORX child and launch it through the guarded HF command."""
import argparse
import json
import re
import subprocess
from pathlib import Path

MAIN = Path('/Users/prateekraj/whispr-nano')
REPO = Path('/Users/prateekraj/whispr-bridge-hf')
PROJECT = 'f8f80981-f6a4-4f34-af55-f292bfdd8708'
PARENT = '07866a77-ea10-44ab-ab3d-9e74fa85c539'
BASE = '33a032e'
REGISTRY = MAIN/'research/bridge_open/artifacts/fleurs_run_registry.json'
MODELS = {
    'turbo': ('openai/whisper-large-v3-turbo','41f01f3fe87f28c78e2fbf8b568835947dd65ed9'),
    'large': ('openai/whisper-large-v3','06f233fe06e710322aca913c1bc4249a0d71fce1'),
    'small': ('openai/whisper-small','973afd24965f72e36ca33b3055d56a652f456b4d'),
}


def command(args):
    return subprocess.check_output(args, cwd=REPO, text=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', choices=MODELS, required=True)
    p.add_argument('--language', choices=['hi','bn','as','all'], required=True)
    p.add_argument('--route', choices=['short','long'], required=True)
    p.add_argument('--start', type=int, default=0)
    p.add_argument('--end', type=int)
    p.add_argument('--legacy-fixed-long-diagnostic', action='store_true',
                   help='Explicitly allow retired fixed splitting for a labeled diagnostic only')
    args = p.parse_args()
    if args.route == 'long' and not args.legacy_fixed_long_diagnostic:
        raise ValueError('Fixed-split long-form dispatch is retired. Use the validated native protocol; see FLEURS_DILIGENCE_REPORT.md.')
    if command(['git','status','--porcelain']).strip():
        raise RuntimeError('Private HF checkout is not clean')
    cfg = json.loads(command(['git','show',f'{BASE}:research/bridge_open/run_config.json']))
    rows = [json.loads(s) for s in Path(cfg['manifest_path']).read_text().splitlines()]
    selected = [r for r in rows if (args.language=='all' or r['language']==args.language)
        and (r['duration_seconds']<=30 if args.route=='short' else r['duration_seconds']>30)]
    selected = selected[args.start:args.end]
    assert selected and args.start >= 0
    suffix = f'{args.model}-{args.language}-{args.route}-{args.start}-{args.end or "end"}-v2'
    job = 'hf-fleurs-test-' + suffix
    registry = json.loads(REGISTRY.read_text())
    if any(r.get('cloud_job_id')==job for r in registry):
        raise RuntimeError('This cohort/model launch already exists; inspect before retry')
    title = f'FLEURS test {args.model} {args.language} {args.route} {args.start} to {args.end or "end"}'
    created = command(['orx','create-experiment',PROJECT,'--parent',PARENT,'--title',title,
        '--description',f'Frozen FLEURS official-test v2 shard: {len(selected)} complete recordings. Same beam5 float32 recipe, original audio and references. Exact IDs committed before launch; no test-score tuning.'])
    exp = re.search(r'id:\s+([0-9a-f-]+)',created).group(1)
    branch = re.search(r'branch:\s+(\S+)',created).group(1)
    command(['git','switch',branch])
    for filename in ['experiments/hf_run.py','research/bridge_open/stage_fleurs.py',
            'research/bridge_open/frozen_corpus_v1.jsonl','research/bridge_open/frozen_references.json',
            'FLEURS_TEST_PROTOCOL_V1.md','FLEURS_TEST_PROTOCOL_V2.md']:
        (REPO/filename).write_bytes(subprocess.check_output(['git','show',f'{BASE}:{filename}'],cwd=REPO))
    model_id, revision = MODELS[args.model]
    # Validation-only correction; the frozen inference and scoring code remain
    # byte-identical to BASE. Record the correction in each new source snapshot.
    for filename in ['research/bridge_open/validate_run.py',
                     'research/bridge_open/FLEURS_VALIDATION_CORRECTION.md']:
        (REPO/filename).write_bytes((MAIN/filename).read_bytes())
    cfg.update(model_id=model_id,model_revision=revision,cloud_job_id=job,
        conversation_ids=[r['conversation_id'] for r in selected],
        decoding_mode='batched_shortform' if args.route=='short' else 'fixed_shortform',
        condition_id='short_utterance_beam5_float32_batch8' if args.route=='short' else 'complete_audio_fixed30s_beam5_float32',
        batch_size=8 if args.route=='short' else 1)
    (REPO/'research/bridge_open/run_config.json').write_text(json.dumps(cfg,indent=2)+'\n')
    command(['git','add','experiments/hf_run.py','research/bridge_open','FLEURS_TEST_PROTOCOL_V1.md','FLEURS_TEST_PROTOCOL_V2.md'])
    command(['git','diff','--check','--cached'])
    command(['git','commit','-m',f'Freeze {job} on {len(selected)} official test recordings'])
    commit = command(['git','rev-parse','HEAD']).strip()
    entry = dict(experiment_id=exp,branch=branch,commit=commit,cloud_job_id=job,
        phase='official_test',model_id=model_id,model_revision=revision,
        output_dir=str(MAIN/'research/bridge_open/artifacts'/job),
        conversation_count=len(selected),status='committed_pending_dispatch')
    registry.append(entry);REGISTRY.write_text(json.dumps(registry,indent=2)+'\n')
    print(json.dumps(entry),flush=True)
    result = subprocess.run([str(MAIN/'.venv/bin/python'),'experiments/launch_hf.py',exp],cwd=REPO,text=True,capture_output=True)
    print(result.stdout,flush=True)
    if result.returncode:
        entry['status']='dispatch_failed_or_ambiguous';entry['dispatch_error']=result.stderr[-2000:]
    else:
        entry['run_id']=re.search(r'run\s+([0-9a-f-]+)',result.stdout).group(1)
        entry['job_id']=re.search(r'job\s+snigenigmatic/(\S+)',result.stdout).group(1)
        entry['status']='submitted'
    REGISTRY.write_text(json.dumps(registry,indent=2)+'\n')
    if result.returncode:
        raise RuntimeError('Dispatch failed or ambiguous; reservation retained, inspect provider before retry.')


if __name__ == '__main__':
    main()
