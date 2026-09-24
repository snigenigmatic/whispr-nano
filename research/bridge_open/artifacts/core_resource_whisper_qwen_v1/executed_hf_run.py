"""Container entry point for the fixed five-model resource suite."""
import hashlib, json, sys, time, traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def validate(cfg):
    from research.bridge_open.core_study.run_resource_suite import validate_config
    validate_config(cfg); return cfg
def main():
    import torch
    from research.bridge_open.artifact_transport import emit
    from research.bridge_open.core_study.run_resource_suite import run
    started=time.monotonic(); source=ROOT/'research/bridge_open/core_study/resource_suite_config.json'
    cfg=json.loads(source.read_text()); out=ROOT/'artifacts'/cfg['cloud_job_id']; failure=None
    validation={'valid':False,'errors':['suite did not start']}
    try:
        if torch.__version__.split('+')[0]!='2.8.0' or not torch.cuda.is_available(): raise RuntimeError('Requires PyTorch 2.8.0 CUDA')
        torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
        validate(cfg); cfg['output_dir']=str(out); effective=ROOT/'effective_resource_suite_config.json'; effective.write_text(json.dumps(cfg,indent=2)+'\n')
        run(effective)
        summary=json.loads((out/'overall_summary.json').read_text()); incomplete=[m['key'] for m in summary['models'] if m['status']!='completed']
        validation={'valid':not incomplete,'errors':([] if not incomplete else [f'model failures retained: {incomplete}']),'completed_models':len(summary['models'])-len(incomplete),'failed_models':incomplete}
    except Exception as exc:
        failure={'type':type(exc).__name__,'message':str(exc),'traceback':traceback.format_exc()}; validation={'valid':False,'errors':[f'worker failure: {type(exc).__name__}: {exc}']}
    finally:
        out.mkdir(parents=True,exist_ok=True)
        receipt={'schema_version':'container-receipt/v1','provider':'huggingface','command':'bash experiments/hf_entrypoint.sh','gpu_flavor':'a100-large','gpu_contract':'NVIDIA A100 80GB','timeout_seconds':1800,'gpu':torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,'torch_version':torch.__version__,'cuda_version':torch.version.cuda,'elapsed_seconds':time.monotonic()-started,'worker_failure':failure,'source_sha256':{str(p.relative_to(ROOT)):sha(p) for p in (Path(__file__),source,ROOT/'experiments/hf_entrypoint.sh',ROOT/'research/bridge_open/core_study/run_resource_suite.py',ROOT/'research/bridge_open/core_study/resource_profile.py')}}
        (out/'container_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
        (out/'remote_validation.json').write_text(json.dumps(validation,indent=2)+'\n')
        (out/'artifact_transport.json').write_text(json.dumps({'schema_version':'artifact-transport/v1','transport':'single zlib/base64 bundle emitted after all top-level artifacts','top_level_only':True},indent=2)+'\n')
        try: emit(out)
        except Exception as exc: (out/'artifact_transport_failure.json').write_text(json.dumps({'type':type(exc).__name__,'message':str(exc)},indent=2)+'\n')
    print(json.dumps({'event':'resource_suite_complete','validation':validation}),flush=True)
    return 0 if validation['valid'] else 2
if __name__=='__main__': raise SystemExit(main())
