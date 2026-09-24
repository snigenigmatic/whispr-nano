"""Build an allowlisted local evidence index; no agents or experiments run.

This is a project-owned handoff format, not an upstream freephdlabor config.
It reads no credentials and makes no network, LLM or cloud calls.
"""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
BRIDGE = ROOT / 'research/bridge_open'
RUNS = ['hindi-turbo-cuda-beam5-v1', 'turbo-multilingual-beam5-v1',
        'small-multilingual-beam5-v1', 'hindi-qwen3-asr-0-6b-v1',
        'hindi-qwen3-asr-1-7b-v1', 'hindi-whisper-large-v3-beam5-v1',
        'omni-ctc-1b-multilingual-v2']


def main():
    names = ['FULL_STUDY_SPEC.md', 'PROTOCOL.md', 'metrics_contract.json',
             'scoring.py', 'validate_run.py', 'analyze_benchmark.py', 'study_panels.py',
             'EXPANDED_RESULTS_20260922.md', 'artifacts/corpus_manifest_v1.jsonl',
             'artifacts/expanded_benchmark/comparison.json',
             'artifacts/expanded_benchmark/validation.json',
             'artifacts/expanded_diagnostics/script_audit.json',
             'artifacts/study_panels/panels.json',
             'artifacts/budget_expansion_snapshot_20260922.json',
             'orchestration/TASK_BRIEF.md']
    for run in RUNS:
        names.extend(f'artifacts/{run}/{name}' for name in
                     ['config.json', 'environment.json', 'predictions.jsonl',
                      'summary.json', 'cloud_receipt.json'])
    inputs=[]
    for name in names:
        path = BRIDGE / name
        payload = path.read_bytes()
        inputs.append(dict(path=str(path.relative_to(ROOT)), bytes=len(payload),
                           sha256=hashlib.sha256(payload).hexdigest()))
    packet = dict(schema='whispr_asr_review_packet_v1', upstream_config=False,
        inspected_upstream_commit='9102d18a161037294d3d963b2799351aa724c58b',
        purpose='Review existing evidence and specify one decoder control before any new run',
        mode='local_index_only', input_root=str(ROOT), inputs=inputs,
        execution_enabled=False, new_external_llm_api_budget_usd=0,
        modal_cap_usd=15, modal_reservations_usd=15,
        budget_note='Reservations are not charges; reconcile provider billing before reserving further jobs.',
        validation_note='Hashes bind this review packet to file contents; this builder does not rerun scientific validation.',
        outputs_requested=['evidence-backed reviewer issues', 'one falsifiable experiment specification',
                           'claims supported and unsupported by current data'])
    output = HERE / 'review_packet.json'
    output.write_text(json.dumps(packet, indent=2)+'\n')
    print(json.dumps(dict(output=str(output), indexed_files=len(inputs),
                          execution_enabled=False, external_calls=0)))


if __name__ == '__main__':
    main()
