"""Generate a compact closeout from validated results, without launching compute."""
import hashlib
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/'research/bridge_open';ART=BASE/'artifacts'


def read(p):return json.loads(Path(p).read_text())


def main():
    whisper=read(ART/'fleurs_whisper_corrected_v3/comparison.json')
    qwen=read(ART/'fleurs_qwen_comparison_v1/comparison.json')
    review=read(ART/'benchmark_review/build_receipt.json')
    assert review['records']==2322 and review['predictions']==5062
    ledger=read(Path.home()/'.local/share/openresearch/whispr-bridge-open/hf-budget-20260922.json')
    reserve=sum(r['reserved_usd'] for r in ledger['reservations'].values());assert reserve<=50
    budget=dict(provider='huggingface',cap_usd=50,reserved_usd=reserve,remaining_unreserved_usd=50-reserve,
                reservation_count=len(ledger['reservations']),provider_bill_verified=False,measured_bill_usd=None,
                note='Conservative reservations retained for all launches; these are not measured billing or remaining account credit.')
    (ART/'benchmark_review/budget_receipt.json').write_text(json.dumps(budget,indent=2)+'\n')
    validation=dict(cpu_existing_benchmark_checks_passed=53,retired_long_dispatch_guard_checks_passed=1,
                    qwen_runner_and_trace_checks_passed=7,
                    gpu_serial_batch_diagnostic_text_matches=3,gpu_serial_batch_diagnostic_n=3,
                    qwen_full_validation=qwen['qwen_audit'],
                    inference_scope='No human semantic transcript review performed; source/audio/model/trace/arithmetic checks passed.')
    (ART/'benchmark_review/validation_receipt.json').write_text(json.dumps(validation,indent=2)+'\n')
    lines=['# Completed ASR benchmark and evidence package', '', 'Prepared for **C. Kaustubh**. This closes the FLEURS pipeline correction and the focused Qwen Hindi extension. It contains **5,062 primary predictions over 2,322 distinct recordings**: two Whisper systems on all three languages, plus Qwen3-ASR-1.7B on all 418 Hindi recordings.', '', '| Declared system | Hindi (418) | Bengali (920) | Assamese (984) |', '|---|---:|---:|---:|']
    for label in ['Turbo','Large-v3']:
        vals=[whisper['models'][label]['languages'][lang]['pooled_duration_routed']['micro_wer'] for lang in ['hi','bn','as']]
        lines.append(f'| Whisper {label} | '+ ' | '.join(f'{v:.2%}' for v in vals)+' |')
    lines.append(f"| Qwen3-ASR-1.7B | {qwen['summary']['Qwen3-ASR-1.7B']['micro_wer']:.2%} | Not evaluated | Not evaluated |")
    lines += ['', 'Values are micro WER, including observed model-quality failures. The same pinned source files, references and primary scorer are used for shared cohorts. WER can exceed 100% through insertions.', '', '## What was corrected and checked', '', '- Verified all 2,322 WAVs and reference joins against the pinned FLEURS source. Files are already mono 16 kHz; independently decoded waveforms agree. No sample-rate relabeling or extra resampling was needed.', '- Matched native/HF Whisper weights, evaluation state and frontend features on the diagnostic cohorts. Short-form batch, single-example and pipeline results reproduced the saved outputs. Native checks reproduced the severe Assamese failures.', '- Replaced independent fixed 30-second splits for **every long recording**: 14 per model, not only favorable examples. Full-waveform native transcription now handles them. All 2,308 short recordings per Whisper model retain the audited HF recipe.', '- Qwen uses its documented native transcribe API with explicit Hindi, complete waveforms, actual bfloat16/evaluation state, frozen greedy generation and batch size up to eight. Serial and batched diagnostics matched exactly on all three recordings before full dispatch.', '- Recovered exact source snapshots, transport hashes, configurations, model/data hashes, predictions, raw Qwen generation traces and receipts. Independently rescored every accepted Qwen output and all corrected Whisper outputs.', '- Passed 54 local benchmark checks and seven Qwen runner/trace checks. The old long-audio launcher requires an explicit historical-diagnostic override; ordinary dispatch is blocked.', '', '## Read and inspect', '', '- [Offline transcript and audio viewer](artifacts/benchmark_review/index.html): all references and predictions, per-clip scores, decoding routes and script diagnostics. Audio playback uses the local verified FLEURS data.', '- [Full Hindi comparison and paired intervals](artifacts/fleurs_qwen_comparison_v1/RESULTS.md).', '- [Whisper comparison, including separate short/long strata](artifacts/fleurs_whisper_corrected_v3/RESULTS.md).', '- [Input/source audit](artifacts/fleurs_input_audit/CONCLUSIONS.md), [native Whisper audit](FLEURS_DILIGENCE_REPORT.md), and [scorer sensitivity](artifacts/fleurs_scoring_audit/REPORT.md).', '- [Qwen batch acceptance](artifacts/qwen_native_batch_canary_v1/BATCH_ACCEPTANCE.json), [full-run validation](artifacts/qwen_native_full_hi_v1/independent_validation.json), and [validation receipt](artifacts/benchmark_review/validation_receipt.json).', '- [Unannotated human review queue](artifacts/benchmark_review/human_review_queue.json): low, middle and high observed WER examples for each evaluated model/language cell. This is a diagnostic selection, not a representative quality sample.', '', '## Completed GPU jobs', '']
    for name,folder in [('Whisper correction + Turbo parity','fleurs_native_long_correction'),('Qwen serial acceptance','qwen_native_canary_v1'),('Qwen batch acceptance','qwen_native_batch_canary_v1'),('Qwen full Hindi','qwen_native_full_hi_v1')]:
        r=read(ART/folder/'orx_source_receipt.json')
        assert r['provider_status']=='done'
        lines.append(f"- [{name}]({r['url']}) — source commit `{r['commit'][:12]}`, preserved in `{folder}/source_snapshot.tar.gz`.")
    lines += ['', f"**Budget:** ${reserve:.2f}/${50:.2f} conservatively reserved in the shared HF ledger; ${50-reserve:.2f} unreserved. Actual provider billing has not been verified. No new Modal spend was made for this closeout. [Budget receipt](artifacts/benchmark_review/budget_receipt.json).", '', '## Interpretation and remaining research work', '', 'This is a completed, reproducible descriptive benchmark of these configurations. It provides evidence of output failures and decoding sensitivity. It does not yet establish a new ASR method or the novelty required for a publication.', '', 'The Whisper pooled scores use a duration-based routing rule; the report preserves separate strata and the historical fixed-split sensitivity results. This correction was made after inspecting test results and is not a held-out method gain. Qwen and Whisper differ in precision, search and API postprocessing, so a ranking here compares declared systems, not architectures under identical decoding.', '', 'FLEURS is read speech. This study does not reproduce Humyn’s conversational BRIDGE audit or establish Indian-English, speaker-demographic fairness, mobile-device performance, or universal best-model claims. Sentence-cluster intervals do not provide speaker/population uncertainty. Native Whisper EOS/cap status is unknown; observed short-form caps remain included.', '', 'Before a paper claim is made, the remaining necessary work is independent linguistic listening/reference review and a clearly scoped contribution checked against prior work. Earlier Humyn public-sample comparisons affected by silence/reference problems remain withdrawn. The results here should not be presented as a finished submission or as all current open ASR models.', '', '## Local replay', '', 'From `/Users/prateekraj/whispr-nano`, the analysis commands require no GPU:', '', '```sh', '.venv-bridge/bin/python -m research.bridge_open.finish_qwen_comparison research/bridge_open/artifacts/qwen_native_full_hi_v1 /tmp/qwen-comparison-recheck', '.venv-bridge/bin/python -m research.bridge_open.finish_whisper_comparison research/bridge_open/artifacts/fleurs_native_long_correction /tmp/whisper-comparison-recheck', '```', '', 'Choose output paths that do not already exist. Portable source archives retain the exact executed code. Frozen experiment nodes remain unchanged; any future inference variant must be a new child.']
    (BASE/'BENCHMARK_COMPLETION.md').write_text('\n'.join(lines)+'\n')
    files=[p for folder in ['fleurs_native_long_correction','fleurs_whisper_corrected_v3','qwen_native_canary_v1','qwen_native_batch_canary_v1','qwen_native_full_hi_v1','fleurs_qwen_comparison_v1','benchmark_review'] for p in (ART/folder).rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.name!='evidence_index.json']
    evidence={str(p.relative_to(BASE)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(files)}
    (ART/'benchmark_review/evidence_index.json').write_text(json.dumps(evidence,indent=2)+'\n')
    return dict(report=str(BASE/'BENCHMARK_COMPLETION.md'),evidence_files=len(evidence),reserved_usd=reserve)

if __name__=='__main__':print(json.dumps(main(),indent=2))
