"""Write the current study report from accepted saved artifacts."""
import hashlib
import json
import shutil
from pathlib import Path
from research.bridge_open.analyze_fleurs import _aggregate

ROOT = Path(__file__).resolve().parents[2]
ART = ROOT / 'research/bridge_open/artifacts'


def load(p): return json.loads(Path(p).read_text())
def lines(p): return [json.loads(x) for x in Path(p).read_text().splitlines()]


def main():
    out = ART / 'expanded_study_v1'; out.mkdir(exist_ok=True)
    omni = load(ART / 'fleurs_omni_comparison_v1/comparison.json')
    diagnostic = load(ART / 'language_conditioning_analysis_v1/analysis.json')
    negative = load(out / 'negative_validation_checks.json')
    assert len(negative) == 8 and all(r['rejected'] for r in negative)
    models = {}
    for name, file in [('Whisper Turbo','turbo_predictions.jsonl'), ('Whisper Large-v3','large-v3_predictions.jsonl')]:
        rows=lines(ART/'fleurs_whisper_corrected_v3'/file)
        models[name]={lang:_aggregate([r for r in rows if r['language']==lang]) for lang in ('hi','bn','as')}
    models['Qwen3-ASR-1.7B']={'hi':load(ART/'fleurs_qwen_comparison_v1/comparison.json')['summary']['Qwen3-ASR-1.7B']}
    models['OmniASR CTC-1B']=omni['groups']
    jobs=[]
    for directory in ['whisper_language_conditioning_v1','omni_fleurs_acceptance_v1','omni_fleurs_full_hi_v1','omni_fleurs_full_bn_v1','omni_fleurs_full_as_v1']:
        receipt=load(ART/directory/'orx_source_receipt.json');assert receipt['provider_status']=='done'
        jobs.append(dict(directory=directory,**receipt))
    ledger=load(Path.home()/'.local/share/openresearch/whispr-bridge-open/hf-budget-20260922.json')
    assert ledger['cap_usd'] is None
    (out/'hf_reservation_receipt.json').write_text(json.dumps(ledger,indent=2)+'\n')
    (out/'study_summary.json').write_text(json.dumps(dict(models=models,jobs=jobs,primary_recordings=2322,
        primary_predictions=7384,new_jobs=5,measured_cost_usd=None,human_review='pending',
        budget_total_cap=None,new_conservative_reservations_usd=7.5),indent=2)+'\n')
    report=['# Expanded ASR study — 23 September 2026', '',
        'Prepared for C. Kaustubh. All five jobs in this expansion completed successfully. The study now contains 7,384 primary predictions over 2,322 distinct FLEURS official-test recordings, plus a separate 44-recording paired Whisper diagnostic and nine-recording Omni implementation check. These subsets overlap the benchmark and are not additional independent data.', '',
        '| System | Hindi, n=418 | Bengali, n=920 | Assamese, n=984 |',
        '|---|---:|---:|---:|']
    for name,g in models.items():
        report.append('| '+name+' | '+' | '.join(f"{g[l]['micro_wer']:.2%}" if l in g else 'Not evaluated' for l in ('hi','bn','as'))+' |')
    report += ['', 'Entries are micro WER; lower is better. WER can exceed100% because insertions count as errors. These are comparisons of declared system configurations, not changes to a single trained model. No new weights were trained in this expansion.', '',
        'Omni is the strongest evaluated Bengali/Assamese system; Qwen remains strongest on Hindi. Relative to Whisper Large-v3, Omni reduces Bengali WER by50.19percentage points (sentence-cluster95% interval 46.20–56.19) and Assamese WER by79.55points (74.67–86.42). These intervals resample source sentences within these fixed datasets; they do not establish speaker-population or demographic fairness.', '',
        'Omni has zero dominant tracked-script mismatches on Bengali and Assamese, but17 on Hindi. This is a Unicode diagnostic and does not replace human evaluation of transcript language or meaning.', '',
        '## Controlled language diagnostic', '',
        'Whisper Large-v3 used the same native checkpoint, float32/eval state, complete waveform, seed and decoding options in both arms; only forced versus automatic language selection changed. Ten short clips per language were chosen by fixed hash order, plus all14 long clips. The short and long strata are reported separately in the detailed analysis.', '',
        '| Diagnostic language | n | Forced WER | Automatic WER |', '|---|---:|---:|---:|']
    for g in diagnostic['groups']:
        if g['scope']=='all_diagnostic':report.append(f"| {g['language']} | {g['forced']['n']} | {g['forced']['micro_wer']:.2%} | {g['automatic']['micro_wer']:.2%} |")
    report += ['', 'Automatic mode selected Urdu for4Hindi clips; Bengali for13Assamese clips; and Sinhala for one Assamese clip. The tracked-script heuristic reports that Sinhala output as unknown, not correct. Bengali outputs were unchanged. Automatic selection therefore does not provide a general repair. Four recordings had native timestamp bounds outside the waveform tolerance in both arms; those outputs remain in the analysis. Native public decoding strips EOS, so generation-cap counts remain unknown.', '',
        '## Validation and reproducibility', '',
        'The frozen manifest has SHA-256967d8dca02f9f729c1ed7548dce7283cf9dfc62623815a7c131f71acb13ce2c7. Each source WAV/reference hash, sample rate, channel count, finite sample range and duration was checked. No resampling was needed: the released inputs were already mono16kHz. All source archives and model assets were pinned and hashed.', '',
        'Omni ran the pinned official fairseq2/OmniASR CTC implementation in BF16/eval mode, one complete waveform per recording. All nine acceptance examples matched the official public API exactly, including a long recording per language. Each full shard used identical native inference code. Raw CTC IDs, collapse evidence, source snapshots, configurations, metrics and predictions are retained. Local independent rescoring and eight corruption-rejection checks passed. Empty outputs and runtime failures are never silently removed; all configured IDs were required.', '',
        'Whisper primary results retain the previously audited HF float32 beam5 short-form route and native full-waveform long-form correction. Qwen uses its native BF16 greedy API, batches up to eight and explicit Hindi. Inference precision, decoding and training differ across systems. The primary scorer preserves Indic marks and applies no numeral conversion.', '',
        '## Deliverables', '',
        '- [Transcript/audio viewer](../benchmark_review_v2/index.html)',
        '- [Blinded listening review](../../listening_review/index.html):50utterances, all unreviewed; autosaved forms and JSON/CSV exports. This frozen packet contains the previous Whisper/Qwen outputs.',
        '- [Omni results and paired intervals](../fleurs_omni_comparison_v1/RESULTS.md)',
        '- [Language diagnostic](../language_conditioning_analysis_v1/RESULTS.md)',
        '- [Comparison figure](../fleurs_omni_comparison_v1/figures/system_comparison.pdf)',
        '- [Prior-art assessment](PRIOR_ART.md)', '',
        '## Jobs and spending', '',
        'The user removed the total budget cap for this expansion. Five explicit A100 jobs retained30-minute provider limits; the shared ledger added$7.50in conservative reservations. Reservations are not measured spending, and provider billing has not been verified. No job in this batch remains running. The full-worker inference guard was1500seconds including model loading; an earlier sizing note said1450seconds, and the executed configuration is authoritative.', '',
        '| Run | Source commit | Job |', '|---|---|---|']
    for j in jobs:report.append(f"| {j['directory']} | {j['commit'][:7]} | [Completed]({j['url']}) |")
    report += ['', '## Publication status and next decision', '',
        'This is a completed benchmark expansion with an auditable negative result for automatic language selection. It is not yet evidence of publishable novelty. Wrong-script ASR output has direct prior art; see[Script collapse in multilingual ASR](https://arxiv.org/abs/2604.08786). A three-language read-speech benchmark does not establish Indian-English accent fairness, conversational robustness, phone performance, or full replication of Humyn BRIDGE.', '',
        'The next acceptance gate is a fluent human review of the frozen examples, followed by an explicit contribution statement tested against that prior work. Any proposed repair should be frozen and evaluated on independent data. IndicConformer remains unavailable because this HF account does not have gated-file access (403); its preparation is not a completed baseline. No messages, model terms, paper submission or external publication were performed.', '']
    # Readable spacing for prose assembled above, preserving hashes/numbers.
    text='\n'.join(report)
    for old,new in [('exceed100','exceed 100'),('by50','by 50'),('points (sentence-cluster95','points (sentence-cluster 95'),('WER by79','WER by 79'),('50.19percentage','50.19 percentage'),('79.55points','79.55 points'),('percentages','percentages'),('but17','but 17'),('all14','all 14'),('for4Hindi','for 4 Hindi'),('for13Assamese','for 13 Assamese'),('SHA-256967','SHA-256 967'),('mono16kHz','mono 16 kHz'),(':50utterances',': 50 utterances'),('retained30-minute','retained 30-minute'),('added$7.50in','added $7.50 in'),('was1500seconds','was 1500 seconds'),('said1450seconds','said 1450 seconds'),('see[','see [')]:text=text.replace(old,new)
    (out/'RESULTS.md').write_text(text)
    (ROOT/'research/bridge_open/EXPANDED_STUDY_20260923.md').write_text(text.replace('](../../listening_review/','](listening_review/').replace('](../','](artifacts/').replace('](PRIOR_ART.md)','](artifacts/expanded_study_v1/PRIOR_ART.md)'))
    return out


if __name__=='__main__':print(main())
