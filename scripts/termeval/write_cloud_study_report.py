"""Write the review-facing report from accepted aggregate receipts."""
import csv
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'results/night1'


def main():
    data=json.loads((OUT/'cloud_decoder_validated_20261013.json').read_text())
    assert data['status']=='independently_validated'
    modes=[('hf_beam5','Small · HF beam5'),('official_default','Small · official'),
           ('hf_turbo_beam5','Turbo · HF beam5'),('official_turbo','Turbo · official')]
    cohorts=['lahaja_replication','lahaja_fresh_audio','commands_repeat_1','commands_repeat_5','commands_repeat_8']
    lines=['# Cloud ASR study: results and research decision',
        '\nC. Kaustubh · 13 October 2026 · Prepared for the October 16 review',
        '\n**Status:** Completed runs below passed independent prediction-level validation. This is an inference study, not a trained-model breakthrough. The Hindi results do not establish Indian-English performance.',
        '\n## Corrected baseline comparison',
        '\nOur earlier decoding recipe overstated the room for a custom loop repair. Official Whisper Small scores 60.31% WER on the 405-clip replication set and 62.13% on 396 additional recordings, versus 79.41% and 74.84% with the earlier HF five-beam recipe. The fresh-recording difference is −12.71 points, paired speaker-bootstrap 95% CI [−19.22, −6.72]. These recipes differ in several settings and implementation details; this is not a one-knob attribution.',
        '\n| Model and decoding recipe | Hindi replication (405) | Hindi fresh audio (396) | Keywords ×1 (100) | Keywords ×5 (100) | Keywords ×8 (100) |',
        '|---|---:|---:|---:|---:|---:|']
    csvrows=[]
    for mode,label in modes:
        if mode not in data['runs']:
            lines.append(f'| {label} | Pending | Pending | Pending | Pending | Pending |');continue
        run=data['runs'][mode]
        values=[run['cohorts'][c]['first_wer_percent'] for c in cohorts]
        lines.append('| '+label+' | '+' | '.join(f'{v:.2f}%' for v in values)+' |')
        for c in cohorts:
            v=run['cohorts'][c]
            csvrows.append(dict(condition=mode,cohort=c,clips=v['clips'],reference_words=v['reference_words'],
                word_errors=v['first_word_errors'],wer_percent=v['first_wer_percent'],
                repair_wer_percent=v['second_wer_percent'],helped=v['helped'],harmed=v['harmed']))
    lines += ['\nWER may exceed 100% when insertion errors exceed the reference-word count. All such clips remain in the denominator. Scores use NFC/case/punctuation normalization, without script conversion or numeral equivalence.',
        '\n## Repairs: aggregate improvement is not preservation',
        '\nWhisper Turbo HF outputs on the constructed repeated-speech controls:',
        '\n| Rule | Five-repeat WER | Five-repeat clips harmed | Eight-repeat WER | Eight-repeat clips harmed |',
        '|---|---:|---:|---:|---:|']
    rules=[('Raw output','hf_turbo_beam5',False),('Collapse ≥5 identical words','hf_turbo_beam5',True),
           ('Zero-duration alignment gate','alignment_zero_turbo',True)]
    for label,mode,repaired in rules:
        a=data['runs'][mode]['cohorts']['commands_repeat_5'];b=data['runs'][mode]['cohorts']['commands_repeat_8']
        field='second_wer_percent' if repaired else 'first_wer_percent'
        lines.append(f"| {label} | {a[field]:.2f}% | {a['harmed'] if repaired else 0}/100 | {b[field]:.2f}% | {b['harmed'] if repaired else 0}/100 |")
    caps=data['published_duration_character_cap']['hf_turbo_beam5']
    a=caps['commands_repeat_5'];b=caps['commands_repeat_8']
    lines.append(f"| Published duration cap | {a['second_wer_percent']:.2f}% | {a['harmed']}/100 | {b['second_wer_percent']:.2f}% | {b['harmed']}/100 |")
    lines += ['\nThe text-collapse rule improves pooled five-repeat WER while harming 77/100 clips. A few huge insertion reductions dominate the aggregate. The alignment gate reduces the harm but still damages genuinely supported repetitions. The duration cap is a prior-work baseline and shows no observed clip-level harm on these two constructed sets; this does not prove safety elsewhere.',
        '\n![Pooled WER versus damage to repeated speech](figures/repetition_tradeoff.svg)',
        '\n[Figure with caption](figures/repetition_tradeoff.caption.txt) · [PDF figure](figures/repetition_tradeoff.pdf) · [Offline audio demonstration](demo/index.html)',
        '\n## Why the loop-only branch is closed for the proper Small baseline',
        '\nThe official Small decoder has **zero five-word-loop triggers across all 801 Hindi recordings**. Any repair constrained to leave a hypothesis unchanged when that trigger is absent must therefore return the original hypothesis on every recording here. Formally, if `T(h_i)=0` for every evaluated `i` and `R(x,h)=h` whenever `T(h)=0`, then `R(x_i,h_i)=h_i` for every `i`; all edit counts and pooled WER are exactly unchanged. This is a finite-sample identity, not a claim that repetition never occurs in other data.',
        '\nA simulated loop-trigger cascade from HF Small to Turbo scores 61.41% on fresh Hindi, but its difference from official Small has a 95% interval crossing zero [−4.11, +2.91]. It adds a larger model without establishing an improvement over the proper Small baseline. The zero-duration gate on Turbo likewise has a fresh-Hindi gain whose interval reaches zero. Neither is accepted as the research contribution.',
        '\n## Natural-reference repetition check']
    natural_path=OUT/'natural_repetition_validated_20261013.json'
    if natural_path.exists():
        natural=json.loads(natural_path.read_text());assert natural['status']=='independently_validated'
        lines += ['\nAll 27 non-development-speaker Lahaja recordings with at least three adjacent identical reference words were evaluated (24 speakers; six recording IDs overlap earlier evaluation). Only one reference has five or more repetitions. These are released annotations, not newly adjudicated listening labels.',
            '\n| Official decoder | Raw WER | Text-collapse WER | Clips helped / harmed |',
            '|---|---:|---:|---:|']
        for mode,r in natural['runs'].items():
            c=r['cohorts']['natural_hindi_repetition']
            lines.append(f"| {mode} | {c['first_wer_percent']:.2f}% | {c['second_wer_percent']:.2f}% | {c['helped']} / {c['harmed']} |")
        lines.append('\nThe one-clip Small improvement is inconclusive; its interval reaches zero. This selected set cannot estimate a population failure rate. [Natural-control evidence](natural_repetition_validated_20261013.json) · [Selection protocol](NATURAL_REPETITION_PROTOCOL_20261013.md).')
    else:lines.append('\nNot yet independently validated.')
    lines.append('\n## Short-Hindi follow-up')
    short_path=OUT/'short_hindi_validated_20261013.json'
    if short_path.exists():
        short=json.loads(short_path.read_text());assert short['status']=='independently_validated'
        lines += ['\nThe one-setting ablation compares official Small with its no-speech skip disabled. Both use the same A100 hardware, pinned model and 654-input manifest.',
            '\n| Condition | Short-word WER / exact accuracy (414) | Empty short-word outputs | Five-repeat Hindi WER (102) | Eight-repeat Hindi WER (102) | Silence inputs emitting words (36) |',
            '|---|---:|---:|---:|---:|---:|']
        for mode,r in short['runs'].items():
            c=r['cohorts'];a=c['hindi_short_natural']['raw']
            lines.append(f"| {mode} | {a['wer_percent']:.2f}% / {a['exact_percent']:.2f}% | {a['empty']} | {c['hindi_repeat_5']['raw']['wer_percent']:.2f}% | {c['hindi_repeat_8']['raw']['wer_percent']:.2f}% | {c['silence_control']['raw']['emitted']}/36 |")
        if short['no_speech_skip_ablation']:
            c=short['no_speech_skip_ablation']['hindi_short_natural'];ci=c['paired_speaker_bootstrap_95_ci_points']
            lines.append(f"\nDisabling skipping changes short-word WER by {c['difference_points']:+.2f} points, paired speaker-bootstrap 95% CI [{ci[0]:+.2f}, {ci[1]:+.2f}]; {c['helped']} clips improve and {c['harmed']} worsen. Silence has no reference-word denominator, so its WER is undefined and is never pooled into speech WER.")
        lines.append('\n[Full short-Hindi evidence](short_hindi_validated_20261013.json) · [Frozen protocol](SHORT_HINDI_PROTOCOL_20261013.md)')
    else:lines.append('\nStill running. No short-Hindi quality claims are made until complete artifacts pass validation.')
    word_path=OUT/'short_word_replication_20261013.json'
    if word_path.exists():
        d=json.loads(word_path.read_text())
        lines += ['\n### Secondary isolated-word diagnostic',
            '\nThis assumes the task requires exactly one word; it is not a general transcription method. All conditions select the most frequent normalized hypothesis word, resolving ties by first occurrence. No reference word is used for selection.',
            '\n| Decoder | Original accuracy | Five-copy accuracy | Eight-copy accuracy |',
            '|---|---:|---:|---:|']
        for mode,c in d['conditions'].items():
            lines.append(f"| {mode} | {c['baseline_accuracy_percent']:.2f}% | {c['copies']['5']['accuracy_percent']:.2f}% | {c['copies']['8']['accuracy_percent']:.2f}% |")
        lines.append('\n[Complete paired effects and time costs](short_word_replication_20261013.json) · [Diagnostic specification](SHORT_WORD_REPLICATION_DIAGNOSTIC_20261013.md). These 102 examples have one source speaker each; this is exploratory, not proof of novelty or independent confirmation.')
    vocabulary_path=OUT/'vocabulary_validated_20261013.json'
    if vocabulary_path.exists():
        v=json.loads(vocabulary_path.read_text());assert v['status']=='independently_validated'
        lines += ['\n## Fixed-vocabulary diagnostic',
            '\nThis tests isolated-word recognition with 92 candidate words from development speakers, plus the original ASR output and an empty candidate. The vocabulary covers 196/414 evaluation words. The remaining 218 clips and all 36 silence controls are retained. No evaluation reference enters the candidate list.',
            '\n| Candidate objective | All-speech WER | All-speech exact accuracy | In-vocabulary accuracy (196) | Out-of-vocabulary accuracy (218) | Silence emissions (36) |',
            '|---|---:|---:|---:|---:|---:|']
        for mode in ('baseline','sum_ll','mean_ll','sum_prior_corrected','mean_prior_corrected'):
            c=v['cohorts'];a=c['all_speech'][mode]
            lines.append(f"| {mode} | {a['wer_percent']:.2f}% | {a['exact_percent']:.2f}% | {c['in_vocabulary'][mode]['exact_percent']:.2f}% | {c['out_of_vocabulary'][mode]['exact_percent']:.2f}% | {c['silence'][mode]['emitted']}/36 |")
        c=v['contrasts']['all_speech']['mean_prior_corrected_minus_mean_ll']['exact'];ci=c['paired_speaker_bootstrap_95_ci_points']
        lines.append(f"\nThe specified primary contrast, mean prior correction minus mean likelihood, changes exact accuracy by {c['difference_points']:+.2f} points (paired speaker-bootstrap 95% CI [{ci[0]:+.2f}, {ci[1]:+.2f}]). It gains {c['correct_gained']} correct outputs and loses {c['correct_lost']}.")
        lines.append('\nA useful limitation follows directly from the candidate set: if the true word lies outside the fixed vocabulary, the original ASR output is the only candidate that could be correct. This selector can preserve or damage out-of-vocabulary exact accuracy, but cannot increase it. Changes in total WER must therefore be read alongside exact-word accuracy and coverage. This is a known constrained-recognition/prior-correction diagnostic, not a new unrestricted ASR architecture. [Validated candidate evidence](vocabulary_validated_20261013.json) · [Protocol and conditional mathematical identity](VOCABULARY_PROTOCOL_20261013.md).')
    lines += ['\n## Limits and reproducibility',
        '\nThe fresh Hindi recordings share speakers with earlier evaluation. No population fairness claim is justified. Repeated-speech controls concatenate recorded words and are not naturally repeated conversation. Silence controls are artificial/calibration floors, cropped explicitly and deduplicated. One decoded run is used per condition; bootstrap intervals describe sampling uncertainty over speakers, not seed variance, and are not multiplicity-adjusted.',
        '\nAll neural inference ran on cloud GPUs: HF A100 or Modal H100 for decoding, with a native Modal L4 App for the candidate-scoring follow-up. Hardware differs across recipe comparisons, so raw timing ratios are not decoder speedups. Per-recording seeds, model/package revisions, code commits, manifests, audio hashes, hypotheses, complete word edits and prediction hashes are saved. The independent validator refuses incomplete bundles, corrupted hashes or mismatched cohorts. Gated references remain private. Modal Sandboxes are disabled; new Modal runs use bounded Apps with no warm workers.',
        '\nA repeatability check on the six recordings shared by the natural-repetition extension and earlier runs found minor differences despite matching audio, references, decoder settings, packages, GPU type and per-recording seed policy. Small changed two hypotheses (98 → 99 errors over 153 words); Turbo changed one (105 → 104). The cause is not isolated. These single-run scores must not be interpreted as bitwise deterministic, and the speaker-bootstrap intervals do not include this run-to-run variation. [Repeatability receipt](cloud_repeated_recordings_20261013.json).',
        '\n[Full aggregate evidence](cloud_decoder_validated_20261013.json) · [Main protocol and prior work](CLOUD_PROTOCOL_20261013.md) · [Validation code](../../scripts/termeval/package_cloud_decoder_study.py) · [Short-study validation](../../scripts/termeval/package_short_hindi_study.py)',
        '\n## Research decision',
        '\nThe defensible result so far is a decoder-and-evaluation audit: the apparent repair gain depends strongly on the baseline recipe, and pooled WER can conceal deletion of valid repeated speech. This is real, reproducible evidence, but it does not yet establish a novel deployable improvement or guarantee a publication. Keep the properly configured decoder as the baseline, report failed controls, and use any positive short-word result only to justify a separately held-out follow-up. Do not launch training merely to spend the remaining credits.']
    (OUT/'CLOUD_RESULTS_20261013.md').write_text('\n'.join(lines)+'\n')
    with (OUT/'cloud_decoder_comparison_20261013.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(csvrows[0]));w.writeheader();w.writerows(csvrows)
    print(OUT/'CLOUD_RESULTS_20261013.md')


if __name__=='__main__':main()
