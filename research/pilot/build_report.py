"""Render a self-contained evidence report from independently audited results."""
from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
import html
import json
from pathlib import Path
import re
import statistics

ROOT = Path(__file__).resolve().parents[2]
P = ROOT / 'research/pilot'
OUT = ROOT / 'output'


def read(path):
    return json.loads(Path(path).read_text())


def make_markdown(audit, decision, selection):
    grouped = defaultdict(list)
    for row in audit['runs']:
        grouped[row['arm_name']].append(row)
    aliases = {'full_gold': 'Full reference pool', 'filtered10_gold': 'Teacher-filtered reference pool',
               'random_matched_gold': 'Matched random reference pool'}
    english = [c['filtered_minus_random_macro_wer_pp'] for c in audit['paired_contrasts'] if c['normalization'] == 'english']
    minimal = [c['filtered_minus_random_macro_wer_pp'] for c in audit['paired_contrasts'] if c['normalization'] == 'minimal']
    complete = audit['status'] == 'valid_complete'
    if complete:
        finding = (f"Across three paired training seeds, teacher-filtered minus matched-random "
                   f"L1-macro WER was {statistics.mean(english):+.3f} percentage points under English normalization "
                   f"and {statistics.mean(minimal):+.3f} points under minimal normalization. "
                   f"Positive values favor the random control. The prespecified pilot signal criterion "
                   f"{'was met' if decision['pilot_positive'] else 'was not met'}.")
    else:
        finding = f"At the user's stop, {len(audit['runs'])} of the nine required runs had passed the independent audit. The registered contrast is incomplete; no complete-experiment scientific conclusion is available."
    verdict = ('The work provides a controlled pilot and an auditable research loop. '
               'It has not demonstrated a new distillation method against practical distillation baselines or established independent-speaker replication.')
    text = f"""# Does teacher-based filtering discard useful accented speech?

Controlled ASR pilot and research-loop assessment | 9 September 2026

## Decision and measured result

**Reviewer recommendation: do not submit in its present form.** {verdict}

This pilot compares teacher-filtered inputs with a matched random pool while training every arm on corpus-provided reference transcripts. Teacher hypotheses determine selection only; no teacher-label or teacher-distribution loss is trained. The matched control inherits the filter's speaker and L1 allocations, so the contrast concerns which recordings are retained within those allocations.

{finding}

The experiment is complete only if all nine scheduled runs pass the independent audit. A complete experiment can produce a negative or inconclusive scientific result. Final-test recordings were never decoded or scored.

## Results from the fixed development set

All scores in this table are English-normalized percentages. Rows show arithmetic means across the available training seeds; seed results follow. Unequal completed seed counts make these means descriptive, not a valid matched comparison. The final column averages each seed's worst-L1 WER. Training-pool counts are eligible examples, not the number actually encountered in the fixed update schedule.

| Training arm | Pool size | Seeds | Macro WER | Micro WER | Mean worst-L1 WER |
|---|---:|---:|---:|---:|---:|
"""
    for arm in aliases:
        runs = grouped[arm]
        values = [statistics.mean(100*r['metrics']['final']['english'][metric] for r in runs) if runs else None
                  for metric in ('macro_language_wer', 'micro_wer', 'worst_language_wer')]
        metrics = ' | '.join(f'{v:.3f}' if v is not None else 'Unavailable' for v in values)
        text += f"| {aliases[arm]} | {selection['arms'][arm]['n']:,} | {len(runs)} | {metrics} |\n"
    if audit['runs']:
        base = audit['runs'][0]['metrics']['baseline']['english']
        text += (f"\nBefore adaptation, the common initial model had macro WER {100*base['macro_language_wer']:.3f}%, "
                 f"micro WER {100*base['micro_wer']:.3f}% and worst-L1 WER {100*base['worst_language_wer']:.3f}%. "
                 "The auditor checks that every baseline prediction is identical across completed runs.\n")
    text += '\n| Seed | Filtered minus random, English (pp) | Filtered minus random, minimal (pp) |\n|---|---:|---:|\n'
    contrasts = {(c['seed'], c['normalization']): c for c in audit['paired_contrasts']}
    for seed in (0, 1, 2):
        values = []
        for norm in ('english', 'minimal'):
            c = contrasts.get((seed, norm))
            values.append(f"{c['filtered_minus_random_macro_wer_pp']:+.3f}" if c else 'Unavailable')
        text += f"| {seed} | {' | '.join(values)} |\n"
    text += '\nTraining-seed variation describes optimization randomness, not uncertainty across new speakers. There is one development speaker per L1, and one fixed random-control pool. These rows are not independent dataset replications.\n'
    if complete:
        text += '\n![Pilot outcomes and paired selection contrast](figures/pilot_contrast.png)\n\nFigure 1. Points show all three training seeds. The right panel includes their mean. The 0.5-point line is the English mean signal criterion; at least two strictly positive English seed contrasts and a nonnegative minimal mean are also required. These are prespecified pilot criteria, not statistical significance thresholds.\n'
    text += '\n### Every run and its realized training exposure\n\nWER columns are L1-macro percentages. Every completed run receives 600 updates and 4,800 unique examples. Pool matching does not exactly equalize the audio or token totals actually encountered.\n\n| Arm / seed | English WER | Minimal WER | Unique examples | Audio hours | Supervised tokens |\n|---|---:|---:|---:|---:|---:|\n'
    short = {'full_gold': 'Full', 'filtered10_gold': 'Filtered', 'random_matched_gold': 'Random'}
    for arm in aliases:
        for run in sorted(grouped[arm], key=lambda r:r['seed']):
            e = run['exposure']; m = run['metrics']['final']
            text += (f"| {short[arm]} / {run['seed']} | {100*m['english']['macro_language_wer']:.3f} | "
                     f"{100*m['minimal']['macro_language_wer']:.3f} | {e['unique_training_utterances_seen']:,} | "
                     f"{e['total_training_audio_seconds']/3600:.3f} | {e['total_supervised_tokens']:,} |\n")
    text += '\n### Language-level outcomes (English)\n\n| L1 | Initial WER | Full pool | Filtered | Matched random |\n|---|---:|---:|---:|---:|\n'
    for language in ('Arabic', 'Mandarin', 'Hindi', 'Korean', 'Spanish', 'Vietnamese'):
        key = 'language:' + language
        initial = 100*audit['runs'][0]['metrics']['baseline']['english']['groups'][key]['wer'] if audit['runs'] else None
        values = [f'{initial:.3f}' if initial is not None else 'Unavailable']
        for arm in aliases:
            runs = grouped[arm]
            values.append(f"{statistics.mean(100*r['metrics']['final']['english']['groups'][key]['wer'] for r in runs):.3f}" if runs else 'Unavailable')
        text += f"| {language} | {' | '.join(values)} |\n"
    full = selection['arms']['full_gold']['n']
    keep = selection['arms']['filtered10_gold']['n']
    overlaps = audit['selected_train_overlap_counts'].get('filtered10_gold:random_matched_gold', 0)
    rejected_in_control = selection['arms']['random_matched_gold']['teacher_rejected_included_n']
    text += (f"\nThe teacher filter retained {keep:,}/{full:,} recordings ({100*keep/full:.2f}%). "
             f"The matched random pool contains {rejected_in_control:,} teacher-rejected recordings and overlaps "
             f"the filtered pool on {overlaps:,} recordings. Both pools preserve the same per-speaker and reference-length-bin counts. "
             "The included selection report records the achieved duration balance in every stratum.\n")
    text += '\n## What the result establishes\n\n'
    if complete and decision['pilot_positive']:
        text += 'Under this fixed corpus, student and short training schedule, the registered evidence supports a selection penalty relative to the matched random control. Because all targets are the supplied references, this is evidence about which inputs are selected. It does not establish that noisy teacher labels should be retained, or that a proposed distillation loss improves practical ASR. Independent replication is warranted.\n'
    elif complete:
        text += 'The experiment did not meet the prespecified criterion for a harmful selection effect. The original coverage concern is not sufficient evidence for this mechanism. We keep the full results and stop this claim at the pilot gate. A favorable subgroup or normalization cannot replace the registered primary contrast.\n'
    else:
        text += 'Incomplete required evidence cannot establish the registered effect. The remaining runs and any infrastructure failures are identified in the evidence package. No missing result is treated as a successful run or a negative scientific observation.\n'
    text += '\n## Research-loop changes informed by AutoResearch\n\n'
    text += ('We inspected [EvoMap/AutoResearch at a pinned revision](https://github.com/EvoMap/AutoResearch/tree/6b85443f9e3d33d357c86c8bb826f9ea03c780b3). '
             'Its persistent work queue, execution receipts, pilot gates and separate criticism are useful engineering mechanisms. '
             'We implemented a smaller controller around the existing ASR experiment: frozen inputs and comparisons; an append-only event ledger; '
             'development-query reservations before dispatch; actual evaluation receipts afterward; independent recomputation from predictions; '
             'and a numerical replication gate. The two early smoke evaluations are explicitly retrospective.\n\n'
             'The upstream planner permits metric or dataset revision after analysis, and its review loop can iterate against reviewer ratings. '
             'Our controller keeps such changes from silently becoming the same experiment. Review scores never override missing evidence or the frozen numerical criterion. '
             'These are integrity checks on this workflow, not a security boundary against a person who can rewrite all local files. '
             'The available reviewers share an underlying model; this session does not claim independent cross-model review.\n')
    sections = (P / 'report_sections.md').read_text()
    sections = sections.split('\n', 1)[1]
    # Retain the literature-grounded question as well as exact methods and limitations.
    sections = sections[sections.index('## Motivation and question'):]
    text += '\n' + sections
    text += '\n## Durable evidence and reproduction\n\n'
    text += ('The companion evidence archive contains the frozen schedule, data manifests and provenance, teacher outputs, all completed student predictions, '
             'per-update exposure logs, independent audits, selection balance, decision ledger, source code and tests. The separate trained-checkpoint archive '
             'preserves four completed models and their processor files, with each model hash checked against its audited execution summary. Its index explicitly records the partial coverage. Other completed or interrupted artifacts and optimizer states remain in the '
             '`accent-selection-pilot-results` Modal volume under `training/<arm>-seed<seed>/checkpoint_last/`; the audio and feature caches are in '
             '`accent-selection-pilot-data`. No checkpoint download is needed to recompute the reported WER.\n\n'
             'Start with `START_HERE.md` in the archive. Run the included offline audit command to verify exact coverage, references, '
             'error counts, configurations and exposure sequences. Original absolute paths in the ledger identify the execution workspace; '
             'the independent auditor can operate on the extracted relative project layout.\n\n')
    verification = P / 'verification_tests.log'
    if verification.exists():
        summary = verification.read_text().strip().splitlines()[-1]
        text += (f'Local integrity and control tests: {summary}. The skipped test requires Torch locally; '
                 'the real tiny-architecture forward/backward test passed in the GPU smoke runtime. '
                 'A separate read-only prefetch process warmed the exact registered input files to reduce remote filesystem latency; '
                 'it performed no model evaluation, changed no training setting, and did not advance the trainer RNG. Its receipts are preserved.\n\n')
    text += ('The stored `generation_reached_token_limit` diagnostic is invalid for this Transformers version: Whisper removes EOS before returning text tokens, then pads shorter batch rows with the EOS/pad token. '
             'The flag therefore identifies the longest outputs in a batch, not truncation. It is never used in WER or the numerical decision. '
             'The observed outputs are far shorter than the configured 224-token limit, but exact stopping reasons cannot be certified from the postprocessed token arrays. '
             'The source-level explanation and five-run diagnostic audit are in `DIAGNOSTIC_FIELD_NOTE.md`. Frozen evidence remains unchanged.\n\n')
    budget = read(P / 'budget.json')
    reserve = budget['prior_work_carryover_allowance_usd'] + sum(j['reserved_usd'] for j in budget['new_jobs'])
    estimate = sum(j.get('estimated_resource_cost_usd', 0) for j in budget['new_jobs'])
    text += (f"Conservative reservations plus the prior-work allowance total ${reserve:.2f}, within the original $30 ceiling. "
             f"Completed new-job resource estimates sum to ${estimate:.2f}; this is not an invoice and excludes unverified failed-call/build/storage charges. "
             'Unused reservations are not spending. The full ledger preserves failed attempts.\n\n')
    text += ('A one-hour timer was set at 07:46:43 UTC on 9 September 2026. The user subsequently requested the stop; the controller, cache helper, checkpoint poller and paid computation were stopped. '
             'The final seed was never launched. The stopped partial run is not counted as successful evidence. '
             f"This handoff report was generated at {datetime.now(timezone.utc).strftime('%H:%M:%S UTC')}.\n")
    return text


def inline(text):
    text = html.escape(text)
    text = re.sub(r'\[([^\]]+)\]\((https?://[^)]+)\)', r'<link href="\2" color="#234d6c">\1</link>', text)
    text = re.sub(r'\*\*([^*]+)\*\*', r'<b>\1</b>', text)
    text = re.sub(r'`([^`]+)`', r'<font name="Courier" size="8">\1</font>', text)
    return text


def make_figure(audit):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    arms = ('full_gold', 'filtered10_gold', 'random_matched_gold')
    labels = ('Full pool', 'Filtered', 'Matched random')
    colors = ('#3e6683', '#b35443', '#41836d')
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.25), constrained_layout=True)
    for x, arm, label, color in zip(range(3), arms, labels, colors):
        values = [100*r['metrics']['final']['english']['macro_language_wer'] for r in audit['runs'] if r['arm_name'] == arm]
        axes[0].bar(x, statistics.mean(values), color=color, alpha=.72, width=.63)
        axes[0].scatter([x-.09, x, x+.09], values, color='#172832', s=24, zorder=3)
    baseline = 100*audit['runs'][0]['metrics']['baseline']['english']['macro_language_wer']
    axes[0].axhline(baseline, linestyle='--', color='#5f6265', linewidth=1.2, label='Initial model')
    axes[0].set(xticks=range(3), xticklabels=labels, ylabel='English L1-macro WER (%)', title='Fixed-update adaptation')
    axes[0].legend(frameon=False, fontsize=8)
    for norm, color, offset in [('english','#335d83',-.07), ('minimal','#ad663f',.07)]:
        values = [c['filtered_minus_random_macro_wer_pp'] for c in sorted(audit['paired_contrasts'], key=lambda r:r['seed']) if c['normalization'] == norm]
        axes[1].plot([i+offset for i in range(4)], values+[statistics.mean(values)], 'o-', color=color,
                     linewidth=1, markersize=5, label=norm.capitalize())
    axes[1].axhline(0, color='#555b60', linewidth=1)
    axes[1].axhline(.5, color='#999999', linestyle=':', linewidth=1)
    axes[1].set(xticks=range(4), xticklabels=('Seed 0','Seed 1','Seed 2','Mean'),
                ylabel='Filtered minus random WER (pp)', title='Positive favors matched random')
    axes[1].legend(frameon=False, fontsize=8)
    for ax in axes:
        ax.spines[['top','right']].set_visible(False)
        ax.grid(axis='y', color='#dde2e6', linewidth=.5, zorder=0)
        ax.set_axisbelow(True); ax.tick_params(labelsize=8)
        ax.title.set_fontsize(10); ax.yaxis.label.set_fontsize(9)
    folder = OUT / 'figures'; folder.mkdir(parents=True, exist_ok=True)
    fig.savefig(folder / 'pilot_contrast.png', dpi=240)
    fig.savefig(folder / 'pilot_contrast.svg')
    plt.close(fig)


def render_pdf(markdown, destination, vector_figure=None):
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_LEFT
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.lib.pagesizes import A4
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, KeepTogether, Flowable
    class VectorFigure(Flowable):
        def __init__(self, path, width, height):
            super().__init__()
            from pdfrw import PdfReader
            from pdfrw.buildxobj import pagexobj
            self.form = pagexobj(PdfReader(str(path)).pages[0])
            box = [float(v) for v in self.form.BBox]
            self.width = min(width, box[2]-box[0])
            self.height = self.width*(box[3]-box[1])/(box[2]-box[0])
            self.hAlign = 'CENTER'
        def draw(self):
            from pdfrw.toreportlab import makerl
            box = [float(v) for v in self.form.BBox]
            self.canv.saveState()
            self.canv.scale(self.width/(box[2]-box[0]), self.height/(box[3]-box[1]))
            self.canv.translate(-box[0], -box[1])
            self.canv.doForm(makerl(self.canv, self.form))
            self.canv.restoreState()
    accent = colors.HexColor('#203c50')
    styles = {
        'body': ParagraphStyle('body', fontName='Times-Roman', fontSize=10.2, leading=13, spaceAfter=6),
        'h1': ParagraphStyle('h1', fontName='Helvetica-Bold', fontSize=22, leading=25, spaceAfter=13, textColor=accent),
        'h2': ParagraphStyle('h2', fontName='Helvetica-Bold', fontSize=12.7, leading=16, spaceBefore=13, spaceAfter=7, keepWithNext=True, textColor=accent),
        'h3': ParagraphStyle('h3', fontName='Helvetica-Bold', fontSize=10.5, leading=13, spaceBefore=8, spaceAfter=5, keepWithNext=True),
        'cell': ParagraphStyle('cell', fontName='Helvetica', fontSize=8.0, leading=10.2),
    }
    story, paragraph = [], []
    lines = markdown.splitlines()
    def flush():
        if paragraph:
            story.append(Paragraph(inline(' '.join(paragraph)), styles['body']))
            paragraph.clear()
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if not line:
            flush(); i += 1; continue
        if line.startswith('#'):
            flush(); level = min(len(line)-len(line.lstrip('#')), 3)
            story.append(Paragraph(inline(line.lstrip('#').strip()), styles[f'h{level}']))
        elif line.startswith('!['):
            flush(); match = re.fullmatch(r'!\[[^]]*\]\(([^)]+)\)', line)
            path = OUT / match.group(1)
            figure = (VectorFigure(vector_figure, A4[0]-92, (A4[0]-92)*3.25/9) if vector_figure else
                      Image(str(path), width=A4[0]-92, height=(A4[0]-92)*3.25/9))
            figure.keepWithNext = True
            story.append(figure)
        elif line.startswith('|'):
            flush(); rows = []
            while i < len(lines) and lines[i].strip().startswith('|'):
                cells = [s.strip() for s in lines[i].strip().strip('|').split('|')]
                if not all(re.fullmatch(r':?-+:?', c) for c in cells):
                    rows.append([Paragraph(inline(c), styles['cell']) for c in cells])
                i += 1
            n = len(rows[0]); width = A4[0]-92
            widths = [width*.29]+[width*.71/(n-1)]*(n-1) if n > 1 else [width]
            if n == 3:
                widths = [width*.12, width*.44, width*.44]
            table = Table(rows, colWidths=widths, repeatRows=1, hAlign='LEFT')
            table.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#edf2f5')),
                ('LINEBELOW',(0,0),(-1,0),.65,accent), ('LINEBELOW',(0,-1),(-1,-1),.4,colors.HexColor('#b5c2c9')),
                ('VALIGN',(0,0),(-1,-1),'TOP'), ('LEFTPADDING',(0,0),(-1,-1),6),
                ('RIGHTPADDING',(0,0),(-1,-1),6), ('TOPPADDING',(0,0),(-1,-1),6),
                ('BOTTOMPADDING',(0,0),(-1,-1),6)]))
            story.append(KeepTogether([table, Spacer(1, 9)])); continue
        elif line.startswith('- '):
            flush(); item = [line[2:]]
            while i+1 < len(lines) and lines[i+1].startswith('  ') and not lines[i+1].lstrip().startswith('- '):
                i += 1; item.append(lines[i].strip())
            story.append(Paragraph('&#8226; '+inline(' '.join(item)), styles['body']))
        else:
            paragraph.append(line)
        i += 1
    flush()
    def page(canvas, doc):
        canvas.saveState(); canvas.setFont('Helvetica', 8); canvas.setFillColor(colors.HexColor('#637480'))
        canvas.drawString(46, 25, 'whispr-nano | controlled ASR research evidence | 9 September 2026')
        canvas.drawRightString(A4[0]-46, 25, str(doc.page)); canvas.restoreState()
    doc = SimpleDocTemplate(str(destination), pagesize=A4, rightMargin=46, leftMargin=46,
                            topMargin=43, bottomMargin=43, title='Teacher filtering and accented ASR: controlled pilot',
                            author='whispr-nano research project')
    doc.build(story, onFirstPage=page, onLaterPages=page)
    # ReportLab misformats half-hour local offsets here; record unambiguous UTC.
    from pypdf import PdfReader, PdfWriter
    reader = PdfReader(destination)
    writer = PdfWriter(clone_from=reader)
    stamp = datetime.now(timezone.utc).strftime('D:%Y%m%d%H%M%SZ')
    writer.add_metadata({'/CreationDate': stamp, '/ModDate': stamp})
    temporary = Path(destination).with_suffix('.utc.pdf')
    with temporary.open('wb') as stream:
        writer.write(stream)
    temporary.replace(destination)


def main():
    audit = read(P / 'results/audit.json')
    decision = read(P / 'results/decision.json')
    selection = read(P / 'selections/selection_report.json')
    if audit['status'] == 'valid_complete':
        make_figure(audit)
    markdown = make_markdown(audit, decision, selection)
    markdown = markdown.translate(str.maketrans({'–':'-', '—':'-', '−':'-', '’':"'", '“':'"', '”':'"', '\u2011':'-'}))
    OUT.mkdir(exist_ok=True); (OUT / 'pdf').mkdir(exist_ok=True)
    (OUT / 'research_report.md').write_text(markdown)
    render_pdf(markdown, OUT / 'pdf/accent_selection_research_report.pdf')
    print('Wrote final Markdown and PDF from audited evidence.')


if __name__ == '__main__':
    main()
