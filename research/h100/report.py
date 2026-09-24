"""Generate the complete H100 evidence report and a vector paired-effect figure."""
from datetime import datetime,timezone
import json
from pathlib import Path
import re
import shutil
import statistics

ROOT=Path(__file__).resolve().parents[2]
H=ROOT/'research/h100'
P=ROOT/'research/pilot'
OUT=ROOT/'output/h100'


def figure(audit):
    from research.h100.orx_figstyle import use_style,figure_grid,PALETTE,panel_labels,save
    from matplotlib.lines import Line2D
    import matplotlib.pyplot as plt
    use_style()
    width=6.75
    fig,axes=figure_grid(width=width,ratio=3.25/9)
    colors=[PALETTE['blue'],PALETTE['orange'],PALETTE['green']]
    markers=['o','^','s']; offsets=[-.13,0,.13]
    arms=['full_gold','filtered10_gold','random_matched_gold']
    for y,arm in enumerate(arms):
        rows=sorted([r for r in audit['runs'] if r['arm_name']==arm],key=lambda r:r['seed'])
        values=[100*r['metrics']['final']['english']['macro_language_wer'] for r in rows]
        axes[0].plot([min(values),max(values)],[y,y],color='#333333',linewidth=.8)
        for i,v in enumerate(values):axes[0].scatter(v,y+offsets[i],s=17,color=colors[i],marker=markers[i],zorder=4)
        axes[0].scatter(statistics.mean(values),y,s=18,color='black',marker='D',zorder=5)
    baseline=100*audit['runs'][0]['metrics']['baseline']['english']['macro_language_wer']
    axes[0].axvline(baseline,color='#777777',linestyle='--',linewidth=.8)
    axes[0].set(yticks=range(3),yticklabels=['Full pool','Filtered','Matched random'],xlabel='English L1-macro WER (%)',ylabel='Training input pool')
    axes[0].invert_yaxis();axes[0].margins(x=.13,y=.28)
    for y,norm in enumerate(['english','minimal']):
        values=[r['filtered_minus_random_macro_wer_pp'] for r in sorted(audit['paired_contrasts'],key=lambda r:r['seed']) if r['normalization']==norm]
        axes[1].plot([min(values),max(values)],[y,y],color='#333333',linewidth=.8)
        for i,v in enumerate(values):axes[1].scatter(v,y+offsets[i],s=17,color=colors[i],marker=markers[i],zorder=4)
        axes[1].scatter(statistics.mean(values),y,s=18,color='black',marker='D',zorder=5)
    axes[1].axvline(0,color='#666666',linewidth=.8)
    axes[1].axvline(.5,color='#777777',linestyle=':',linewidth=.8)
    axes[1].set(yticks=[0,1],yticklabels=['English','Minimal'],xlabel='Filtered minus random WER (pp)',ylabel='Normalization')
    axes[1].invert_yaxis();axes[1].margins(x=.15,y=.45)
    for ax in axes:ax.grid(axis='x');ax.grid(axis='y',visible=False)
    panel_labels(axes)
    handles=[Line2D([],[],marker=markers[i],color=colors[i],linestyle='None',label=f'Seed {i}') for i in range(3)]
    handles += [Line2D([],[],marker='D',color='black',linestyle='-',label='Mean / seed range'),
                Line2D([],[],color='#777777',linestyle='--',label='Initial model (a)')]
    fig.legend(handles=handles,loc='outside lower center',ncol=5,fontsize=6.5)
    folder=OUT/'figures';folder.mkdir(parents=True,exist_ok=True)
    save(fig,str(folder/'selection_contrast'),close=False)
    fig.savefig(folder/'selection_contrast.png',dpi=240);plt.close(fig)


def main():
    from research.pilot import build_report as base
    audit=json.loads((H/'results/audit.json').read_text());decision=json.loads((H/'results/decision.json').read_text())
    if audit['status']!='valid_complete' or len(audit['runs'])!=9:raise ValueError('Full H100 evidence is required')
    OUT.mkdir(parents=True,exist_ok=True)
    sections=(P/'report_sections.md').read_text()
    sections=sections.replace('All nine registered runs start from the same Whisper-tiny weights, revision',
        'All nine H100 runs start from the same Whisper-tiny weights, revision')
    old=('The schedule was frozen after an infrastructure smoke test using 16 training\n'
         'and 12 development clips, and before any full-development training outcome.\n'
         'Those two smoke evaluations are disclosed development accesses.')
    new=('The scientific settings were originally frozen after a smoke test using 16 training and 12 development clips, before full-development outcomes. '
         'The H100 execution was separately registered after five completed T4 runs had been inspected. All nine H100 runs repeat the same settings; '
         'the prior outcomes are retained and disclosed. The new ledger reserves 18 evaluations in addition to 14 conservatively charged prior slots, '
         'including two smoke evaluations and an interrupted run. Reuse of these development speakers is exploratory and is not independent confirmation.\n\n'
         'Three H100 workers each train one arm across the three seeds. Each worker first copies and checksum-verifies the exact union of its scheduled audio and feature inputs onto local disk. '
         'The untouched training core uses full FP32, deterministic operations, two Torch CPU threads, and disables TF32. Batch size, loss, optimizer, decoder and update counts are unchanged. '
         'No additional model call is used for staging, and all nine runs use the same hardware class.')
    if old not in sections:raise ValueError('Expected original registration paragraph not found')
    sections=sections.replace(old,new)
    (H/'report_sections.md').write_text(sections)
    for filename in ('budget.json','verification_tests.log'):shutil.copy2(P/filename,H/filename)
    base.P=H;base.OUT=OUT
    markdown=base.make_markdown(audit,decision,json.loads((P/'selections/selection_report.json').read_text()))
    markdown=markdown.replace('Controlled ASR pilot and research-loop assessment | 9 September 2026',
                              'Complete H100 experiment: three arms, three seeds | 9 September 2026')
    markdown=markdown.replace('Unequal completed seed counts make these means descriptive, not a valid matched comparison. ','')
    markdown=markdown.replace('figures/pilot_contrast.png','figures/selection_contrast.png')
    caption=('Figure 1. All nine H100 runs are shown. Panel (a) reports final English macro WER; panel (b) shows the paired filtering effect under both normalizers. '
             'Colored markers identify the three training seeds; black diamonds and lines show their mean and range, not population confidence intervals. '
             'Positive differences favor random selection. The dotted 0.5-point line is the English mean pilot criterion; two positive seeds and a nonnegative minimal mean are also required.')
    markdown=re.sub(r'Figure 1\..*?\n\n',caption+'\n\n',markdown,flags=re.S)
    markdown=markdown.replace('preserves four completed models and their processor files, with each model hash checked against its audited execution summary. Its index explicitly records the partial coverage. Other completed or interrupted artifacts and optimizer states remain in the ',
                              'preserves all nine final H100 models and their processor files, with each weight hash checked against its audited execution summary. Optimizer states remain in the ')
    markdown=markdown.replace('`training/<arm>-seed<seed>/checkpoint_last/`','`h100-v1/<arm>-seed<seed>/checkpoint_last/`')
    markdown=markdown.replace('The source-level explanation and five-run diagnostic audit are in `DIAGNOSTIC_FIELD_NOTE.md`.',
                              'The source-level explanation and prior five-run diagnostic audit are in `research/pilot/DIAGNOSTIC_FIELD_NOTE.md`; the same frozen diagnostic remains unused in this H100 study.')
    markdown=markdown.replace('A separate read-only prefetch process warmed the exact registered input files to reduce remote filesystem latency; it performed no model evaluation, changed no training setting, and did not advance the trainer RNG. Its receipts are preserved.',
                              'All H100 staging receipts verify audio and feature checksums before training. The stage performs no model evaluation and leaves the trainer RNG unchanged.')
    markdown=markdown[:markdown.index('A one-hour timer was set')]
    receipts=[json.loads(path.read_text()) for path in H.glob('*.receipt.json')]
    seconds=max(r['wall_seconds'] for r in receipts)
    cost=sum(r['estimated_resource_cost_usd'] for r in receipts)
    markdown+=(f'The three H100 workers completed all nine runs in {seconds/60:.2f} minutes of maximum worker wall time, including staging. '
               f'The H100 GPU/CPU/memory resource estimate is ${cost:.3f}, using [Modal\'s published rates](https://modal.com/pricing); it is not an invoice. '
               'Each worker had a 1200-second timeout and 180-second startup cap, with total additional reservations capped at $5.40.\n\n'
               'The 30-minute extension began at 08:23:53 UTC on 9 September 2026. '
               f'This complete report was generated at {datetime.now(timezone.utc).strftime("%H:%M:%S UTC")}.\n')
    markdown=markdown.translate(str.maketrans({'–':'-','—':'-','−':'-','’':"'",'“':'"','”':'"','\u2011':'-'}))
    (OUT/'research_report.md').write_text(markdown)
    figure(audit)
    destination=ROOT/'output/pdf/h100_selection_research_report.pdf'
    base.render_pdf(markdown,destination,vector_figure=OUT/'figures/selection_contrast.pdf')
    print('Complete H100 report written:',destination)


if __name__=='__main__':main()
