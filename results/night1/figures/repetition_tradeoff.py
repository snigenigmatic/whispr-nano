"""Regenerate from independently validated cloud artifacts, not hand-entered data."""
import json
from pathlib import Path
import matplotlib.pyplot as plt
from orx_figstyle import WIDE, PALETTE, use_style, panel_labels, save

HERE=Path(__file__).resolve().parent


def main():
    data=json.loads((HERE.parent/'cloud_decoder_validated_20261013.json').read_text())
    cohort='commands_repeat_5'
    values=[data['runs']['hf_turbo_beam5']['cohorts'][cohort],
            data['runs']['alignment_zero_turbo']['cohorts'][cohort],
            data['published_duration_character_cap']['hf_turbo_beam5'][cohort]]
    names=['Text collapse','Zero-span gate','Published cap']
    use_style()
    fig,axes=plt.subplots(1,2,figsize=(WIDE,2.65),sharey=True,gridspec_kw={'width_ratios':[1.45,1]},layout='constrained')
    left,right=axes
    for y,v in enumerate(values):
        lo,hi=v['paired_speaker_bootstrap_95_ci_points'];delta=v['difference_points']
        left.plot([lo,hi],[y,y],color=PALETTE['blue'],lw=1.4)
        left.plot(delta,y,'o',color=PALETTE['blue'],ms=4)
        left.annotate(f'{delta:+.1f}',(delta,y),xytext=(0,9),textcoords='offset points',ha='center',fontsize=7)
    left.axvline(0,color='#777777',lw=.8,ls='--')
    left.set_yticks(range(3),names);left.set_xlabel('WER change vs. raw output (points)')
    left.set_ylabel('Postprocessing rule');left.set_ylim(2.5,-.7);left.set_xlim(-355,60)
    bars=right.barh(range(3),[v['harmed'] for v in values],height=.42,color=PALETTE['red'],edgecolor='black',lw=.4)
    right.bar_label(bars,padding=3,fontsize=8)
    right.tick_params(axis='y',labelleft=False);right.set_ylim(2.5,-.7);right.set_xlim(0,100)
    right.set_xlabel('Utterances harmed (out of 100)')
    panel_labels(axes)
    save(fig,str(HERE/'repetition_tradeoff'),close=False)
    fig.savefig(HERE/'repetition_tradeoff.png',dpi=180)
    plt.close(fig)
    caption=("Pooled WER improvement can hide damage to genuine repetition. Whisper Turbo HF beam5 on 100 five-repeat human-keyword controls: "
             "(a) change from raw WER, with paired speaker-bootstrap 95% intervals (4,000 draws, 68 source speakers); "
             "(b) utterances whose word-edit count increases. One decoded run per condition. Repeats are constructed, not natural conversation. "
             "The published cap is the one-character-per-20-ms duration baseline. Negative WER change is better; no novelty is claimed for that cap.")
    (HERE/'repetition_tradeoff.caption.txt').write_text(caption+'\n')


if __name__=='__main__':main()
