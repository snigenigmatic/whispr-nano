"""Plot observed full-cohort Hindi WER, with no invented seed uncertainty."""
import json
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
ART=ROOT/'research/bridge_open/artifacts'
sys.path.insert(0,str(ART/'fleurs_whisper_corrected_v3/figures'))
from orx_figstyle import use_style,figure,save,COLUMN,PALETTE,MUTED


def main():
    report=json.loads((ART/'fleurs_qwen_comparison_v1/comparison.json').read_text())
    names=['Whisper Turbo','Whisper Large-v3','Qwen3-ASR-1.7B']
    vals=[]
    for name in names:
        r=report['summary'][name];assert r['utterances']==418
        vals.append(r['micro_wer']*100)
    out=ART/'fleurs_qwen_comparison_v1/figures';out.mkdir(exist_ok=False)
    use_style();fig,ax=figure(width=COLUMN,ratio=.85)
    bars=ax.bar(range(3),vals,color=[MUTED,PALETTE['blue'],PALETTE['green']],edgecolor='black',linewidth=.4)
    ax.bar_label(bars,fmt='%.2f',padding=3,fontsize=7)
    ax.set_xticks(range(3),['Whisper\nTurbo','Whisper\nLarge-v3','Qwen3-ASR\n1.7B'])
    ax.set_ylabel('Micro WER (%)');ax.set_ylim(0,max(vals)*1.22)
    ax.set_xlabel('FLEURS Hindi official test (n = 418)')
    save(fig,str(out/'hindi_comparison'),close=False)
    fig.savefig(out/'hindi_comparison_preview.png',dpi=180,bbox_inches='tight')
    (out/'CAPTION.md').write_text('Observed WER for three declared systems on the same 418 FLEURS Hindi official-test recordings. Qwen uses native API postprocessing, bfloat16 and greedy decoding; Whisper uses float32 beam-5 for short clips and native transcription for the one long clip. A single run per declared condition is shown. Paired sentence-cluster intervals are reported separately; these bars are not training-seed or speaker-population intervals. No method novelty or general fairness claim is established.\n')

if __name__=='__main__':main()
