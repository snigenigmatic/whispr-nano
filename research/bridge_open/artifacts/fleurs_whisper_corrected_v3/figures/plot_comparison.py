"""Render corrected observed values only; run from any working directory."""
import json
from pathlib import Path
import numpy as np
from orx_figstyle import use_style,figure,save,TEXT,PALETTE,MUTED
HERE=Path(__file__).resolve().parent
report=json.loads((HERE.parent/'comparison.json').read_text())
use_style();fig,ax=figure(width=TEXT,ratio=.48)
langs=[('hi','Hindi',418),('bn','Bengali',920),('as','Assamese',984)]
for i,(model,color,hatch) in enumerate([('Turbo',MUTED,'///'),('Large-v3',PALETTE['blue'],'')]):
 vals=[]
 for lang,_,n in langs:
  a=report['models'][model]['languages'][lang]['pooled_duration_routed'];assert a['n']==n
  vals.append(a['micro_wer']*100)
 bars=ax.bar(np.arange(3)+(i-.5)*.34,vals,width=.34,label=model,color=color,hatch=hatch,edgecolor='black',linewidth=.4)
 ax.bar_label(bars,fmt='%.2f',fontsize=6,padding=2)
ax.set_xticks(range(3),[f'{label}\n(n={n})' for _,label,n in langs])
ax.set_ylim(0,150);ax.set_yticks(np.arange(0,151,25));ax.set_ylabel('Micro WER (%)');ax.set_xlabel('FLEURS official-test language')
ax.legend(loc='upper left',frameon=True,facecolor='white',edgecolor='black')
save(fig,str(HERE/'corrected_whisper'),close=False)
fig.savefig(HERE/'corrected_whisper_preview.png',dpi=180,bbox_inches='tight')
(HERE/'CAPTION.md').write_text('Large-v3 has lower observed WER than Turbo under the declared duration-based routing policy. All 2,322 recordings per model are included. Clips up to 30 seconds use the audited HF beam-5 recipe; all 14 longer clips use native transcription. Seed 12; these bars show observed scores without seed intervals. Sentence-cluster paired intervals are provided separately in comparison.json. This is a post-result protocol correction, not a held-out improvement.\n')
