"""Render recorded decoder error components. Run: python plot.py"""
import json
from pathlib import Path
from orx_figstyle import TEXT, PALETTE, figure, save, use_style
HERE=Path(__file__).resolve().parent
rows=json.loads((HERE/'comparison.json').read_text())
use_style()
fig,ax=figure(width=TEXT,ratio=.52)
left=[0.0]*len(rows)
for key,label,color in [('substitution_percent','Substitutions',PALETTE['blue']),('deletion_percent','Deletions',PALETTE['orange']),('insertion_percent','Insertions',PALETTE['green'])]:
 values=[r[key] for r in rows]
 ax.barh(range(len(rows)),values,left=left,label=label,color=color,edgecolor='black',linewidth=.35)
 left=[a+b for a,b in zip(left,values)]
for i,r in enumerate(rows):
 ax.text(r['wer_percent']+1,i,f"{r['wer_percent']:.2f}",va='center',fontsize=7)
ax.set_yticks(range(len(rows)),[r['recipe'] for r in rows]);ax.invert_yaxis()
ax.set_xlabel('Word errors per 100 reference words');ax.set_ylabel('Turbo inference recipe')
ax.set_xlim(0,max(left)*1.18);ax.grid(axis='x',alpha=.25);ax.grid(axis='y',visible=False)
ax.legend(loc='upper center',bbox_to_anchor=(.5,1.24),ncol=3,fontsize=7)
save(fig,str(HERE/'decoder_error_components'))
fig.savefig(HERE/'decoder_error_components.png',dpi=180)
(HERE/'decoder_error_components.caption.md').write_text('Native decoding reduces deletions but increases insertions on this Hindi development sample. Same3 conversations and6,503 reference words; seed12. No confidence intervals at this sample size. Native timestamp decoding uses batch1; fixed-window recipes use batch8. All capped output is retained.\n')
