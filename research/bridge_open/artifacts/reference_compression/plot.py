"""Reference-token compression distributions; medians/IQR, not confidence intervals."""
import csv,statistics
from pathlib import Path
from orx_figstyle import TEXT,PALETTE,figure,save,use_style
HERE=Path(__file__).resolve().parent
names=dict(as_='Assamese',bn='Bengali',gu='Gujarati',hi='Hindi',kn='Kannada',ml='Malayalam',mr='Marathi',ta='Tamil',te='Telugu');names['as']=names.pop('as_')
groups={}
for row in csv.DictReader((HERE/'windows.csv').open()):
 if int(row['window_tokens'])==200:groups.setdefault(row['language'],[]).append(float(row['compression_ratio']))
ordered=sorted(groups,key=lambda k:statistics.median(groups[k]))
use_style();fig,ax=figure(width=TEXT,ratio=.65)
ax.axvline(1.35,color='#555555',linestyle='--',linewidth=.8,label='Documented cutoff: 1.35')
for i,lang in enumerate(ordered):
 values=groups[lang];q=statistics.quantiles(values,n=4,method='inclusive');m=statistics.median(values)
 ax.plot([q[0],q[2]],[i,i],color=PALETTE['blue'],linewidth=2)
 ax.plot(m,i,'o',color=PALETTE['blue'],markersize=4)
ax.set_yticks(range(len(ordered)),[f'{names[k]} (n={len(groups[k])})' for k in ordered]);ax.invert_yaxis()
ax.set_xlabel('Token-byte compression ratio (dimensionless)');ax.set_ylabel('Released reference text, 200-token windows')
ax.set_xlim(1.2,3.2);ax.grid(axis='x',alpha=.25);ax.grid(axis='y',visible=False)
ax.legend(loc='upper right',fontsize=7,frameon=True)
save(fig,str(HERE/'reference_token_compression'));fig.savefig(HERE/'reference_token_compression.png',dpi=180)
(HERE/'reference_token_compression.caption.md').write_text('The token-byte compression cutoff also triggers on released reference text. Points and bars show medians and interquartile ranges of nonoverlapping 200-token windows, with window counts shown. These dependent text windows are not audio-aligned model outputs; bars are descriptive, not confidence intervals or detector false-positive rates.\n')
