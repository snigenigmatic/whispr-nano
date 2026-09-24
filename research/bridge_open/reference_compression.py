"""CPU reference-text stress test of Transformers' token-byte repetition statistic."""
import csv,hashlib,json,math,statistics,zlib
from pathlib import Path
from huggingface_hub import snapshot_download
from transformers import WhisperTokenizer
ROOT=Path(__file__).resolve().parents[2]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 cfg=json.loads((ROOT/'research/bridge_open/run_config.json').read_text())
 manifest=Path(cfg['manifest_path']);assert sha(manifest)==cfg['manifest_sha256']
 snap=Path(snapshot_download(cfg['model_id'],revision=cfg['model_revision'],allow_patterns=['*.json','*.txt'],local_files_only=True))
 tok=WhisperTokenizer.from_pretrained(snap,local_files_only=True)
 width=int(math.log2(len(tok))/8)+1
 rows=[]
 for item in map(json.loads,manifest.read_text().splitlines()):
  if item['reference_qc']!='pass' or item['language'] not in cfg['languages']:continue
  ref=Path(item['reference_path']);assert sha(ref)==item['reference_sha256']
  text=json.loads(ref.read_text())['reference_text']
  tokens=tok.encode(text,add_special_tokens=False)
  for size in cfg['window_tokens']:
   for start in range(0,len(tokens)-size+1,size):
    selected=tokens[start:start+size];encoded=b''.join(t.to_bytes(width,'little') for t in selected)
    ratio=len(encoded)/len(zlib.compress(encoded))
    rows.append(dict(conversation_id=item['conversation_id'],language=item['language'],reference_sha256=item['reference_sha256'],window_tokens=size,start_token=start,compression_ratio=ratio,over_1_35=ratio>1.35,over_2_4=ratio>2.4,token_bytes_sha256=hashlib.sha256(encoded).hexdigest()))
 summary=[]
 for lang in sorted({r['language'] for r in rows}):
  for size in cfg['window_tokens']:
   group=[r for r in rows if r['language']==lang and r['window_tokens']==size]
   summary.append(dict(language=lang,window_tokens=size,windows=len(group),median_ratio=statistics.median(r['compression_ratio'] for r in group),fraction_over_1_35=sum(r['over_1_35'] for r in group)/len(group),fraction_over_2_4=sum(r['over_2_4'] for r in group)/len(group)))
 out=ROOT/'research/bridge_open/artifacts/reference_compression';out.mkdir(parents=True,exist_ok=False)
 for name,data in [('windows',rows),('summary',summary)]:
  with (out/f'{name}.csv').open('w',newline='') as f:
   w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
 evidence=dict(config=cfg,tokenizer_files={p.name:sha(p) for p in snap.iterdir() if p.is_file() and p.suffix in ['.json','.txt']},token_byte_width=width,zlib_version=zlib.ZLIB_RUNTIME_VERSION,scope='Reference-text sensitivity diagnostic; arbitrary fixed tokenizer windows, not audio-aligned hypotheses or measured detector false-positive rate',summary=summary)
 (out/'evidence.json').write_text(json.dumps(evidence,indent=2)+'\n')
 print(json.dumps(dict(event='final_reference_compression_diagnostic',**evidence)),flush=True)
 return 0
