"""Build an offline transcript viewer from accepted benchmark artifacts only."""
import argparse
import hashlib
import json
import os
from pathlib import Path
from research.bridge_open.audit_scripts import distribution

ROOT=Path(__file__).resolve().parents[2]
ART=ROOT/'research/bridge_open/artifacts'
OUT=ART/'benchmark_review'


def jsonl(p):
    return [json.loads(s) for s in Path(p).read_text().splitlines() if s.strip()]


def build(include_omni=False, output=None):
    OUT = Path(output) if output else ART/'benchmark_review'
    OUT.mkdir(exist_ok=False)
    manifest=jsonl(ROOT/'data/bridge_open/fleurs_test_v1/corpus-v2.jsonl')
    models={
        'Whisper Turbo':jsonl(ART/'fleurs_whisper_corrected_v3/turbo_predictions.jsonl'),
        'Whisper Large-v3':jsonl(ART/'fleurs_whisper_corrected_v3/large-v3_predictions.jsonl'),
        'Qwen3-ASR-1.7B':jsonl(ART/'qwen_native_full_hi_v1/predictions.jsonl'),
    }
    accepted=json.loads((ART/'fleurs_qwen_comparison_v1/comparison.json').read_text())
    assert accepted['qwen_audit']['records']==418
    if include_omni:
        comparison=json.loads((ART/'fleurs_omni_comparison_v1/comparison.json').read_text())
        assert sum(g['utterances'] for g in comparison['groups'].values())==2322
        models['OmniASR CTC-1B']=jsonl(ART/'fleurs_omni_comparison_v1/predictions.jsonl')
        assert len(models['OmniASR CTC-1B'])==2322
    bymodel={name:{r['conversation_id']:r for r in rows} for name,rows in models.items()}
    items=[];script_counts={};review_rows=[]
    for source in manifest:
        cid=source['conversation_id'];ref=json.loads(Path(source['reference_path']).read_text())['reference_text']
        record=dict(id=cid,language=source['language'],duration=source['duration_seconds'],reference=ref,
                    audio=os.path.relpath(source['audio_path'],OUT),audio_sha256=source['audio_sha256'],
                    reference_sha256=source['reference_sha256'],systems={})
        for name,predictions in bymodel.items():
            if cid not in predictions:continue
            r=predictions[cid];refscript=distribution(ref)['dominant_script'];hypdist=distribution(r['text_raw'])
            mismatch=bool(refscript and hypdist['dominant_script'] and refscript!=hypdist['dominant_script'])
            route=r.get('route','native OmniASR CTC, bfloat16, full waveform' if name=='OmniASR CTC-1B' else 'native Qwen API, bfloat16, greedy')
            record['systems'][name]=dict(text=r['text_raw'],wer=r['metrics']['wer'],cer=r['metrics']['cer'],
                status=r['status'],route=route,dominant_script=hypdist['dominant_script'],script_mismatch=mismatch)
            key=(name,source['language']);c=script_counts.setdefault(key,dict(n=0,dominant_script_mismatches=0,unknown_dominant_script=0))
            c['n']+=1;c['dominant_script_mismatches']+=mismatch;c['unknown_dominant_script']+=hypdist['dominant_script'] is None
        items.append(record)
    for name in models:
        for lang in ['hi','bn','as']:
            eligible=sorted((r for r in items if r['language']==lang and name in r['systems']),key=lambda r:(r['systems'][name]['wer'],r['id']))
            if not eligible:continue
            for tag,index in [('lowest observed WER',0),('middle observed WER',len(eligible)//2),('highest observed WER',-1)]:
                r=eligible[index];review_rows.append(dict(model=name,language=lang,selection=tag,conversation_id=r['id'],
                      wer=r['systems'][name]['wer'],human_review_status='not reviewed'))
    script_report=dict(scope='Unicode dominant-script diagnostic; not language identification or human semantic review',
                       groups=[dict(model=k[0],language=k[1],**v) for k,v in script_counts.items()])
    (OUT/'script_diagnostics.json').write_text(json.dumps(script_report,indent=2)+'\n')
    (OUT/'human_review_queue.json').write_text(json.dumps(review_rows,indent=2)+'\n')
    payload=json.dumps(dict(items=items,review_queue=review_rows),ensure_ascii=False,separators=(',',':')).replace('<','\\u003c')
    html='''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>ASR evidence viewer — C. Kaustubh</title><style>
:root{font:16px/1.5 system-ui,sans-serif;color:#182b3c;background:#f4f6f7}body{max-width:1180px;margin:auto;padding:28px}h1{font-size:28px;margin-bottom:4px}h2{font-size:19px}.muted{color:#526577}.notice{border-left:4px solid #167878;padding:12px 16px;background:#e6f2f0}.controls{display:flex;gap:14px;flex-wrap:wrap;margin:24px 0}label{display:flex;flex-direction:column;gap:4px}select,input,button{font:inherit;padding:8px;border:1px solid #b3bfc7;border-radius:6px;background:white}#clip{width:min(620px,85vw)}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:16px}article,.reference{background:white;border:1px solid #d3dce1;padding:18px;border-radius:8px;margin:12px 0}article h2{margin-top:0}.text{white-space:pre-wrap;overflow-wrap:anywhere}.metric{font-size:22px;font-weight:650}.badge{font-size:13px;padding:3px 7px;background:#e9eff4;border-radius:4px}.flag{background:#fae6d0}.hash{overflow-wrap:anywhere;font:12px/1.5 monospace}summary{cursor:pointer}audio{width:min(580px,100%)}#review{max-width:420px}a{color:#00696b}footer{margin-top:32px;font-size:14px}
</style><body>
<h1>ASR evidence viewer</h1><p class="muted">C. Kaustubh · FLEURS official test · 2,322 recordings · 5,062 primary predictions</p>
<p class="notice">Whisper covers Hindi, Bengali and Assamese. Qwen covers Hindi only. Compare complete waveforms, references and saved outputs. Scores describe these declared system settings; they do not establish demographic fairness or a new method.</p>
<div class="controls"><label>Language<select id="lang"><option value="hi">Hindi · 418</option><option value="bn">Bengali · 920</option><option value="as">Assamese · 984</option></select></label><label>Find recording<input id="search" placeholder="Recording ID or reference text"></label><label>Sort by<select id="sort"><option value="id">Recording ID</option><option value="duration">Longest first</option><option value="wer">Largest Whisper Large-v3 WER first</option></select></label></div>
<label>Recording<select id="clip"></select></label><p id="count" class="muted"></p>
<label>Unreviewed examples — selected by observed WER<select id="review"><option value="">Choose low / middle / high example</option></select></label>
<section id="detail"></section>
<footer><p>Scorer: NFC, casefold, punctuation to spaces; meaningful Indic marks retained. WER may exceed 100% through insertions. Whisper ≤30 s: audited float32 HF beam-5; &gt;30 s: native transcription. Qwen: native API, bfloat16, greedy, batches up to eight; official postprocessing included.</p><p>Audio uses the verified local dataset beside this workspace. A copied HTML file retains transcripts but needs those audio files for playback. The review queue is unannotated; viewing an example does not mark it reviewed.</p><a href="../fleurs_qwen_comparison_v1/RESULTS.md">Hindi results</a> · <a href="../fleurs_whisper_corrected_v3/RESULTS.md">Whisper results</a> · <a href="../../BENCHMARK_COMPLETION.md">Evidence report</a></footer>
<script type="application/json" id="payload">PAYLOAD</script><script>
const data=JSON.parse(document.getElementById('payload').textContent);const $=id=>document.getElementById(id);let visible=[];
function text(tag,value,cls){const e=document.createElement(tag);e.textContent=value;if(cls)e.className=cls;return e;}
function render(){const r=data.items.find(x=>x.id===$('clip').value);const d=$('detail');d.replaceChildren();if(!r)return;
 d.append(text('h2',r.id),text('p',`${r.duration.toFixed(2)} seconds · mono 16 kHz · complete waveform`,'muted'));const audio=document.createElement('audio');audio.controls=true;audio.preload='none';audio.src=r.audio;d.append(audio);
 const reference=text('div','','reference');reference.append(text('h2','Dataset reference'),text('p',r.reference,'text'));d.append(reference);
 const grid=text('div','','grid');for(const [name,s] of Object.entries(r.systems)){const c=document.createElement('article');c.append(text('h2',name),text('p',`WER ${(s.wer*100).toFixed(2)}% · CER ${(s.cer*100).toFixed(2)}%`,'metric'),text('span',s.status,'badge'),text('p',s.route,'muted'),text('p',s.text||'(empty output)','text'),text('p',`Dominant tracked script: ${s.dominant_script||'none'}`,'muted'));if(s.script_mismatch)c.append(text('span','Script differs from reference — diagnostic flag','badge flag'));grid.append(c);}d.append(grid);
 const details=document.createElement('details');details.append(text('summary','Source hashes'),text('p','Audio SHA-256: '+r.audio_sha256,'hash'),text('p','Reference SHA-256: '+r.reference_sha256,'hash'));d.append(details);
}
function filter(){const query=$('search').value.toLocaleLowerCase();visible=data.items.filter(r=>r.language===$('lang').value&&(r.id.includes(query)||r.reference.toLocaleLowerCase().includes(query)));if($('sort').value==='duration')visible.sort((a,b)=>b.duration-a.duration);if($('sort').value==='wer')visible.sort((a,b)=>b.systems['Whisper Large-v3'].wer-a.systems['Whisper Large-v3'].wer);const old=$('clip').value;$('clip').replaceChildren();for(const r of visible){const o=text('option',r.id+' · '+r.duration.toFixed(2)+' s');o.value=r.id;$('clip').append(o);}if(visible.some(r=>r.id===old))$('clip').value=old;$('count').textContent=visible.length+' recordings shown';render();}
for(const r of data.review_queue){const o=text('option',`${r.model} · ${r.language} · ${r.selection}`);o.value=r.conversation_id;$('review').append(o);}
$('review').onchange=()=>{const r=data.items.find(x=>x.id===$('review').value);if(!r)return;$('lang').value=r.language;$('search').value='';filter();$('clip').value=r.id;render();};$('clip').onchange=render;$('lang').onchange=filter;$('sort').onchange=filter;$('search').oninput=filter;filter();
</script></body></html>'''
    if include_omni:
        html=html.replace('5,062 primary predictions','7,384 primary predictions')
        html=html.replace('Whisper covers Hindi, Bengali and Assamese.','Whisper and OmniASR CTC cover Hindi, Bengali and Assamese.')
        html=html.replace('<footer><p>','<footer><p>OmniASR CTC: native BF16 evaluation, complete waveforms, batch one, CTC token evidence; no autoregressive EOS.</p><p>',1)
        html=html.replace('<a href="../fleurs_qwen_comparison_v1/RESULTS.md">','<a href="../fleurs_omni_comparison_v1/RESULTS.md">OmniASR results</a> · <a href="../../listening_review/index.html">Blinded listening review</a> · <a href="../language_conditioning_analysis_v1/RESULTS.md">Language diagnostic</a> · <a href="../fleurs_qwen_comparison_v1/RESULTS.md">',1)
    (OUT/'index.html').write_text(html.replace('PAYLOAD',payload))
    receipt=dict(records=len(items),predictions=sum(len(r['systems']) for r in items),
         unreviewed_examples=len(review_rows),html_sha256=hashlib.sha256((OUT/'index.html').read_bytes()).hexdigest())
    (OUT/'build_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    return receipt

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--include-omni',action='store_true');parser.add_argument('--output');args=parser.parse_args()
    print(json.dumps(build(args.include_omni,args.output),indent=2))
