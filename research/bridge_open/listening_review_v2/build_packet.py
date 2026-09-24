"""Build blinded v2 listening packet from frozen v1 cohort; no judgments are added."""
from __future__ import annotations
import hashlib,json,random
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'research/bridge_open/listening_review_v2'
V1=ROOT/'research/bridge_open/listening_review'
MANIFEST=ROOT/'data/bridge_open/fleurs_test_v1/corpus-v2.jsonl'
PRED_PATHS={
 'Whisper Turbo':ROOT/'research/bridge_open/artifacts/fleurs_whisper_corrected_v3/turbo_predictions.jsonl',
 'Whisper Large-v3':ROOT/'research/bridge_open/artifacts/fleurs_whisper_corrected_v3/large-v3_predictions.jsonl',
 'Qwen3-ASR-1.7B':ROOT/'research/bridge_open/artifacts/qwen_native_full_hi_v1/predictions.jsonl',
 'OmniASR CTC 1B':ROOT/'research/bridge_open/artifacts/fleurs_omni_comparison_v1/predictions.jsonl',
}
SEED=20260924
LABELS=['system_A','system_B','system_C','system_D']
def digest(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def rows(p): return [json.loads(x) for x in Path(p).read_text().splitlines() if x.strip()]
def main():
 manifest={r['conversation_id']:r for r in rows(MANIFEST)}
 cohort=json.loads((V1/'review_items.json').read_text())
 assert len(cohort)==50 and len({x['review_id'] for x in cohort})==50
 predictions={name:{r['conversation_id']:r for r in rows(path)} for name,path in PRED_PATHS.items()}
 sealed=[]; review=[]; missing=[]
 for item in cohort:
  cid=item['review_id']; source=manifest[cid]
  assert item['selection_kinds'] and source['language'] in {'as','bn','hi'}
  audio=Path(source['audio_path']); refpath=Path(source['reference_path'])
  assert digest(audio)==source['audio_sha256'] and digest(refpath)==source['reference_sha256']
  ref=json.loads(refpath.read_text())['reference_text']; labels=LABELS.copy(); random.Random(f'{SEED}:{cid}').shuffle(labels)
  cards=[]
  for (model,path),label in zip(PRED_PATHS.items(),labels):
   pred=predictions[model].get(cid)
   if pred is None: continue
   assert pred['audio_sha256']==source['audio_sha256'] and pred['reference_sha256']==source['reference_sha256']
   assert pred['status'] in {'success','empty_output','generation_limit'}
   text=pred.get('text_raw','')
   cards.append({'system_label':label,'transcript':text})
   sealed.append({'review_id':cid,'system_label':label,'system_name':model,'condition_id':pred.get('condition_id'),'prediction_status':pred['status'],'prediction_metrics':pred.get('metrics'),'prediction_audio_sha256':pred['audio_sha256'],'prediction_reference_sha256':pred['reference_sha256']})
  assert cards, cid
  review.append({'review_id':cid,'selection_kinds':item['selection_kinds'],'language':source['language'],'released_language':source.get('released_language'),'audio_url':'../../../data/bridge_open/fleurs_test_v1/'+audio.name,'audio_path_relative':'data/bridge_open/fleurs_test_v1/'+audio.name,'audio_sha256':source['audio_sha256'],'reference_sha256':source['reference_sha256'],'reference_text':ref,'systems':cards,'review_status':'UNREVIEWED','reviewer_response':{'reference_correct':'','audio_quality':'','language_or_script_by_system':{},'meaning_preserved_by_system':{},'notes':''}})
 (OUT/'review_items.json').write_text(json.dumps(review,ensure_ascii=False,indent=2)+'\n')
 (OUT/'review_items.jsonl').write_text('\n'.join(json.dumps(x,ensure_ascii=False) for x in review)+'\n')
 (OUT/'sealed_mapping.json').write_text(json.dumps({'warning':'SEALED: do not share with blinded reviewer','seed':SEED,'mapping':sealed},ensure_ascii=False,indent=2)+'\n')
 import csv
 with (OUT/'review_responses.csv').open('w',newline='') as f:
  w=csv.writer(f);w.writerow(['review_id','language','reference_correct','audio_quality','system_label','language_or_script','meaning_preserved','notes'])
  for i in review:
   for c in i['systems']:w.writerow([i['review_id'],i['language'],'','',c['system_label'],'','',''])
 counts={l:sum(i['language']==l for i in review) for l in ('as','bn','hi')}
 meta={'packet_name':'listening_review_v2','packet_status':'UNREVIEWED','random_seed':SEED,'fixed_cohort_source':'../listening_review/review_items.json','fixed_cohort_count':len(review),'counts_by_language':counts,'selection_flag_counts':{k:sum(k in i['selection_kinds'] for i in review) for k in ('random_stratified','targeted_review_queue')},'targeted_distinct_count':len({i['review_id'] for i in review if 'targeted_review_queue' in i['selection_kinds']}),'reviewer_view_excludes_true_system_names_and_scores':True,'source_manifest_sha256':digest(MANIFEST),'source_prediction_files':{k:{'path':str(v),'sha256':digest(v)} for k,v in PRED_PATHS.items()}}
 meta['available_prediction_card_counts']={model:sum(cid in predictions[model] for cid in manifest if cid in {i['review_id'] for i in review}) for model in PRED_PATHS}
 (OUT/'packet_metadata.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2)+'\n')
 print(json.dumps({'count':len(review),'languages':counts,'cards':len(sealed),'models':meta['available_prediction_card_counts']}))
if __name__=='__main__':main()
