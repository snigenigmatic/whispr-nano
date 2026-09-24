"""Build a blinded local listening/reference-review packet; no judgments are added."""
from __future__ import annotations
import hashlib, json, random
from pathlib import Path

ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'research/bridge_open/listening_review'
MANIFEST=ROOT/'data/bridge_open/fleurs_test_v1/corpus-v2.jsonl'
PRED_PATHS={
 'Whisper Turbo':ROOT/'research/bridge_open/artifacts/fleurs_whisper_corrected_v3/turbo_predictions.jsonl',
 'Whisper Large-v3':ROOT/'research/bridge_open/artifacts/fleurs_whisper_corrected_v3/large-v3_predictions.jsonl',
 'Qwen3-ASR-1.7B':ROOT/'research/bridge_open/artifacts/qwen_native_full_hi_v1/predictions.jsonl',
}
SYSTEM_LABELS=['system_A','system_B','system_C']
SEED=20260923

def digest(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def rows(p): return [json.loads(x) for x in Path(p).read_text().splitlines() if x.strip()]

def main():
 manifest_rows=rows(MANIFEST); byid={r['conversation_id']:r for r in manifest_rows}
 predictions={name:{r['conversation_id']:r for r in rows(path)} for name,path in PRED_PATHS.items()}
 targeted= json.loads((ROOT/'research/bridge_open/artifacts/benchmark_review/human_review_queue.json').read_text())
 targeted_byid={x['conversation_id']:x for x in targeted}
 rng=random.Random(SEED); random_ids=[]
 for lang in ('as','bn','hi'):
  candidates=[r['conversation_id'] for r in manifest_rows if r['language']==lang]
  random_ids.extend(sorted(rng.sample(candidates,10)))
 selection_info={}
 for cid in random_ids:
  info=selection_info.setdefault(cid, {'selection_kinds': [], 'targeted_reasons': []})
  if 'random_stratified' not in info['selection_kinds']: info['selection_kinds'].append('random_stratified')
 for cid in sorted(targeted_byid):
  info=selection_info.setdefault(cid, {'selection_kinds': [], 'targeted_reasons': []})
  if 'targeted_review_queue' not in info['selection_kinds']: info['selection_kinds'].append('targeted_review_queue')
  reason=targeted_byid[cid].get('selection')
  if reason and reason not in info['targeted_reasons']: info['targeted_reasons'].append(reason)
 selected=list(selection_info)
 assert len(selected)==len({x for x in selected})
 sealed=[]; review=[]
 for cid in selected:
  info=selection_info[cid]
  source=byid[cid]; assert source['language'] in {'as','bn','hi'}
  ref_path=Path(source['reference_path']); audio_path=Path(source['audio_path'])
  assert digest(audio_path)==source['audio_sha256']; assert digest(ref_path)==source['reference_sha256']
  ref=json.loads(ref_path.read_text())['reference_text']
  cards=[]
  labels=SYSTEM_LABELS.copy(); random.Random(f'{SEED}:{cid}').shuffle(labels)
  per_item_labels=dict(zip(PRED_PATHS, labels))
  for model in PRED_PATHS:
   pred=predictions[model].get(cid)
   if pred is None: continue
   assert pred['audio_sha256']==source['audio_sha256'] and pred['reference_sha256']==source['reference_sha256']
   assert pred['status'] in {'success','empty_output','generation_limit'}
   cards.append({'system_label':per_item_labels[model],'transcript':pred.get('text_raw','')})
   sealed.append({'review_id':cid,'system_label':per_item_labels[model],'system_name':model,'condition_id':pred.get('condition_id'),'prediction_status':pred['status'],'prediction_metrics':pred.get('metrics')})
  assert cards
  review.append({'review_id':cid,'selection_kinds':info['selection_kinds'],'targeted_reasons':info['targeted_reasons'],'language':source['language'],'released_language':source.get('released_language'),
    'audio_url':'../../../data/bridge_open/fleurs_test_v1/'+audio_path.name,
    'audio_path_relative':'data/bridge_open/fleurs_test_v1/'+audio_path.name,
    'audio_sha256':source['audio_sha256'],'reference_sha256':source['reference_sha256'],
    'reference_text':ref,'systems':cards,
    'review_status':'UNREVIEWED','reviewer_response':{'reference_correct':'','language_or_script_correct':'','meaning_preserved_by_system':{},'audio_quality':'','notes':''}})
 (OUT/'review_items.jsonl').write_text('\n'.join(json.dumps(x,ensure_ascii=False) for x in review)+'\n')
 (OUT/'review_items.json').write_text(json.dumps(review,ensure_ascii=False,indent=2)+'\n')
 (OUT/'sealed_mapping.json').write_text(json.dumps({'warning':'SEALED: do not share with blinded reviewer','seed':SEED,'mapping':sealed},indent=2,ensure_ascii=False)+'\n')
 # Blank CSV is intentionally reviewer-facing and contains no true model names.
 import csv
 with (OUT/'review_responses.csv').open('w',newline='') as f:
  w=csv.writer(f); w.writerow(['review_id','language','reference_correct','language_or_script_correct','system_label','meaning_preserved','audio_quality','notes'])
  for item in review:
   for card in item['systems']: w.writerow([item['review_id'],item['language'],'','',''+card['system_label'],'','',''])
 counts={l:sum(x['language']==l for x in review) for l in ('as','bn','hi')}
 targeted_distinct=len({x['conversation_id'] for x in targeted})
 (OUT/'packet_metadata.json').write_text(json.dumps({'packet_status':'UNREVIEWED','random_seed':SEED,'random_per_language':10,'targeted_source_entry_count':len(targeted),'targeted_distinct_count':targeted_distinct,'selected_deduplicated_count':len(review),'counts_by_language':counts,'targeted_ids':sorted(targeted_byid),'reviewer_view_excludes_true_system_names':True,'audio_server_root':'/Users/prateekraj/whispr-nano','source_manifest_sha256':digest(MANIFEST),'source_prediction_files':{k:{'path':str(v),'sha256':digest(v)} for k,v in PRED_PATHS.items()}},indent=2)+'\n')
 print(json.dumps({'selected':len(review),'counts_by_language':counts,'targeted_source_entries':len(targeted),'targeted_distinct':targeted_distinct,'deduplicated':len(review)}))
if __name__=='__main__': main()
