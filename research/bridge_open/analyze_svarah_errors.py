"""Post-hoc text/length diagnostics; no language or semantic judgments."""
import json
import unicodedata
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ART = ROOT/'research/bridge_open/artifacts'
BUNDLE = ROOT/'data/bridge_open/svarah_native_shard02_full_v1/bundle'


def rows(path): return [json.loads(line) for line in path.read_text().splitlines()]


def script(character):
    name=unicodedata.name(character,'')
    for label in ('LATIN','ARABIC','DEVANAGARI','BENGALI','CYRILLIC','GREEK','HEBREW','TAMIL','TELUGU','KANNADA','MALAYALAM','THAI','HANGUL','HIRAGANA','KATAKANA'):
        if label in name:return label
    if 'CJK' in name or 'IDEOGRAPH' in name:return 'HAN'
    return 'UNMAPPED_LETTER'


def analyze():
    manifest={r['conversation_id']:r for r in rows(BUNDLE/'manifest.jsonl')}
    result={'analysis_type':'posthoc descriptive text diagnostics; no human language/meaning judgments',
            'selection':'All 518 accepted paired recordings; example rows selected by descending word-edit disagreement, then ID',
            'script_rule':'Unicode names of category-L characters; mixed script is not a language classification',
            'models':{}}
    predictions={}
    for model in ['qwen','omni']:
        p=ART/f'{model}_svarah_full_v1'
        accepted=json.loads((p/'local_independent_acceptance.json').read_text())
        if not accepted['valid'] or accepted['n']!=518:raise ValueError('Run not independently accepted')
        rs=rows(p/'predictions.jsonl');predictions[model]={r['conversation_id']:r for r in rs}
        if set(predictions[model])!=set(manifest):raise ValueError('Prediction/source cohort mismatch')
        classifications=[];characters=Counter()
        for r in rs:
            counts=Counter(script(c) for c in r['text_raw'] if unicodedata.category(c).startswith('L'))
            characters.update(counts)
            classifications.append(dict(conversation_id=r['conversation_id'],letters_by_script=dict(counts),
                mixed_arabic_latin='LATIN' in counts and 'ARABIC' in counts,
                any_non_latin_letter=any(k!='LATIN' for k in counts)))
        counts={'recordings':len(rs),'any_non_latin_letter':sum(r['any_non_latin_letter'] for r in classifications),
                'mixed_arabic_latin':sum(r['mixed_arabic_latin'] for r in classifications),
                'no_letter':sum(not r['letters_by_script'] for r in classifications)}
        duration_bins=[]
        for lower,upper in [(0,2),(2,5),(5,float('inf'))]:
            selected=[r for r in rs if lower<=manifest[r['conversation_id']]['duration_seconds']<upper]
            words=sum(r['metrics']['ref_words'] for r in selected)
            components={k:sum(r['metrics'][k] for r in selected) for k in ('substitutions','deletions','insertions')}
            duration_bins.append(dict(lower_seconds=lower,upper_seconds=None if upper==float('inf') else upper,
                n=len(selected),reference_words=words,**components,wer=sum(components.values())/words))
        result['models'][model]=dict(counts=counts,letter_counts=dict(characters),duration_bins=duration_bins,
            word_errors={k:sum(r['metrics'][k] for r in rs) for k in ('substitutions','deletions','insertions')},records=classifications)
    def edits(r):return sum(r['metrics'][k] for k in ('substitutions','deletions','insertions'))
    ranked=sorted(manifest,key=lambda uid:(-abs(edits(predictions['omni'][uid])-edits(predictions['qwen'][uid])),uid))
    result['examples']=[]
    for uid in ranked[:12]:
        ref=json.loads((BUNDLE/'references'/Path(manifest[uid]['reference_path']).name).read_text())['reference_text']
        result['examples'].append(dict(conversation_id=uid,duration_seconds=manifest[uid]['duration_seconds'],reference=ref,
            qwen=predictions['qwen'][uid]['text_raw'],omni=predictions['omni'][uid]['text_raw'],
            qwen_word_errors=edits(predictions['qwen'][uid]),omni_word_errors=edits(predictions['omni'][uid])))
    out=ART/'svarah_error_diagnostics_v1';out.mkdir(exist_ok=True)
    (out/'analysis.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    md=['# Svarah text-error diagnostics','',
        'Post-hoc inspection of the accepted 518-record subset. These checks describe written outputs and edit counts; they do not establish spoken language, semantic accuracy or the cause of a model failure. Primary scoring and cohort membership are unchanged.','',
        '| System | Outputs with non-Latin letters | Outputs mixing Arabic and Latin letters | Outputs with no letters |',
        '|---|---:|---:|---:|']
    for name,r in result['models'].items():
        c=r['counts'];md.append(f"| {name} | {c['any_non_latin_letter']} | {c['mixed_arabic_latin']} | {c['no_letter']} |")
    md+=['','Unicode letter names are a script heuristic, not a language classifier. Empty outputs and punctuation-only outputs count as having no letters. Arabic-script characters do not by themselves establish Arabic language.','',
        '| System | Duration | Recordings | Reference words | Micro WER |','|---|---|---:|---:|---:|']
    for name,r in result['models'].items():
        for b in r['duration_bins']:
            label=f"{b['lower_seconds']}–<{b['upper_seconds']} s" if b['upper_seconds'] is not None else '≥5 s'
            md.append(f"| {name} | {label} | {b['n']} | {b['reference_words']} | {b['wer']:.2%} |")
    md+=['','Duration bins were chosen after the primary results. They are descriptive, not a pre-registered hypothesis test. The gap also occurs among recordings ≥5 s, so it is not confined to very short utterances. That observation does not isolate the effect of duration, accent or audio quality.','',
        'The [machine-readable profile](analysis.json) preserves letter counts, word-edit components, per-record flags and the 12 largest paired disagreements with references and both outputs. Selection emphasizes disagreements and cannot estimate error prevalence. No listening judgments have been filled in.','',
        '## Next controlled question','',
        'The declared conditions provide different language information: Qwen receives explicit English, whereas the native Omni CTC path has no corresponding language input. This comparison does not isolate model architecture from language conditioning.','',
        'Verify the most discrepant audio/reference pairs and inspect the native CTC frame/blank and output traces. If testing a script-constrained decoder or input preprocessing, define the rule on separate development data and compare against the unchanged accepted native baseline. Treat this inspected Svarah subset as exploratory; reserve independent confirmation for a frozen method. A Unicode/transliteration patch alone must not be reported as recovered spoken meaning.','']
    (out/'RESULTS.md').write_text('\n'.join(md))
    print(json.dumps({name:{k:v for k,v in r.items() if k!='records'} for name,r in result['models'].items()},indent=2))


if __name__=='__main__':analyze()
