"""Offline review examples, drawn only from the public Speech Commands controls."""
import html
import io
import json
from pathlib import Path
import tarfile
import hashlib
import numpy as np
import soundfile as sf

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'results/night1/demo'


def main():
    evidence=json.loads((ROOT/'results/night1/cloud_decoder_validated_20261013.json').read_text())
    archive=Path.home()/'.cache/whispr-nano/speech_commands_test_set_v0.02.tar.gz'
    assert hashlib.sha256(archive.read_bytes()).hexdigest()=='cc2a00c1147c2254e9be3fa0f779d8c17421dc349b86366567a8edfa9acd51df'
    choices=[]
    for mode,cohort,kind in [('official_default','commands_repeat_5','correct speech damaged'),
                              ('hf_turbo_beam5','commands_repeat_8','correct speech damaged'),
                              ('hf_turbo_beam5','commands_repeat_5','large insertion error reduced')]:
        folder=Path(evidence['runs'][mode]['private_artifacts'])
        rows=[json.loads(l) for l in (folder/'predictions.jsonl').read_text().splitlines()]
        candidates=[r for r in rows if r['cohort']==cohort]
        if kind=='correct speech damaged':
            candidates=[r for r in candidates if r['raw_counts']['word_errors']==0 and r['repair_counts']['word_errors']>0]
            r=min(candidates,key=lambda r:r['id'])
        else:r=max(candidates,key=lambda r:r['raw_counts']['word_errors']-r['repair_counts']['word_errors'])
        choices.append((mode,kind,r))
    OUT.mkdir(parents=True,exist_ok=True)
    examples=[];cards=[]
    with tarfile.open(archive) as tf:
        for index,(mode,kind,r) in enumerate(choices,1):
            raw=tf.extractfile(r['source_id']).read()
            a,sr=sf.read(io.BytesIO(raw),dtype='float32');assert sr==16000
            pieces=[]
            for k in range(r['n_repeats']):
                if k:pieces.append(np.zeros(2400,dtype=np.float32))
                pieces.append(a)
            signal=np.concatenate(pieces)
            assert hashlib.sha256(signal.astype('<f4').tobytes()).hexdigest()==r['audio_sha256']
            filename=f'example-{index}.wav';sf.write(OUT/filename,signal,16000,subtype='FLOAT')
            e={k:r[k] for k in ('id','reference','prediction','repaired_prediction','audio_sha256','raw_counts','repair_counts','source_id','n_repeats')}
            e.update(mode=mode,selection=kind,audio_file=filename);examples.append(e)
            esc=html.escape
            cards.append(f'''<article><p class="tag">{index} · {esc(kind)} · {esc(mode)}</p>
<audio controls preload="none" src="{filename}"></audio>
<dl><dt>Known spoken words</dt><dd>{esc(r['reference'])}</dd>
<dt>Actual model output</dt><dd class="output">{esc(r['prediction'])}</dd>
<dt>After text collapse</dt><dd class="output">{esc(r['repaired_prediction'])}</dd></dl>
<p><strong>Word edits: {r['raw_counts']['word_errors']} → {r['repair_counts']['word_errors']}</strong> / {r['raw_counts']['ref_words']} reference words.</p>
<details><summary>Source and audio hash</summary><code>{esc(r['source_id'])}<br>{r['audio_sha256']}</code></details></article>''')
    (OUT/'examples.json').write_text(json.dumps(examples,indent=2,ensure_ascii=False)+'\n')
    (OUT/'index.html').write_text('''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>ASR repetition audit · C. Kaustubh</title><style>
body{font:16px/1.6 system-ui,sans-serif;background:#f5f3ee;color:#17252b;max-width:920px;margin:40px auto;padding:0 22px}h1{font-size:34px;line-height:1.2}article{background:white;border:1px solid #d1d8d8;border-radius:12px;padding:22px;margin:22px 0}dt,.tag{font-weight:650}dd{margin:5px 0 15px;overflow-wrap:anywhere}.output{max-height:150px;overflow:auto;padding:10px;background:#f4f7f7}audio{width:100%;margin:12px 0}code{font-size:12px;overflow-wrap:anywhere}a{color:#116e85}img{max-width:100%}footer{font-size:13px}strong{color:#7b2b0b}
</style><h1>Lower WER can hide damaged speech</h1><p>C. Kaustubh · Reproducible cloud inference audit · 13 October 2026</p>
<p>These are actual decoded outputs. The audio repeats a labelled human word recording with 150 ms gaps. Listen, then compare the original model output with a rule that collapses five or more repeated words.</p>
<p><strong>Illustrative cases selected after scoring; not a random sample.</strong> The full paired study contains 100 original keyword recordings, each tested with one, five and eight copies. These constructed controls do not represent natural conversational repetition.</p>
<img src="../figures/repetition_tradeoff.svg" alt="Pooled WER improves while text collapse harms 77 of 100 five-repeat utterances.">'''+''.join(cards)+'''
<footer><p>Audio: <a href="https://www.tensorflow.org/datasets/catalog/speech_commands">Speech Commands v0.02</a>, Pete Warden / TensorFlow, CC BY 4.0. Adaptation: deterministic selection and waveform concatenation. No Lahaja audio, gated transcripts, or API keys are included.</p><p>These examples diagnose failure; they do not establish a publishable method or an Indian-English improvement. <a href="examples.json">Download exact example records</a>.</p></footer></html>''')
    print(OUT/'index.html')


if __name__=='__main__':main()
