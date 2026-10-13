"""Unused short Hindi speech stratum, controlled repetition, and silence."""
from collections import defaultdict
import hashlib
import io
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
import soundfile as sf
from huggingface_hub import hf_hub_download
from research.lahaja import baseline as b

SILENCE_REV='8126be4a4b83afefeef9394ff1312b635a499d82'

def prepare():
    token=b._token()
    dev=hf_hub_download(b.DATASET_ID,b.DEV_FILE,revision=b.DATASET_REVISION,repo_type='dataset',token=token)
    excluded=set(pq.read_table(dev,columns=['sp_id']).column('sp_id').to_pylist())
    paths={f:Path(hf_hub_download(b.DATASET_ID,f,revision=b.DATASET_REVISION,repo_type='dataset',token=token)) for f in b.DATA_FILES}
    all_rows,_=b._metadata(paths,excluded)
    chosen=[r for r in all_rows if 0<r['duration']<2 and len(b._normalize(r['verbatim']).split())==1]
    assert len(chosen)==414 and len({r['sp_id'] for r in chosen})==102
    audios=b._audio(paths,{(r['data_file'],r['row_index']) for r in chosen})
    records=[];groups=defaultdict(list)
    for r in chosen:
        a,sr=audios[(r['data_file'],r['row_index'])]
        assert sr==16000 and abs(len(a)/sr-r['duration'])<.005 and np.max(np.abs(a))>0
        base=dict(id=f"{r['data_file']}:{r['row_index']}",cohort='hindi_short_natural',
            sp_id=r['sp_id'],reference=r['verbatim'],language='hi',audio=a,
            audio_sha256=hashlib.sha256(a.astype('<f4').tobytes()).hexdigest(),
            data_file=r['data_file'],row_index=r['row_index'],scenario=r['scenario'],
            native_state=r['native_state'],duration_s=len(a)/sr)
        records.append(base);groups[r['sp_id']].append(base)
    for speaker,options in sorted(groups.items()):
        source=min(options,key=lambda r:hashlib.sha256(('short-hindi-20261013|'+r['id']).encode()).hexdigest())
        for repeats in (5,8):
            pieces=[]
            for k in range(repeats):
                if k:pieces.append(np.zeros(2400,dtype=np.float32))
                pieces.append(source['audio'])
            a=np.concatenate(pieces);assert len(a)<30*16000
            row=dict(source);row.update(id=f"{source['id']}|repeat={repeats}",source_id=source['id'],
                cohort=f'hindi_repeat_{repeats}',n_repeats=repeats,reference=' '.join([source['reference']]*repeats),
                audio=a,audio_sha256=hashlib.sha256(a.astype('<f4').tobytes()).hexdigest(),duration_s=len(a)/16000)
            records.append(row)
    f=hf_hub_download('Scicom-intl/Whisper-Hallucination','data/silence/test-00000-of-00001.parquet',
        revision=SILENCE_REV,repo_type='dataset',token=token)
    silence=pq.read_table(f).to_pylist();assert len(silence)==42
    for r in silence:
        assert r['reference_text']==''
        a,sr=sf.read(io.BytesIO(r['audio']['bytes']),dtype='float32',always_2d=True)
        assert sr==16000 and a.shape[1]==1 and np.isfinite(a).all()
        source_duration=len(a)/sr
        a=a[:25*16000,0] # Uniform explicit first-25s view; never silently truncate inside the processor.
        records.append(dict(id='silence:'+r['id'],cohort='silence_control',sp_id=r['id'],reference='',
            language='hi',audio=a,audio_sha256=hashlib.sha256(a.astype('<f4').tobytes()).hexdigest(),
            duration_s=len(a)/sr,source_duration_s=source_duration,floor=r['floor'],crop_policy='first 25 seconds maximum'))
    assert len(records)==660
    # Cropping makes several controls byte-identical; avoid treating copies as new evidence.
    kept=[];seen={}
    for row in records:
        if row['cohort']!='silence_control':kept.append(row);continue
        key=row['audio_sha256']
        if key not in seen:
            row['equivalent_source_ids']=[row['id']];seen[key]=row;kept.append(row)
        else:seen[key]['equivalent_source_ids'].append(row['id'])
    assert len(kept)==654 and len(seen)==36
    return kept
