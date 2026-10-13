"""All reference-marked natural repetitions among non-development speakers."""
import hashlib
from pathlib import Path
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download
from research.lahaja import baseline as b


def max_run(text):
    longest=streak=0;last=None
    for word in b._normalize(text).split():
        streak=streak+1 if word==last else 1;last=word;longest=max(longest,streak)
    return longest


def prepare():
    token=b._token()
    dev=hf_hub_download(b.DATASET_ID,b.DEV_FILE,revision=b.DATASET_REVISION,repo_type='dataset',token=token)
    excluded=set(pq.read_table(dev,columns=['sp_id']).column('sp_id').to_pylist())
    paths={f:Path(hf_hub_download(b.DATASET_ID,f,revision=b.DATASET_REVISION,repo_type='dataset',token=token)) for f in b.DATA_FILES}
    all_rows,_=b._metadata(paths,excluded)
    chosen=[r for r in all_rows if 0<r['duration']<=25 and max_run(r['verbatim'])>=3]
    assert len(chosen)==27
    audio=b._audio(paths,{(r['data_file'],r['row_index']) for r in chosen})
    records=[]
    for r in chosen:
        a,sr=audio[(r['data_file'],r['row_index'])]
        assert sr==16000 and abs(len(a)/sr-r['duration'])<.005
        records.append(dict(id=f"{r['data_file']}:{r['row_index']}",cohort='natural_hindi_repetition',
            sp_id=r['sp_id'],reference=r['verbatim'],language='hi',audio=a,
            audio_sha256=hashlib.sha256(a.astype('<f4').tobytes()).hexdigest(),data_file=r['data_file'],
            row_index=r['row_index'],scenario=r['scenario'],native_state=r['native_state'],
            duration_s=len(a)/sr,reference_max_adjacent_run=max_run(r['verbatim'])))
    return records
