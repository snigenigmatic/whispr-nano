"""Portable H100 evidence bundle, including the disclosed earlier pilot history."""
import hashlib
import json
from pathlib import Path
import statistics
import zipfile

ROOT=Path(__file__).resolve().parents[2]
H=ROOT/'research/h100'
OUT=ROOT/'output/h100'


def sha(data):return hashlib.sha256(data).hexdigest()


def main():
    audit=json.loads((H/'results/audit.json').read_text());decision=json.loads((H/'results/decision.json').read_text())
    if audit['status']!='valid_complete' or len(audit['runs'])!=9:raise ValueError('Nine valid runs required')
    means={arm:100*statistics.mean(r['metrics']['final']['english']['macro_language_wer'] for r in audit['runs'] if r['arm_name']==arm)
           for arm in ('full_gold','filtered10_gold','random_matched_gold')}
    intro=f"""# Complete H100 experiment: start here

**All nine runs completed and passed independent audit.** Three input-selection
arms were trained across three seeds on H100, with the original fixed scientific
settings. The prior partial T4 study is preserved separately as history.

| Input pool | Mean English L1-macro WER |
|---|---:|
| Full reference pool | {means['full_gold']:.3f}% |
| Teacher-filtered pool | {means['filtered10_gold']:.3f}% |
| Matched random pool | {means['random_matched_gold']:.3f}% |

Filtered minus random WER was **{decision['mean_english_contrast_pp']:+.3f} percentage
points** under English normalization and
**{decision['mean_minimal_contrast_pp']:+.3f} points** under minimal normalization.
Positive would favor random selection. The registered harmful-selection
criterion was **not met**. This does not establish equivalence or rule out other
filtering effects. All targets were the supplied corpus references, so this
experiment does not demonstrate a new practical distillation method.

**Recommendation: do not submit this as a completed new-method paper.** It is a
completed, controlled pilot that can support an honest research decision.

## Read the evidence

- `output/pdf/h100_selection_research_report.pdf`: complete report, with a vector figure.
- `output/h100/research_report.md`: editable report.
- `research/h100/results/evidence.md`: every arm, seed, L1 and normalization.
- `research/h100/results/decision.json`: the fixed numerical decision.
- `research/h100/README.md`: runtime, registration and reproduction details.
- `docs/AUTORESEARCH_LOOP_REVIEW.md`: source-level upstream review and adaptations.
- `research/pilot/LOOP.md`: the evidence-controller API and integrity gates.

## Verify without GPU or Modal

From this extracted directory, using Python 3.12:

```bash
python3 -m venv .venv-audit
.venv-audit/bin/python -m pip install -r research/pilot/requirements-audit.txt
.venv-audit/bin/python -m research.h100.reproduce
```

Dependency installation may access the package index. The audit itself is
offline: no inference, audio download, network request, GPU, or Modal credential.
It verifies every published error count, reference and utterance identity,
training exposure sequence, hardware/source identity, and the registered
decision. `HASHES.sha256` covers every other archive entry; run
`shasum -a 256 -c HASHES.sha256` on macOS to check integrity.

## Models and retained history

The separate `trained_checkpoints.zip` contains all nine H100 model weights and
their saved processor/configuration files. `output/h100/model_index.json`
records the exact model hashes checked against the audited training receipts.
The model archive is separate to keep this evidence download compact. No audio
is redistributed here. Optimizer states and audio/features remain in the Modal
volumes described in the report.

`research/pilot/results/` and the older
`output/pdf/accent_selection_research_report.pdf` record the first stop with five
audited T4 runs. They are historical, not part of the nine-run H100 comparison.
The H100 study reused the same development speakers after those partial results
were inspected; all 18 additional accesses are explicitly disclosed. This is
not independent confirmation. Final-test recordings were never decoded or scored.

The execution ledgers preserve original absolute paths. The reproduction
command above resolves the extracted project layout and remains portable.
The known stopping-reason logging defect is disclosed in
`research/pilot/DIAGNOSTIC_FIELD_NOTE.md`; it does not enter scoring or promotion.
"""
    (ROOT/'START_HERE.md').write_text(intro)
    files=[]
    for folder in (ROOT/'research/pilot',H):
        for path in folder.rglob('*'):
            rel=path.relative_to(folder)
            if not path.is_file() or path.is_symlink() or any(x.startswith('.') or x=='__pycache__' for x in rel.parts):continue
            if path.suffix in ('.pid','.pyc','.lock') or path.name.startswith('spire_'):continue
            files.append(path)
    for rel in ('START_HERE.md','paper/data/whisper_normalizer.json','paper/analysis/analyze.py',
                'docs/AUTORESEARCH_LOOP_REVIEW.md','docs/RESEARCH_FRONTIER_20260909.md','docs/RESEARCH_RESET_20260909.md',
                'output/research_report.md','output/pdf/accent_selection_research_report.pdf',
                'output/pdf/h100_selection_research_report.pdf','output/h100/research_report.md',
                'output/h100/model_index.json','output/h100/CHECKPOINTS.md','output/h100/checkpoint_archive_receipt.json',
                'output/h100/figures/selection_contrast.pdf','output/h100/figures/selection_contrast.svg','output/h100/figures/selection_contrast.png'):
        path=ROOT/rel
        if not path.is_file():raise ValueError(f'Missing delivery artifact: {rel}')
        files.append(path)
    payloads={str(p.relative_to(ROOT)):p.read_bytes() for p in sorted(set(files))}
    payloads['HASHES.sha256']=''.join(f'{sha(data)}  {name}\n' for name,data in sorted(payloads.items())).encode()
    target=OUT/'evidence.zip'
    with zipfile.ZipFile(target,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for name,data in sorted(payloads.items()):z.writestr('h100-selection-evidence/'+name,data)
    with zipfile.ZipFile(target) as z:
        if z.testzip():raise ValueError('Evidence archive CRC failure')
        for name,data in payloads.items():
            if sha(z.read('h100-selection-evidence/'+name))!=sha(data):raise ValueError('Archive checksum mismatch')
    receipt={'archive':str(target),'bytes':target.stat().st_size,'files':len(payloads),'sha256':sha(target.read_bytes()),'crc_and_sha_verified':True,'audited_runs':9}
    (OUT/'archive_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt,indent=2))


if __name__=='__main__':main()
