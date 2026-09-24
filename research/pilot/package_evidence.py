"""Create a portable evidence archive from the completed, audited pilot."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import statistics
import zipfile

ROOT = Path(__file__).resolve().parents[2]
P = ROOT / 'research/pilot'
OUT = ROOT / 'output'


def sha(data):
    return hashlib.sha256(data).hexdigest()


def start_here(audit, decision):
    contrasts = {norm: [row['filtered_minus_random_macro_wer_pp']
                       for row in audit['paired_contrasts'] if row['normalization'] == norm]
                 for norm in ('english', 'minimal')}
    complete = audit['status'] == 'valid_complete'
    finding = (f"The complete nine-run experiment measured filtered minus matched-random "
               f"L1-macro WER of {statistics.mean(contrasts['english']):+.3f} percentage points "
               f"under English normalization and {statistics.mean(contrasts['minimal']):+.3f} "
               f"under minimal normalization. Positive values favor random selection. "
               f"The registered pilot criterion was {'met' if decision['pilot_positive'] else 'not met'}.") if complete else (
               f"The evidence is incomplete: {len(audit['runs'])}/9 required runs passed audit. "
               "No complete-experiment conclusion is available.")
    return f"""# Start here: controlled ASR research evidence

**Submission recommendation: do not submit this as a completed new-method paper.**

{finding}

This experiment isolates input selection while keeping corpus-provided reference
targets in every arm. It does not establish a new knowledge-distillation loss,
practical distillation improvement, or generalization across independently
sampled evaluation speakers. The report explains these limits and retains all
results, including unfavorable ones.

## Read the result

- `output/pdf/accent_selection_research_report.pdf`: self-contained report.
- `output/research_report.md`: the same narrative in editable form.
- `research/pilot/results/evidence.md`: numerical evidence from the independent auditor.
- `research/pilot/results/decision.json`: the registered numerical decision.
- `docs/AUTORESEARCH_LOOP_REVIEW.md`: source-level review of EvoMap/AutoResearch.
- `research/pilot/LOOP.md`: the implemented evidence controller and API.
- `research/pilot/README.md`: artifact map, verification, and training provenance.

## Recompute every reported score

From this extracted directory, using Python 3.12:

```bash
python3 -m venv .venv-audit
.venv-audit/bin/python -m pip install -r research/pilot/requirements-audit.txt
.venv-audit/bin/python -m research.pilot.reproduce
```

Dependency installation may access the package index. The audit itself is
offline: no Modal credentials, model inference, audio download, or paid compute.
It recomputes word errors from raw predictions, checks exact utterance coverage
and references, verifies frozen code/data identities, reconstructs all training
exposures, and compares every published metric with the delivered audit.

`HASHES.sha256` covers all other archive entries. On macOS, run
`shasum -a 256 -c HASHES.sha256` to verify their contents. File hashes provide
integrity checks, not an external trusted signature.

## What is preserved

The archive includes all nine registered configurations, teacher predictions,
completed student predictions, per-update losses and exposure sequences, runtime and
audio-hash receipts, selection pools and balance, data provenance, the explicit
pre-training matching amendment, evaluation-access ledger, tests, and source.
Final-test recordings were never decoded or scored. Test identities and supplied
references remain in the metadata so split disjointness can be checked.

The separate `trained_checkpoints.zip` contains the final model and processor
files for four completed runs; it is explicitly partial. Its `model_index.json` records the verified SHA-256
identities. Optimizer states and source audio/features remain in the named
Modal volumes documented in the report; the compact evidence archive contains
neither audio nor model tensors. Scores can be audited without those tensors.

The execution ledger preserves the original machine's absolute paths. Use the
portable reproduction command above after extraction. The historical launcher
is deliberately time- and budget-bounded; a new training study requires a new
experiment registration and output namespace.
"""


def selected_files():
    files = []
    for path in sorted(P.rglob('*')):
        if not path.is_file() or path.is_symlink():
            continue
        rel = path.relative_to(P)
        if any(part.startswith('.') or part == '__pycache__' for part in rel.parts):
            continue
        if path.suffix in ('.pyc', '.pid', '.lock') or path.name.startswith('spire_'):
            continue
        files.append(path)
    for rel in ('paper/data/whisper_normalizer.json', 'paper/analysis/analyze.py',
                'docs/AUTORESEARCH_LOOP_REVIEW.md', 'docs/RESEARCH_RESET_20260909.md',
                'docs/RESEARCH_FRONTIER_20260909.md', 'output/research_report.md',
                'output/pdf/accent_selection_research_report.pdf',
                'output/figures/pilot_contrast.png', 'output/figures/pilot_contrast.svg',
                'output/checkpoints/model_index.json', 'output/checkpoints/README.md',
                'START_HERE.md'):
        path = ROOT / rel
        if path.is_file():
            files.append(path)
    return sorted(set(files))


def main():
    audit = json.loads((P / 'results/audit.json').read_text())
    decision = json.loads((P / 'results/decision.json').read_text())
    (ROOT / 'START_HERE.md').write_text(start_here(audit, decision))
    payloads = {str(path.relative_to(ROOT)): path.read_bytes() for path in selected_files()}
    required = {'output/pdf/accent_selection_research_report.pdf', 'research/pilot/results/audit.json',
                'research/pilot/results/decision.json', 'research/pilot/reproduce.py'}
    if not required <= payloads.keys():
        raise ValueError(f'Missing required delivery files: {required-payloads.keys()}')
    payloads['HASHES.sha256'] = ''.join(f'{sha(data)}  {name}\n'
                                      for name, data in sorted(payloads.items())).encode()
    target = OUT / 'accent_selection_evidence.zip'
    with zipfile.ZipFile(target, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for name, data in sorted(payloads.items()):
            archive.writestr('accent-selection-evidence/' + name, data)
    with zipfile.ZipFile(target) as archive:
        if archive.testzip() is not None:
            raise ValueError('Archive CRC verification failed')
        for name, data in payloads.items():
            if sha(archive.read('accent-selection-evidence/' + name)) != sha(data):
                raise ValueError(f'Archive content mismatch: {name}')
    receipt = {'archive': target.name, 'sha256': sha(target.read_bytes()),
               'bytes': target.stat().st_size, 'files': len(payloads), 'crc_and_sha_verified': True,
               'model_tensors_included': False, 'audio_included': False}
    (OUT / 'evidence_archive_receipt.json').write_text(json.dumps(receipt, indent=2)+'\n')
    print(json.dumps(receipt, indent=2))


if __name__ == '__main__':
    main()
