# Svarah replication and reliability study

**Status, September 9, 2026:** retained as a technical report and diagnostic
artifact; **not recommended for conference/journal submission in its current
form**. The reviewer assessment and replacement research plan are in
[`../docs/RESEARCH_RESET_20260909.md`](../docs/RESEARCH_RESET_20260909.md).
This supersedes the sprint's earlier "submission candidate" description.

This package revisits the historical six-system audit with matched samples,
consistent text normalization, native-language breakdowns, and uncertainty
diagnostics. It contains a manuscript and the evidence needed to inspect it.

## Read the paper

- Manuscript source: [`manuscript/manuscript.md`](manuscript/manuscript.md)
- Rendered paper: [`../output/pdf/svarah_reliability_paper.pdf`](../output/pdf/svarah_reliability_paper.pdf)
- References: [`manuscript/references.bib`](manuscript/references.bib)
- Retrospective analysis protocol: [`analysis/PROTOCOL.md`](analysis/PROTOCOL.md)
- Source and identity limitations: [`metadata/README.md`](metadata/README.md)

## Reproduce without compute spending

Run from the repository root. Python 3.12 and `uv` are used here:

```bash
uv venv .venv --python 3.12
uv pip sync --python .venv/bin/python paper/requirements.lock.txt
bash paper/reproduce.sh
```

The script uses only stored predictions and metadata. It validates source hashes,
checks the paired panels, regenerates scoring and bootstrap diagnostics, builds
figures, runs regression tests, and renders the manuscript. No Modal or Sarvam
request is made. The PDF builder currently uses macOS Times New Roman and Menlo
fonts; the analysis and figures are independent of the PDF typography.

The main result files are `results/metrics.csv`, `results/contrasts.csv`,
`results/support.csv`, `results/diagnostics.json`, and `results/analysis.json`.
The parallel `results/demographic_profile/` directory changes only the assumed
grouping used in the exploratory bootstrap. Both variants use 2,000 resamples
and seed 20260909.

## What the panels mean

- **Open-2500:** unchanged historical predictions from five open systems on the
  same 2,500 utterances.
- **Matched-450:** fresh, pinned predictions from those five open systems on the
  450 archived Saaras utterances. Saaras predictions are copied unchanged.
- Their intersection has 156 utterances. These are overlapping samples of the
  same corpus, not independent datasets or replications.

The historical 450-versus-2,500 aggregate ranking was not a matched comparison.
Its independent Poisson p-values must not be read as established structural or
speaker-population disparities. Source CSVs remain unchanged for traceability.

## Metadata and uncertainty

The pinned release and the recovered original metadata/manifest contain no
verified speaker identifier. Filename blocks and complete demographic profiles
are explicitly labeled grouping assumptions, not identified speakers. All
speaker-inference p-values are withheld. Intervals are diagnostic or exploratory,
and neither grouping guarantees speaker independence. Bodo is the sole represented
language in the former Sino-Tibetan category; it has five unverified demographic
profiles. A result about Bodo is not a result about that entire language family.

## Optional fresh transcription

`inference/modal_panel.py` reproduces open-model transcription with recorded
checkpoint revisions and runtime versions. It requires authorized Modal access,
the `huggingface` secret with dataset access, and the named data/cache volumes.
It uses T4 GPUs, float32, greedy decoding and batch size four. Partial results are
checkpointed in a Modal output volume.

Paid launches are separated from reproduction. The sprint ledger records a
$30 maximum, conservative reservations, execution state, and available runtime
estimates. It does not claim to be an invoice. The runner refuses new launches
after the sprint deadline or if reservations would exceed the budget.

## Attribution and scope

The source corpus is AI4Bharat Svarah (Javed et al., Interspeech 2023), distributed
under the dataset card's CC BY 4.0 terms. Source links and revision identifiers
are recorded in the metadata evidence and bibliography. No new participants were
recruited and no raw audio is included in this package. The manuscript is an
anonymous research draft; it has not been submitted or accepted by a venue.
