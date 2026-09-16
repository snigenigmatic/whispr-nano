# Three-hour paper sprint

Started 2026-09-09 at 06:35 UTC (12:05 IST). Target finish: 09:35 UTC (15:05 IST).
User-authorized maximum Modal spending: **US$30**. Root agent alone launches paid jobs.

## Deliverable

A complete short empirical manuscript, reviewed PDF, evidence tables and figures,
and a reproducibility package. Publication suitability depends on the findings and
remaining limitations. This sprint does not submit or publish the paper externally.

## Work sequence

1. Freeze source hashes and cohorts; check inference access, dataset provenance,
   and genuine speaker or recording metadata. Repair portable execution paths.
2. Complete the five open models on the 294 Saaras-only clips. Verify decoding
   consistency on overlap clips before reusing historical predictions. If this
   check fails, run the full 450-clip panel under a new, consistently pinned
   configuration and clearly distinguish it from the historical results.
3. Analyze consistent normalization, native-language versus family aggregation,
   error tails, and uncertainty appropriate to the available grouping metadata.
   Absence of actual speaker IDs must not be hidden behind recording-level claims.
4. Finish manuscript and references from verified results, render the PDF, inspect
   every page, and run reproduction checks.

CPU analysis, dataset investigation, and manuscript drafting run in parallel.
New training, extra API purchases, and additional datasets are outside this sprint.

## Budget controls

Record every paid launch in `paper/inference/budget_ledger.json`. Reserve a
conservative maximum including GPU, requested CPU and memory, startup, and retries
before each launch. Use explicit timeouts, bounded concurrency, and incremental
prediction saves. Actual invoiced cost is reported only if verified; otherwise
label runtime-derived costs as estimates. Do not launch if the remaining reserve
would exceed $30. Initial experiment allocation: $15; contingency: $15.

## Claim rules

- This is a replication and reliability analysis, not the first accent audit.
- The historical Saaras and open-model aggregates use different samples.
- Bodo is the only represented Sino-Tibetan language; it is not the worst
  individual native-language group for every model.
- Independent Poisson significance is not evidence of population-wide or causal
  structural unfairness. Failure to reject is not evidence of equality.
- Conclusions may change or become descriptive if robust evidence is unavailable.

## Execution state

- Source hashes and cohort IDs: `paper/data/input_manifest.json`.
- Isolated Python environment installed at `.venv/`.
- No Modal credentials were initially configured on this machine; login started.
- Existing transcripts are retained unchanged.

## Extension after the matched-panel result

At approximately 06:57 UTC, the complete matched panel showed that one short
Tamil clip contributes 200 of Whisper-small's 621 word errors and changes the
observed worst-group conclusion. To measure sample sensitivity against the full
available benchmark, all five open recognizers are therefore being evaluated on
all 6,656 pinned test rows. This decision was made after the matched-panel result;
it is a documented retrospective extension, not a preregistered experiment.

The full run has 30 jobs (six disjoint shards per model), at most ten containers,
each with a 1,800-second function timeout, 300-second startup timeout, two-core
CPU limit, 16 GiB memory limit, and no function retries. Each job reserves $0.50,
covering 2,100 seconds at the published requested-resource rates with a margin.
Total reservations including all earlier jobs are $29.10, below the $30 ceiling.
All earlier reservations remain in the ledger rather than being silently reused.

## Final scope after the request for faster delivery

The optional full-benchmark run stopped on a provenance guard at an utterance
containing CRLF line endings: the host CSV reader had normalized those endings.
The reader was corrected to preserve them. Following the user's request to finish
faster, this extension was not retried. Its partial outputs are excluded from
every manuscript result. The final paper uses the completed Matched-450 and
Open-2500 analyses, as originally scoped. All Modal applications are stopped.

## Completed delivery

The five-page manuscript, figures, bibliography, supplement, stored predictions,
and offline reproduction package were completed by approximately 07:06 UTC on
2026-09-09, about 31 minutes after the sprint started. The complete reproduction
script passed, including all 22 tests; artifact validation confirmed unchanged
historical inputs and the six-system, 450-utterance matched panel. Every final
PDF page was visually inspected. This is an anonymous submission candidate,
not a submitted or accepted paper.
