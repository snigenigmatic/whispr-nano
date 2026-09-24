# Blinded listening-review handoff

The frozen v2 packet contains 50 items and 168 anonymous transcript cards. Its
entire cohort is still `UNREVIEWED`; this handoff does not assign ratings,
alter the packet, or open the sealed mapping.

Use reviewers only for languages they state they can competently judge. The
small assignment plan is 18 Hindi clips to a Hindi-competent reviewer, 16
Bengali clips to a Bengali-competent reviewer, and 16 Assamese clips to an
Assamese-competent reviewer. Assign ten clips as a second-reviewer overlap,
with both reviewers competent in the assigned clip language. Keep the overlap
responses independently blinded. Adjudication is not computed until real,
independent review data exist.

No reviewer has been contacted and no dates are promised by this document.

## Reviewer procedure

Serve the repository root over HTTP so relative audio paths resolve, then open
`/research/bridge_open/listening_review_v2/index.html` and download the JSON export. Do not give reviewers
`sealed_mapping.json`, model names, metric scores, or a modified packet.
Blank items must stay in the downloaded JSON: incomplete work is a valid
handoff state.

An analyst imports a download with the reviewer's pseudonym and their actual
language competency, for example:

```sh
.venv/bin/python research/bridge_open/core_study/review_import.py \
  --input listening_review_responses.json \
  --reviewer reviewer_hindi_01 \
  --languages hi \
  --output-dir research/bridge_open/core_study/review_imports
```

The importer rejects missing or duplicated item IDs, changed languages or
anonymous-card labels, invalid ratings, stale UI statuses, and completed items
outside the declared competency. It accepts a full export containing blanks
and emits a create-only receipt plus a reviewed-only descriptive summary.
`uncertain` is reported separately from `no`; the summary is explicitly a
descriptive result from a mixed random/targeted cohort, not a population claim.
