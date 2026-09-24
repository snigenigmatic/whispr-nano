# Remaining tasks for C. Kaustubh

Updated 24 September 2026. The planned compute batch is complete: 20 accepted quality comparisons, six resource profiles, 13,232 predictions over 2,840 unique recordings. The manuscript is a working draft, not a submitted or accepted paper. No additional GPU purchase is required to complete the current handoff.

Dates below are proposed internal targets except the explicitly cited venue dates and your 16 October review. You can delegate recruiting and administration to teammates; language judgments must come from competent human reviewers.

## 1. Clarify the grading requirement — first priority, today

- [ ] Ask Prof. Bhaskarjyoti Das, in writing, whether the capstone requires **submission, acceptance, or proceedings publication**, and the exact date by which that evidence must exist.
- [ ] Confirm whether an archival workshop paper counts and what the 16 October review should demonstrate if a decision is still pending.
- [ ] Agree on a fallback with the guide if acceptance cannot arrive before the grading cutoff. A venue choice cannot guarantee acceptance.

Suggested message to send yourself:

> Sir, our planned ASR comparison and resource measurements are complete, and we are finishing human error review and the manuscript. BHASHA's submission deadline is 15 October, but its acceptance notification is 10 November, after our 16 October review. Could you confirm whether submission evidence is sufficient at that review, whether an archival workshop paper counts, and the final date by which acceptance or publication is required? We would also like to agree on the fallback if the decision is pending at grading time.

The [official BHASHA dates](https://bhasha-workshop.github.io/) were checked on 24 September: submission **15 October 2026 AoE**, notification **10 November**, camera-ready **25 November**, workshop **20 December**. An on-time submission cannot provide a BHASHA acceptance decision by 16 October.

## 2. Arrange human reviewers — aim for 26 September

- [ ] Find reviewers competent in **Hindi (18 clips), Bengali (16), Assamese (16), and Indian English (12)**. A person may cover multiple languages only if they can actually judge them.
- [ ] Arrange **10 independently double-reviewed FLEURS clips**, distributed across the three languages and assigned to competent second reviewers. Keep the two initial responses independent before discussing disagreements.
- [ ] Confirm reviewer availability and agree on a return date. Suggested first-pass target: **30 September**; overlap and adjudication by **2 October**.
- [ ] Give reviewers only their blinded packet and working audio access. Do not direct them to this repository's model results, manuscript, or identity mappings before review. Check one audio example on each reviewer's machine before assigning the full batch.

The FLEURS packet has **50 clips / 168 transcript cards**. The separate English resource packet has **12 clips / 72 cards**. These are bounded review tasks, not a request to label thousands of recordings. The English clips overlap the resource experiment and add no independent test recordings.

Instructions: [FLEURS handoff](research/bridge_open/core_study/REVIEW_HANDOFF.md) and [English handoff](research/bridge_open/listening_review_core_en_v1/REVIEW_HANDOFF.md). The GitHub HTML preview does not provide a working listening session by itself; the packet must be served with its verified local audio. An isolated distribution bundle or supervised local session can be prepared before delivery.

## 3. Return the actual review exports

- [ ] Collect each reviewer's downloaded JSON export, pseudonymous reviewer ID, declared language competency, and any uncertainty notes. Return them for import; no manual spreadsheet reformatting is needed.
- [ ] Ask the competent reviewers to adjudicate disagreements after the independent responses are retained. Leave genuinely unclear cases marked uncertain or unjudgeable.
- [ ] Record reviewer contributions for acknowledgment or authorship discussion with the guide. Avoid collecting unnecessary personal information.

You do not need to compute WER, agreement statistics, or merge JSON. The technical work is listed below. Automated script detection and LLM-generated opinions cannot replace these listening judgments.

## 4. Confirm authorship and obtain a scientific go/no-go

- [ ] Confirm author names, order, affiliations, institutional email addresses, contributions, corresponding author and guide's authorship role with the team. The draft currently says “C. Kaustubh and collaborators.”
- [ ] Have the guide and, if available, an ASR researcher read the completed results and contribution statement. Ask specifically whether this bounded audit provides a useful finding beyond another leaderboard.
- [ ] Agree on the target venue and short/long paper category by **5 October**, after interpreting the human review. Do not claim a new ASR method, demographic fairness, phone performance or reproduction of private BRIDGE data.

The current candidate contribution is an auditable comparison of inference recipes, failure behavior and measured resource use. Its novelty and usefulness remain a real scientific decision, not something the completed run count proves.

BHASHA is the current tentative target. It accepts four-page short and eight-page long papers using the ACL-style template; accepted papers must be registered and presented for inclusion in the ACL Anthology. Verify scope with the organizers if the guide is uncertain. See the [submission instructions](https://bhasha-workshop.github.io/submission.html).

If BHASHA is unsuitable or the work is not ready, the previously considered backup is the **archival research-paper track of EACL 2027 SRW**: direct submission **15 December 2026**, notification **5 January 2027**. It permits undergraduate first authors and preliminary research results. This does not solve an earlier grading cutoff. Check the [official call](https://2027.eacl.org/calls/srw/) and use venues sequentially, not simultaneous review of the same paper.

## 5. Approve the final paper and capstone materials — aim for 9 October

- [ ] Read the revised manuscript after review results are incorporated; confirm that every author agrees with the claims, limitations and submitted version.
- [ ] Give the exact university review rubric, slide template, required project title/ID, and report format if the existing versions are outdated.
- [ ] Check whether the university form due **23 September** was already submitted. If needed, arrange a correction with the guide: use the current audit scope and “To be submitted”; do not describe the earlier numeric-correction proposal as completed work.
- [ ] Review and rehearse the slides and offline demonstration for **16 October**. Be able to explain the poor decoder results, partial Svarah coverage, and why GPU measurements do not demonstrate phone performance.

The current PDF is a six-page working article, not a final venue-formatted submission. Formatting, figure/table preparation, and drafting can be handled technically after the scope and reviewer findings are settled.

## 6. Complete the actual submission and preserve the receipt

- [ ] Confirm the final author list and all coauthors' permission to submit.
- [ ] Use the selected venue's submission account and complete its author declarations. Check paper category, anonymity, page limits and required files against the live portal.
- [ ] Submit only after the guide's go/no-go, ideally by **14 October** internally if targeting BHASHA's **15 October AoE** deadline. Save the submission ID, receipt, submitted PDF and timestamp.
- [ ] Update the guide and university records to **Submitted** only after the portal confirms submission. Acceptance, presentation and publication are later, separate statuses.
- [ ] If accepted, arrange registration, camera-ready corrections and an author to present. Confirm attendance costs and options with the guide when relevant.

## 7. Close the administrative loop

- [ ] Check actual Hugging Face and Modal billing dashboards for the completed runs, and retain the statements if reimbursement or a project spending record is required. The repository records reservations and runtime, not verified provider charges.
- [ ] Keep access to the original datasets and local evidence archives. Do not delete them after seeing the GitHub copy: the public branch intentionally excludes original data-bearing execution archives and sealed review mappings.

## Technical work that can be handled for you

These remain project tasks, but do not require you to implement or research them personally:

- Prepare isolated reviewer bundles with working audio from the existing local assets, once reviewer languages and access method are known.
- Validate and import actual reviewer exports; calculate descriptive findings and agreement on the overlap; preserve adjudication separately.
- Update the claims and manuscript based on those findings; format for the agreed venue and check all tables, citations and limitations.
- Prepare the final capstone deck, status slide, report and offline demonstration from the accepted evidence and university rubric.
- Prepare the submission file bundle and any later camera-ready revisions. The author declarations, coauthor approval, scientific go/no-go and university grading agreement remain human responsibilities.

No further model expansion, training run, or opening of the reserved SPIRE test set is required by this checklist. A future experiment should have a separately agreed question and protocol.
