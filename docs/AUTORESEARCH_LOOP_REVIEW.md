# AutoResearch loop review for the ASR mechanism study

Reviewed 9 September 2026. Source: [EvoMap/AutoResearch at commit
6b85443f9e3d33d357c86c8bb826f9ea03c780b3](https://github.com/EvoMap/AutoResearch/tree/6b85443f9e3d33d357c86c8bb826f9ea03c780b3)
(HEAD at inspection; commit dated 1 September 2026). The repository was cloned
to `/tmp/evomap-autoresearch-review-20260909` for static inspection. No upstream
code, installation script, model request, or test suite was executed. This is a
code-grounded assessment, not a claim that its advertised workflow has been
independently reproduced.

## Decision

Adopt its discipline around persistent work units, execution evidence,
falsification and independent criticism. Keep our existing Modal and ASR code.
The additional component we need now is a small deterministic evidence and
promotion controller, not another general-purpose autonomous researcher.

AutoResearch has considerably more real orchestration machinery than a sequence
of persuasive prompts. But its machine-enforced completion criteria primarily
establish that workflow steps produced fresh artifacts and reviews. They do
not establish that an adaptive experiment has discovered a generalizable
scientific effect. These are different questions. Its own closure can mean
that further iteration is not justified, even when a review is negative.

## Features actually implemented, and how we should use them

| Feature | Evidence in the inspected code | Adaptation for this project |
|---|---|---|
| Persistent, resumable work units | The workflow engine stores a queue, dependency states, atomic writes, and timestamped decisions. `queue_lock`, `lease_active`, and `reclaim_expired_leases` serialize claims and recover expired workers; repeated reclamation blocks a poisoned unit. | Give every teacher shard and arm/seed training run an immutable ID. Resuming a controller should inspect receipts and active Modal jobs before launching anything. A process interruption is an execution event, not a new scientific trial. |
| Execution receipts that bind actual work | `command_execute_run` checks the requested stage and output namespace, captures a new attempt log, records argv/environment/exit status, and verifies that the old shared-log prefix was not rewritten. `run_receipt_evidence` verifies all recorded output hashes and rejects omitted files. | Bind each run to data/model/code/config hashes and actual counts. Store every attempt and failed run. A success marker or fluent summary cannot substitute for predictions, gradients/exposure counts, and the fixed evaluator's output. |
| Separate execution failure from hypothesis failure | The runner prompt sends a clean run that misses its metric to `not_met`, instead of endlessly treating it as a coding bug. Debugging is bounded. | Do not change preprocessing, decoding, labels or metrics to make the mechanism look positive. Runtime repairs preserve the experiment specification; scientific changes create a new registered trial. |
| Pilot before scaling, with falsification | The planner prompt requires primary and auxiliary criteria, concrete falsifiers, revision history, and pilot/main phases. The engine inserts criticism and later iterations as explicit units. | Finish the controlled selection pilot before adding conditional KD or a new on-policy objective. Give the selection mechanism an explicit stop route if it fails against the matched random control. |
| Fresh, producer-bound criticism | Critic outputs are bound to unit/cycle, artifact hashes, role/model identities and a producer receipt; the engine checks freshness. The external critic requires two successful distinct model identities. | A methods review must state exactly which immutable evidence bundle it reviewed. An old review cannot approve new code or a new dataset. Start with the independent reviewers available in this session; do not claim cross-model independence if they share one underlying model. |
| A separate review without self-promotion | The blind-review prompt receives a paper-style submission without internal scores or coordinator context. The packaging instructions require numerical results, including negative results and all seeds. | Generate a neutral evidence packet after the pilot: hypothesis, intervention, full arm/seed table, exclusions, limitations, code/data hashes. Reviewers should judge the actual effect and alternative explanations, without being told that it is publishable. |

Primary sources for this table: [atomic writes, decisions and event hashes](https://github.com/EvoMap/AutoResearch/blob/6b85443f9e3d33d357c86c8bb826f9ea03c780b3/ar-runtime/scripts/ar-workflow-engine.py#L209-L316),
[queue locks and leases](https://github.com/EvoMap/AutoResearch/blob/6b85443f9e3d33d357c86c8bb826f9ea03c780b3/ar-runtime/scripts/ar-workflow-engine.py#L594-L669),
[run execution](https://github.com/EvoMap/AutoResearch/blob/6b85443f9e3d33d357c86c8bb826f9ea03c780b3/ar-runtime/scripts/ar-workflow-engine.py#L1266-L1478),
[artifact validation](https://github.com/EvoMap/AutoResearch/blob/6b85443f9e3d33d357c86c8bb826f9ea03c780b3/ar-runtime/scripts/ar-workflow-engine.py#L1505-L1617),
[runner failure handling](https://github.com/EvoMap/AutoResearch/blob/6b85443f9e3d33d357c86c8bb826f9ea03c780b3/ar-runtime/.claude/agents/ar-runner.md#L98-L159),
[planner protocol](https://github.com/EvoMap/AutoResearch/blob/6b85443f9e3d33d357c86c8bb826f9ea03c780b3/ar-runtime/.claude/agents/ar-planner.md#L37-L79),
[critic producer contract](https://github.com/EvoMap/AutoResearch/blob/6b85443f9e3d33d357c86c8bb826f9ea03c780b3/ar-runtime/scripts/ar-critic-contract.ts#L26-L136),
[actual critic invocation](https://github.com/EvoMap/AutoResearch/blob/6b85443f9e3d33d357c86c8bb826f9ea03c780b3/ar-runtime/scripts/ar-external-critic-mcp.ts#L161-L210), and
[blind-review packaging](https://github.com/EvoMap/AutoResearch/blob/6b85443f9e3d33d357c86c8bb826f9ea03c780b3/ar-runtime/.claude/agents/ar-blind-reviewer.md#L33-L63).

The upstream tests explicitly encode stale-report, missing-artifact, forged
queue, wrong-stage and skipped-review failures. That is useful evidence of
intended invariants; these tests were inspected, not run. See
[execution-receipt tests](https://github.com/EvoMap/AutoResearch/blob/6b85443f9e3d33d357c86c8bb826f9ea03c780b3/ar-runtime/scripts/tests/test_engine_terminal_receipts.py),
[blind-review convergence tests](https://github.com/EvoMap/AutoResearch/blob/6b85443f9e3d33d357c86c8bb826f9ea03c780b3/ar-runtime/scripts/tests/test_blind_review_convergence.py), and
[skip/critic binding tests](https://github.com/EvoMap/AutoResearch/blob/6b85443f9e3d33d357c86c8bb826f9ea03c780b3/ar-runtime/scripts/tests/test_skip_bypass_and_critic_binding.py).

## Failure modes we should not import

### Adaptive validation overfitting is not addressed by workflow completion

The planner is explicitly allowed to revise the dataset or success metric after
an analysis suggests it. Those changes are logged, which is better than hiding
them, but a revision log does not restore the independence of already inspected
evaluation data. The inspected engine has no explicit held-out-query ledger,
sealed test capability, accounting for the number of adaptive comparisons, or
fixed reference-evaluator identity. Its code-review prompt mentions leakage,
but detecting it depends on the supplied context and an LLM judgment. This is a
bounded finding about the inspected implementation, not a proof that every
possible user-written experiment leaks. [Planner revisions](https://github.com/EvoMap/AutoResearch/blob/6b85443f9e3d33d357c86c8bb826f9ea03c780b3/ar-runtime/.claude/agents/ar-planner.md#L52-L79),
[review prompt](https://github.com/EvoMap/AutoResearch/blob/6b85443f9e3d33d357c86c8bb826f9ea03c780b3/ar-runtime/scripts/ar-gemini-review-mcp.ts#L11-L54).

**Our response:** every dev evaluation and checkpoint comparison is logged;
selection remains explicitly exploratory. Lock the primary metric, normalizer,
contrasts, seed schedule and independent confirmation protocol before opening
confirmation outcomes. The current L2-ARCTIC test and test-shared rows remain
excluded from pilot inference. Having repeatedly queried development speakers
does not make a later development confidence interval confirmatory.

### Reviewer ratings can become the wrong optimization target

The engine starts another revision when the mean blind-review rating is below
5.5, weaknesses are present, and its bounded revision/cycle allowance remains.
The proposed focus is then the reviewers' weaknesses. That can improve a real
experiment, but can also reward changes in how the evidence is narrated.
Nothing in that branch requires better independent-speaker ASR performance.
A fresh reviewer context removes conversational memory; it does not remove
common model biases or adaptivity from repeatedly revising for the same panel.
[Revision branch and limits](https://github.com/EvoMap/AutoResearch/blob/6b85443f9e3d33d357c86c8bb826f9ea03c780b3/ar-runtime/scripts/ar-workflow-engine.py#L2772-L2821),
[reviewer prompt and invocation](https://github.com/EvoMap/AutoResearch/blob/6b85443f9e3d33d357c86c8bb826f9ea03c780b3/ar-runtime/scripts/ar-external-critic-mcp.ts#L213-L270).

**Our response:** use reviews to identify a falsifiable missing comparison or a
validity defect. Their scores never promote a model, relax an error bound, or
trigger publication claims. A presentation improvement and an empirical
improvement get separate records. A good review cannot override a failed
scientific gate.

### Hashes authenticate artifacts, not what the metric means

The engine verifies that a run happened with the expected stage and that its
artifacts match their hashes. `run_receipt_evidence` does not recompute the
reported metric against a frozen evaluator. A correctly executed script can
still score the wrong reference field, omit difficult clips, select the best
seed, or compare unequal training exposure. The code reviewer is instructed to
trace hard constraints, but `validateConstraintAudit` checks the structure of
supplied audit entries; it does not independently derive the complete expected
constraint set from a machine-readable experiment specification.
[Run evidence checks](https://github.com/EvoMap/AutoResearch/blob/6b85443f9e3d33d357c86c8bb826f9ea03c780b3/ar-runtime/scripts/ar-workflow-engine.py#L1505-L1617),
[constraint-audit implementation](https://github.com/EvoMap/AutoResearch/blob/6b85443f9e3d33d357c86c8bb826f9ea03c780b3/ar-runtime/scripts/ar-review-contract.ts#L22-L74).

**Our response:** a separate deterministic evaluator recomputes WER from raw
prediction rows, rejects unknown/missing/duplicate utterances, checks all arms
used the same dev IDs, and verifies fixed decoder/normalizer and initialization
identities. Register every required arm and seed before execution; absent or
failed required runs prevent an overall scientific conclusion. Store both
unique audio and repeated exposures so fixed-update and equal-exposure results
cannot be conflated.

### Closure is not an acceptance decision

After the permitted review revision is exhausted, the engine can append close
even for a low score, or explicitly record unavailable review. `verify-close`
checks completion evidence, not journal acceptance. Consequently an
`AUTORESEARCH_DONE` marker cannot mean that a paper is publishable. This is
particularly relevant to our earlier mistake of equating a finished PDF with
a research contribution. [Closure after review](https://github.com/EvoMap/AutoResearch/blob/6b85443f9e3d33d357c86c8bb826f9ea03c780b3/ar-runtime/scripts/ar-workflow-engine.py#L2809-L2837),
[closure verifier](https://github.com/EvoMap/AutoResearch/blob/6b85443f9e3d33d357c86c8bb826f9ea03c780b3/ar-runtime/scripts/ar-workflow-engine.py#L3655-L3687).

**Our response:** track separate outcomes: `execution_complete`,
`valid_experiment`, `hypothesis_not_supported`, `needs_replication`, and
`confirmation_complete`. A later manuscript decision remains a distinct
research judgment with novelty and external-validity requirements.

### Static implementation discrepancies warrant selective adoption

The current README describes an optional secondary critic. The external-critic
execution function nevertheless requires exactly two successful distinct
critics after the general helper allows an explicitly skipped secondary. Blind
review can render one configured critic's output, but the workflow engine still
requires two reviews before treating its panel as usable. These paths mean
README-level configuration assumptions do not fully describe execution.
[External-critic requirement](https://github.com/EvoMap/AutoResearch/blob/6b85443f9e3d33d357c86c8bb826f9ea03c780b3/ar-runtime/scripts/ar-external-critic-mcp.ts#L169-L191),
[optional helper](https://github.com/EvoMap/AutoResearch/blob/6b85443f9e3d33d357c86c8bb826f9ea03c780b3/ar-runtime/scripts/ar-critic-contract.ts#L166-L185),
[engine minimum](https://github.com/EvoMap/AutoResearch/blob/6b85443f9e3d33d357c86c8bb826f9ea03c780b3/ar-runtime/scripts/ar-workflow-engine.py#L52-L59).

The blind-review renderer uses `parseFloat` and a finiteness filter; it does not
check that a rating is an integer between 1 and 10. It derives its displayed
accept/borderline/reject decision largely from the average. A malformed numeric
rating can therefore enter that average despite the prompt's stated scale.
This is static control-flow reasoning; it was not exercised against live
models. [Rating parser and decision logic](https://github.com/EvoMap/AutoResearch/blob/6b85443f9e3d33d357c86c8bb826f9ea03c780b3/ar-runtime/scripts/ar-critic-contract.ts#L202-L233).

Its event hash chain and read-only file modes are valuable accidental-corruption
checks. Because the agent and controller share a writable host, they should
not be presented as a security boundary against a fully adversarial same-user
process. Its own launch documentation calls for an isolated disposable
environment. We need reliable accounting here, not claims of tamper-proof
scientific truth. [Artifact permissions](https://github.com/EvoMap/AutoResearch/blob/6b85443f9e3d33d357c86c8bb826f9ea03c780b3/ar-runtime/scripts/ar-workflow-engine.py#L1311-L1344),
[launch guidance](https://github.com/EvoMap/AutoResearch/blob/6b85443f9e3d33d357c86c8bb826f9ea03c780b3/README.md).

## Bounded implementation we can add now

Add a controller around the existing `research/pilot/training.py`,
`selection.py`, and `modal_runner.py`. Do not transplant the upstream runtime or
add another paid reviewer dependency merely to complete a workflow diagram.

1. **Freeze one pilot contract.** Record the hypothesis, permitted train/dev
   IDs, sealed-test identity, teacher/student revisions, evaluator/normalizer
   hashes, intervention arms, paired seed schedule, decoder parameters and
   exposure policy. The primary mechanism contrast is teacher-filtered inputs
   versus matched-random inputs using the same verified-reference targets.
   Additional target-source arms answer a different question and are named as
   such. Record pretraining overlap and unadjudicated-reference limitations.
2. **Register runs before dispatch.** Write immutable run ID, contract hash,
   arm, seed, code/config hashes, intended input IDs, expected dev IDs and
   planned resource ceiling. Failed/retried executions remain visible. Reusing
   cached teacher predictions requires matching model, data and decoding
   identities, not only a filename.
3. **Ingest evidence deterministically.** Accept a receipt only when predictions
   cover the exact expected dev IDs, metrics are finite and recomputed from
   those rows, the initialization and evaluator match the contract, required
   training counts and exposure statistics exist, and no sealed-test row was
   evaluated. Append an event with the result hash and whether that result was
   shown to the research agent. A duplicate receipt is idempotent; a conflicting
   receipt for the same run ID is rejected.
4. **Compute the next permissible stage.** Missing/inconsistent evidence means
   `invalid_or_incomplete`. A runtime-correct but unresolved or negative pilot
   means `hypothesis_not_supported` or `inconclusive`, not a code-repair loop.
   A promising pilot means `needs_replication`, not publishable. Require the
   decisive matched contrast across the prespecified seeds, with every L1's
   absolute change visible, before adding complex interventions. A mechanism
   that disappears after amount/composition/exposure controls is a stop signal.
5. **Create a neutral evidence packet and critique once.** Include all runs,
   failures, exclusions, exact contrasts, seed variation, speaker support,
   normalization sensitivity, and alternative explanations. The critic must
   cite a row or artifact for each claim. Save criticisms and proposed next
   experiments; do not use an LLM numerical rating as an optimization objective.

An ordinary Python module plus `contract.json`, `runs.jsonl`,
`evaluation_access.jsonl`, and `decision.json` is sufficient for this stage.
Use atomic updates and a file lock; an append-only hash chain is optional until
multiple controllers make it useful. Three meaningful tests should precede
wiring it into dispatch: changed evaluator/data identity is rejected; an omitted
required arm/seed cannot promote; any test/test-shared evaluation is rejected.
Add idempotent-resume coverage if the controller is allowed to launch jobs.

The next research milestone remains a controlled training result. The controller
exists to make that result trustworthy and make a negative result stop the
wrong idea. It is not a replacement contribution for the ASR paper.
