# Using freephdlabor for the open-ASR study

Inspected 22 September 2026 at commit [`9102d18a161037294d3d963b2799351aa724c58b`](https://github.com/ltjed/freephdlabor/tree/9102d18a161037294d3d963b2799351aa724c58b). Read-only source inspection; no installation, framework execution, credentials access or external LLM/cloud calls. A separate [runtime audit](../../publication/freephdlabor_runtime_audit_20260922.md) records the execution paths and cost controls checked.

## Recommendation

Adopt its explicit specialist handoffs, persisted state and evidence preparation. Keep our existing ASR execution and independent scoring machinery. Running another generic autonomous research stack now would duplicate orchestration while leaving the material blockers—decoder controls, reference quality and unavailable annotations—unresolved. This is an engineering assessment, not a claim that the framework cannot be useful.

The framework's source implements a manager, specialist agents, saved memory and resource organization before manuscript writing. Those are useful organizational capabilities. Its README's overnight-paper and peer-review claims do not establish that our specific ASR question is novel, validated or publishable. [Manager](https://github.com/ltjed/freephdlabor/blob/9102d18a161037294d3d963b2799351aa724c58b/freephdlabor/agents/manager_agent.py); [memory persistence](https://github.com/ltjed/freephdlabor/blob/9102d18a161037294d3d963b2799351aa724c58b/freephdlabor/agents/base_research_agent.py#L140-L201); [README](https://github.com/ltjed/freephdlabor/blob/9102d18a161037294d3d963b2799351aa724c58b/README.md).

## Concrete adaptation

| Component | Use in our study | Required change |
|---|---|---|
| Coordinator and persistent history | Track one active question, tried conditions, failures and next decision | Reference immutable ORX nodes and a claims/evidence ledger; avoid a new ideation cycle at every setback |
| Resource preparation | Give the reviewer/writer exact configs, predictions, source hashes and tables | Use the project-owned allowlisted packet prepared here; no generic whole-workspace scan |
| Experiment agent | Implement a prespecified decoder control | Replace upstream RunExperimentTool with a narrow ORX adapter that can launch only an approved committed experiment after its budget reservation |
| Reviewer | Find invalid comparisons and unsupported claims before writing | Review source artifacts and design first; accept negative results and explicit blockers |
| Writer and proofreader | Turn accepted evidence into venue-appropriate prose and figures | Use the actual venue template and evidence checks; leave missing experiments visible |

The upstream experiment tool invokes a nested AI-Scientist subprocess and defaults to four stages. It is not an adapter for our Modal/ORX pipeline. Replacing the tool alone is insufficient: the experiment prompt currently requires that tool, and the resource-preparation prompt expects AI-Scientist-specific result files. Both would need updating. [Experiment tool](https://github.com/ltjed/freephdlabor/blob/9102d18a161037294d3d963b2799351aa724c58b/freephdlabor/toolkits/run_experiment_tool.py#L120-L149), [subprocess invocation](https://github.com/ltjed/freephdlabor/blob/9102d18a161037294d3d963b2799351aa724c58b/freephdlabor/toolkits/run_experiment_tool.py#L261-L279), [experiment instructions](https://github.com/ltjed/freephdlabor/blob/9102d18a161037294d3d963b2799351aa724c58b/freephdlabor/prompts/experimentation_instructions.py), [resource instructions](https://github.com/ltjed/freephdlabor/blob/9102d18a161037294d3d963b2799351aa724c58b/freephdlabor/prompts/resource_preparation_instructions.py).

## Two policies to replace

The manager prompt steers repeated revision toward an AI reviewer score of at least six, while also stating iteration limits. Those are prompt instructions, not a scientific validity test or a monetary limit. Our stop condition should be a checked claim, a failed hypothesis or a concrete unresolved dependency—not an increasingly favorable generated review. [Manager instructions](https://github.com/ltjed/freephdlabor/blob/9102d18a161037294d3d963b2799351aa724c58b/freephdlabor/prompts/manager_instructions.py#L24-L148).

The LaTeX verification tool checks document structure, length, citations and figure counts, including a three-figure requirement. Those can help with formatting but cannot establish research soundness and need not fit our intended short paper. We should instead require traceable numeric claims, valid comparisons, correct citations and the chosen venue's real rules. [Verification implementation](https://github.com/ltjed/freephdlabor/blob/9102d18a161037294d3d963b2799351aa724c58b/freephdlabor/toolkits/writeup/latex_content_verification_tool.py#L340-L360).

## Cost and integration boundary

The default launcher uses provider-backed LLMs; those calls are separate from Modal compute. The inspected execution paths do not enforce our USD15 cap. The nested cost tracker returns zero when its model-price lookup fails, so its displayed estimate cannot serve as a spending authorization. [Launcher](https://github.com/ltjed/freephdlabor/blob/9102d18a161037294d3d963b2799351aa724c58b/launch_multiagent.py#L74-L122), [cost calculation](https://github.com/ltjed/freephdlabor/blob/9102d18a161037294d3d963b2799351aa724c58b/external_tools/run_experiment_tool/ai_scientist/utils/token_tracker.py#L107-L136).

A standalone deployment would need an explicitly funded LLM budget, model/provider verification, code-enforced limits covering nested calls and retries, and a budget-aware experiment adapter. These are not implemented by the files in this folder. The current Modal reservations remain untouched. Workflow ideas can be applied within the existing agent setup without launching the external stack.

## Prepared now

- [Task brief](TASK_BRIEF.md): one bounded ASR question, role outputs, evidence rules and termination conditions.
- [Review packet](review_packet.json): paths, byte counts and SHA-256 digests for 50 selected local files. Includes all seven run-artifact directories, scoring/validation code, data manifest, analyses and study scope.
- [Packet builder](build_review_packet.py): stdlib-only local indexing. It does not run agents, experiments, validation or networking, and it does not read credentials. Rebuilding reflects changed evidence; do not overwrite a packet already attached to a completed review.

The packet is our own handoff format, not a drop-in freephdlabor configuration. The useful next application is a fresh evidence review of the proposed Whisper decoder control. A reviewer should say what result would disprove the suspected decoder problem and which checks must pass before the study expands. Writing the full paper comes after that scientific decision.
