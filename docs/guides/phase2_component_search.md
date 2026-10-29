---
title: Launch a bounded component compiler search
kind: guide
status: current
owner: merlin-experiments
last_verified: 2026-10-12
related: [component_phase2_workflow, phase2_cpu_timing_study, phase2_test_justification]
code_refs:
  - packages/merlin-experiments/src/merlin_experiments/phase2/search_configuration.py
  - packages/merlin-experiments/src/merlin_experiments/phase2/search_coordinator.py
  - packages/merlin-experiments/src/merlin_experiments/phase2/evaluation_pool.py
  - packages/merlin-experiments/src/merlin_experiments/phase2/retained_revisions.py
  - packages/merlin-experiments/src/merlin_experiments/phase2/component_launch_inputs.py
  - packages/merlin-experiments/src/merlin_experiments/phase2/component_stage.py
  - packages/merlin-experiments/src/merlin_experiments/phase2/component_workflow.py
  - packages/merlin-experiments/src/merlin_experiments/phase0/operator_exclusion.py
---

# Launch a bounded component compiler search

This guide covers the separate optional component workflow. The normal Phase 2
experiment reuses Phase 1's corpus and neutral runtime; follow the
[ordinary search guide](phase2_corpus_search.md). Its launch does not
require the component qualification ledger below.

This route improves one fresh compiler against an independently generated public
development corpus. The search objective is paired measured simulator cycles over
the complete selected component roster. Complete host and model costs remain
UNKNOWN unless separately measured by their original owners. Private validation
programs and reference implementations are outside author inputs and tuning.

## Qualify the component runtime

The prepared component runtime refuses a complete stage witness when its
required feature, CCA or RTL measurement producers are absent. Select and qualify
those producers independently, together with compatible public declarations and
an ordinary source generation run. Binaries or saved receipts cannot replace
missing producers. Resolve these requirements before using the launch example
below; configuration and routing checks establish no execution or timing
qualification.

## Required operator inputs

The original experiment controller must retain these live owners:

- The exact fresh Phase 1 compiler origin and its observed authoring provenance.
- Its admitted functional baseline, edit authority and immutable public view.
- The independent functional and measurement runtime, including its original
  grading, semantic, feature, CCA and RTL executor roles.
- The independently qualified analytical selection and exact development timing
  certificate, facts and engine source membership.
- A frozen public performance corpus and its generation manifest.
- Explicit author transport, public tool grants, accounting inputs and assigned
  worker resources.

`load_component_launch_inputs(configuration, phase1_origin=origin,
measurement_support=runtime)` reopens these original owners. Closed v2 launch
data additionally selects `rtl` certificate/facts pins and a fresh private work
root. It supplies no callback, compiler seed or new qualification. Saved PASS
JSON cannot reconstruct the owners. The standalone CLI refuses absent live
owners; a trusted in-process controller can also pass them explicitly to
`authoring_cli.main`. Do not substitute reference support to satisfy a refusal.

Absent physical runtime, timing, screening or source membership is a startup
prerequisite failure. Configuration parsing and synthetic routing tests cannot
establish that an actual target has satisfied it.

## Generate and seal public inputs

Use the ordinary Phase 0 source generation and performance-corpus freeze APIs.
Every member retains its source rationale, independent reference outputs and
family membership. Include typed resource-boundary and connected source cases,
including real movement, cast and reduction semantics. Source preparation does
not establish physical placement, overlap, correctness or timing qualification.

If an operator maintains a private exclusion inventory, select its private
mode-0600 file through `phase0.operator_exclusion.select(path, exact_sha256)`.
Pass that same opaque selection explicitly to generation and sealing through
`operator_exclusion=`. The guard compares emitted source bytes and every typed
contraction's ordered shapes, dtypes and selected numerical semantics. It is an
exact-signature exclusion, not a shape-only or semantic-equivalence proof.
Unknown or matched members abort publication without a named match diagnostic.
Do not put the file, its hashes or match details in public recipes or grants.
Omitting the selection performs no private exclusion check.

## Prepare the search configuration on the assigned worker

After constructing launch inputs from the live owners, use
`prepare_search_configuration`. It derives all corpus, baseline and certificate
identities from that policy; the operator provides budgets and stopping choices.
It writes a fresh closed configuration, checks CPU assignment against actual
affinity and reopens the policy before returning. It does not allocate AWS
resources or issue an enforcing envelope.

```python
from merlin_experiments.phase2 import component_launch as launch
from merlin_experiments.phase2.evaluation_pool import AssignedResources, JobResources
from merlin_experiments.phase2.search_configuration import prepare_search_configuration
from merlin_experiments.phase2.search_coordinator import SearchSettings

# inputs comes from the original controller's live owner handoff above.
qualified = launch.qualify_component_launch(
    inputs, model=selected_model, effort=selected_effort,
    max_concurrent_subagents=3,
)
selection = prepare_search_configuration(
    inputs.policy, inputs.stage_root / "search-configuration.json",
    resources=AssignedResources(
        cpu_ids=assigned_cpu_ids, ram_bytes=assigned_ram_bytes,
        disk_bytes=assigned_disk_bytes, engine_slots=3,
        author_cores=4, reserve_fraction=0.2, require_hard_limits=False,
    ),
    job_resources=JobResources(
        cores=1, ram_bytes=job_ram_bytes, disk_bytes=job_disk_bytes,
        engine_slots=1, files=100_000,
    ),
    settings=SearchSettings(
        beam_width=3, soft_seconds=14_400, hard_seconds=28_800,
        final_reserve_seconds=2_700, minimum_improvement=0.01,
        plateau_batches=3,
    ),
    source_bytes=256 << 20, source_files=20_000,
    retained_source_bytes=2 << 30, max_pending=6, max_jobs=96,
    author_agents=3,
)
record = launch.run_component_stage(
    qualified, model=selected_model, effort=selected_effort,
    wall_budget_seconds=14_400, max_tool_calls=1_000,
    tool_timeout_seconds=600, suite=accounting_suite,
    search_configuration=selection.path,
)
```

The variables describing allocated CPUs and RAM/disk budgets are required
operator selections, not inferred spare capacity on a shared host. Fit the job
grants to that allocation. The queue reserves author capacity and the selected
fraction, admits at most three concurrent roster jobs, and enforces pending-job,
job-count, source-byte/file and retained-feedback bounds. Each worker uses one
member at a time. This version does not claim to fan out member simulations over
20–25 cores; assigning more CPUs does not create that parallelism.

CPU affinity, per-process address-space and per-file size limits are applied;
disk usage is sampled. These do not establish aggregate descendant RAM, disk
quota or an interruptible OS deadline. `require_hard_limits=True` refuses without
a supported enforcing envelope. An AWS operator requiring aggregate hard bounds
must provide and verify the experiment's actual cgroup/container and output
quota. The pool's configuration or lifecycle receipts do not prove those limits.

## Author and evaluation lifecycle

The coordinator alone merges edits and operates the broker. Up to three native
authors use disjoint edit ownership. Freeze a revision with a stated hypothesis
and edit family, then submit analytical or RTL evaluation. Authors may continue
editing the live compiler; queued workers use the exact immutable revision and
shared read-only inputs. Options do not create another source-tree copy.

`search-status` reports conservative paired measurement ranges, the incumbent,
the beam and a stop reason. Ranking requires the exact complete correct measured
roster and original certificate. Equal family/member weights come from the
public roster, not a validation model profile. Analytical results cannot select
a measured winner. Later repeats can revoke an earlier improvement; partial
results cannot count as successful trials or supply selection feedback.

Stop on the declared exploration deadline, job budget or three completed diverse
beam batches without the selected minimum improvement. These are budget and
plateau decisions, not proof of global optimality. Before leaving authoring:

1. Join every author and stop edits.
2. Call `search-finalize` to stop submissions and finish or cancel/reap jobs.
3. Call `search-select` to restore the current completely measured incumbent.
4. Make no further edits and let the original final controller verify it.

With `continue_author_session=True`, return a completed native turn after each
hypothesis, submission or feedback checkpoint. The driver resumes the same thread;
an intermediate turn return does not finalize the search. The canonical prompt
and `STAGE_CONTEXT.json` publish fixed exploration, author-return and campaign
deadlines, including setup already spent. Finalize at the target or exploration
boundary and complete required feedback before the author-return deadline.
`queued_job_timeout_seconds` is a maximum lifetime: admission clips it to the
remaining exploration window after validation. Queue waiting and dispatch retain
the same absolute completion boundary and original cleanup checks.

Measure the unchanged starting compiler once if baseline fallback may be needed.
A fallback with no complete measurement cannot be selected. Selection takes
place before observed author lineage freezes. Its broker receipt binds the
restored source to original feedback. A later edit, unfinished queue or missing
selection refuses final admission. Cleanup UNKNOWN retains ownership and grants;
do not launch replacement jobs until the original handle reaches a verified
terminal state. Durable filesystem operations may delay status and cleanup.

## Fidelity, reuse and final acceptance

Use compilation/static observations and ordinary numerical execution for the
facts they establish. Spike is numerics/ISA evidence, not cycle timing. Qualified
analytical intervals screen proposals; selected CPU RTL runs establish timing
only inside the original certified envelope. Run the timing-cost study in AWS
using its explicit private controller and allocated-instance checks; the study
CLI itself writes a plan and executes no tools. Retain setup, calibration, held
validation, cold/warm and final-check costs in the denominator.

Re-emission must precede any execution reuse. An LLVM-text cache or a previous
verdict does not prove an unchanged full executable. Absent a qualified original
reuse owner, execute every case. Cached historical observations are not new
independent replicates. Final and protected evaluation retain the original
complete execution roster regardless of development reuse.

The final controller runs fresh complete compile/static and numerical gates on
the observed selected compiler. `NO_FINAL_ACCEPTANCE` feedback and a consumable
authoring record do not establish held-out performance or convergence. Preserve
the frozen compiler and its identities before the separately authorized final
validation run. No FPGA measurement belongs in the authoring loop.
