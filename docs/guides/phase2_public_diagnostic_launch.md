---
title: Launch a public diagnostic retained queue
kind: guide
status: current
owner: merlin-experiments
last_verified: 2026-10-19
related: [phase2_component_search, spike_timing]
code_refs:
  - packages/merlin-experiments/src/merlin_experiments/phase2/public_diagnostic_launch.py
  - packages/merlin-experiments/src/merlin_experiments/phase2/public_source_diagnostic.py
  - packages/merlin-experiments/src/merlin_experiments/phase2/public_native_exploration.py
  - packages/merlin-experiments/src/merlin_experiments/phase2/ordinary_fast_feedback.py
---

# Public diagnostic retained queue

`merlin-phase2-public-diagnostic` (or `python -m merlin_experiments.phase2.public_diagnostic_launch`)
executes one same-revision public source pair through the original retained queue.
This is an infrastructure diagnostic. It does not satisfy the formal Phase1 gate,
issue a performance cohort, accept an incumbent, or qualify timing.

The original operator supplies `--plan FILE --plan-sha256 SHA256 --output FRESH_DIRECTORY`.
The exact `merlin.public_diagnostic_launch.v1` plan selects public published source directories,
a real hash-only operator exclusion selection, a separate full planning inventory,
the immutable compiler inventory, an edit contract, ordinary contract source,
the target descriptor/source root/board selection, genuine compiler profile archive/workspace,
the exact selected shared library, and an original source-pinned admission factory.
Only executing the factory creates live diagnostic source owners; a saved successful
registry cannot issue them. The clock is an existing explicit infrastructure packet,
limited to 30 minutes and 600 CPU seconds with at least eight minutes for cleanup.
An original enforcing process supervisor must cover setup and the queue, using the
clock's CPU, RSS, file and aggregate storage limits; the plan itself is not hard enforcement.

The launcher copies the original compiler into separate baseline and candidate
packages, freezes original edit authority and a retained revision, and submits
`corpus-selected-compile-spike-feedback` outside its metadata CPU lease. Each arm
uses the original boxed compiler profile, complete source/native numerical checks
and a fresh original source/effect checker. Existing public declarations stay
immutable; the versioned host execution view requests L0–L3 while retaining L3 as
pending. Refusals and source-admission costs are retained in `status.json`.

The result is permanently `DIAGNOSTIC_ONLY`, with null measured cycles and UNKNOWN
timing. A one-member smoke roster does not replace the frozen workload inventory.
The Python `run_plan` API optionally exposes its actual live completed pool/job to
an original in-process consumer before cleanup; saved feedback does not recreate
that authority. An optional typed native exploration owner may be selected by an
original host when the corresponding policy API supports it; there is no JSON
native-qualification switch in this launcher.

An original Python host may instead select an actual edited compiler directory
with `run_plan(..., candidate_source=source)`. This is an API opt-in, not a new
field in the V1 JSON plan or a saved qualification. The launcher preserves the
original baseline and frozen edit authority, retains the initial control
revision as parent, checks the selected source inventory before and after
retention, and submits that issued revision through the same one-job pool.
`candidate-source-selection.json` records both revisions and whether their
source bytes differ. Edits outside the existing authority refuse; changing a
source hash alone establishes no emitted-program, correctness or performance
claim. The ordinary boxed numerical/source checks and permanent diagnostic
decision remain mandatory. The default still submits the unchanged control.

For native predictions, the original host calls
`prepare_public_native_exploration(native_selection)` with the existing exact
`NativeArchiveScreenSelection`. Preparation reopens public archive eligibility,
native held gates, hardware correspondence and independently reviewed event
semantics. Pass the returned live owner as `native_exploration=` to
`run_plan` or `prepare_public_source_diagnostic`. Saved qualification records
cannot select it. The default remains the original diagnostic route.

When compiler and native owners use distinct software identity namespaces, the
original host may supply `binding_correspondence=` to that preparation call.
This must be an original live reviewed `compiler_source_native_correspondence`
capability, bound to the complete native binding. Its original callback checks
the complete compiler binding against its own source selection and proofs.
This explicit choice uses native exploration record version 2; the default
version 1 direct identity comparison stays unchanged. Equalizing digest strings
or loading a successful correspondence report cannot select this route.
The explicit native owner retains an original bridge registry for one bounded
metadata traversal. Existing scoped verification checks the original owner before
and after that traversal while keeping getter identity/source/binding checks
active. No scope spans source execution, count observation or timing prediction;
a failed final owner recheck prevents the metadata result from returning.
One diagnostic policy selection record groups its native metadata getters in
that scope. Every actual compiler-binding callback and native runtime/semantics
check still executes; subsequent records enter new fresh scopes. The original
profile, corpus, baseline, source and numerical checks are unchanged.

The original host may explicitly select
`metadata_protocol="merlin.public_native_metadata.v1"` during preparation.
This privately issues one exact ordered registry of the selected Count,
native-semantics and bridge capabilities. One pure selection-record operation
then checks these original owners at entry and exit while retaining every
getter's object, implementation, source and full binding checks. Arbitrary
containing registries and copied records cannot select this protocol. Runtime
verification and the actual current compiler-binding callback still run on each
use. Execution, count observation and prediction stay outside the scope.
Omitting the protocol preserves the original V1 and bridge-only V2 behavior;
selecting it changes no V2 record semantics or timing qualification.

This explicit selection uses diagnostic feedback version 2. After each original
complete L0/L1/L2/source observation, it runs the existing boxed plain and annotated
count observer on that same ELF. Its original numerical receipt is unchanged.
Unsupported source geometry, capacity, layouts or event activity remain UNKNOWN;
no extra process is started for a rejected source domain. The first version
admits only the existing checked direct signed integer contraction source domain.

Per-member residual intervals provide exploration priorities. Overlapping
intervals leave ordering UNKNOWN. Unselected obligations remain visible, with no
complete performance-objective claim. Predictions never populate measured cycles,
roofline attainment, a measured incumbent or plateau evidence. Existing formal
search and final qualification retain their original gates. Count observation is
additional evaluation work inside the same parent deadline and resource envelope.

To inspect setup separately from execution, run:

```sh
merlin-phase2-public-diagnostic --check-plan --plan PLAN.json --plan-sha256 SHA256
```

This reuses the same plan/schema, declared metadata leaf-pin and existing clock
validation, prints selected public paths/identities and remaining packet time,
and returns `NOT_EXECUTED` with no authority. It creates no output directory and
never reconstructs or executes an admission factory, verifies a profile, freezes
a compiler, captures a source, or submits a queue job. Public source/reference/
private-exclusion checks, compiler inventories, original factory source closure,
actual resource admission, correctness and timing remain explicitly deferred.
The private exclusion selection's path/hash are omitted from the report. An
expired packet refuses; setup never creates or resets a clock. Omit
`--check-plan` and supply `--output FRESH_DIRECTORY` for the existing execution
route under its original enforcing supervisor.

Setup probes may start from an incomplete or provisionally qualified Phase 1
compiler. Preserve its package bytes and original qualification status. The
diagnostic launcher copies those bytes into separate arms; it does not upgrade
the Phase 1 result or require a new compiler-generation run merely to test the
infrastructure. A formal optimization launch still needs its original admission
gates and an independently admitted performance cohort.

Use the following probe levels to distinguish setup from execution:

| Probe | Executes target work? | What it establishes |
| --- | --- | --- |
| `--check-plan` | No | Metadata/schema, selected leaf pins and an existing valid packet clock; all deferred gates are listed. |
| Original source/profile/capability and native-selection preflight | No target run; it may replay retained public calibration evidence | Current original owners can reopen their exact inputs, contract correspondence and covered timing gates. |
| Same-revision public diagnostic queue | Yes: bounded compilation, complete-output execution and optionally fresh counts | End-to-end queue, sandbox, resources, copied compiler, library selection and feedback wiring for those selected cases. No author or optimization search. |
| Formal Phase 2 launch | Yes | Runs the admitted search policy and objective under their original gates. |

A receipt inventory can check saved evidence hashes and report missing stages,
but it cannot reconstruct live owners or prove the next execution will pass.
Likewise, no metadata-only probe can establish target execution, cleanup under
failure or numerical correctness. Record PASS, REFUSED, UNKNOWN and NOT RUN per
stage; never convert the diagnostic smoke roster into the performance cohort.

For source capture and independent reference execution, select both environments
explicitly. `MERLIN_PHASE0_SEALED_CAPTURE` selects the original sealed capture
configuration; `MERLIN_M2M_PYTHON` selects the existing reference-interpreter
interface. Selecting one does not select the other. Keep the virtual-environment
Python path rather than resolving its executable symlink to the base interpreter.
An unavailable historical runtime stays unavailable; a fresh runtime needs its
own capture and replay evidence.

An original Python host may supply `external_boundary=` to `run_plan`. This
requires the existing exact live `ExternalResourceBoundary`, verifies its CPU,
RAM, storage and output bindings before setup/output creation, scopes setup and
execution with that owner, and passes the identical owner to the queue with
`require_hard_limits=True`. JSON or a reconstructed boundary cannot select it.
The default route has no new hard-mode requirement and makes no hard-limit claim.
This API verifies already provisioned controls; it creates no cgroup, filesystem,
quota or supervisor. CPU bandwidth verification does not prove a cumulative
600-second CPU budget, and the original enforcing operator still covers the
whole call's CPU-time and wall limits. Successful test routing with neutral
kernel observations is not evidence of actual kernel enforcement.
