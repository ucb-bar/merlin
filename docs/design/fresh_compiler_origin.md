---
title: "Fresh compiler origin and Phase 2 lineage"
kind: design
status: current
owner: merlin-experiments
last_verified: 2026-10-09
related: [component_compiler_convergence, component_phase2_workflow, component_final_evaluation]
code_refs:
  - packages/merlin-experiments/src/merlin_experiments/phase1/component_origin.py
  - packages/merlin-experiments/src/merlin_experiments/phase1/component_generation_admission.py
  - packages/merlin-experiments/src/merlin_experiments/phase1/component_lineage.py
  - packages/merlin-experiments/src/merlin_experiments/phase1/component_qualification.py
  - packages/merlin-experiments/src/merlin_experiments/phase1/component_qualification_evidence.py
  - packages/merlin-experiments/src/merlin_experiments/phase2/component_runtime_support.py
  - packages/merlin-experiments/src/merlin_experiments/phase2/component_runtime_controls.py
  - packages/merlin-experiments/tests/test_component_origin.py
  - packages/merlin-experiments/tests/test_fresh_phase1_generation_budget.py
---

# Fresh compiler origin

The handwritten implementation is a protected final golden. Phase 1 authors a
new OOT compiler, and Phase 2 optimizes that exact output. The experiment does
not admit a handwritten compiler, its support adapter, passes, instruction
schedules or previously emitted workload artifacts as a starting compiler.
The golden remains unavailable through both authoring phases.

`run_fresh_component_phase1` creates the complete initial implementation itself:
one inert CLI and its manifest. Every entrypoint raises until the author writes
the compiler. Operators cannot replace this scaffold with a working backend.
The permitted inputs are independently issued RTL and protected minimal software
intakes, a reviewed generic compiler library, the independently generated
component corpus and exact public runtime grants. The software intake replays
independent example operation correspondences and binds explicit numerical
choices; neither a `reviewed` label nor a matching historical software hash can
admit a legacy specification. It must share the exact live hardware origin, the
coverage report's software authority and the complete public
`contract/software_spec.json` projection. Generic infrastructure upstreamed to
Merlin remains available through explicitly reviewed APIs.

Fresh inputs require the actual budgeted Phase 0 coverage v2 report. Its ordinary
reader reopens the selected policy, complete admission ledger and generated
source declarations, recalculating reference work, materialized elements, typed
payload and scalar widths. The coverage, exact policy and admission ledger
commitments enter the immutable fresh-origin input identity. A legacy v1 report
remains readable as historical evidence, but cannot establish this requirement;
resigning supplied costs or removing budget fields cannot replace source replay.
No budget limit is inferred from a workload or winning implementation.

This is logical reference-generation admission. It does not establish compiler
execution time limits, peak process heap, a large compile-only qualification
mode, hardware correctness or timing. Those still require their separate gates.

Shared translation, host compilation, linking and functional simulation tools
are probed before authoring. These probes exercise independent tool inputs;
they cannot require the inert candidate to pass compiler or CCA checks. After
authoring, the ordinary complete-domain grader must check the resulting compiler.
Origin alone establishes neither correctness nor performance.
Independently qualified grading/runtime support must also exist before authoring
spends tokens. The compiler and runtime authorities are separate to avoid
requiring a compiler before Phase 1 can start.

The actual provider runs with a fresh credential-isolated home. Candidate
commands have a networkless native permission profile; the outer namespace
contains only the current workspace, reviewed public view and individually
pinned runtime files. The private evaluator, goldens, compiler histories and
other scratch worktrees are absent. The provider performs its actual no-model
credential/workspace permission probe before requesting an author turn.

## Issued authorities

`FreshCompilerOrigin` is issued around the actual normal author transport. It
binds the independent hardware and minimal software intakes, initial inert scaffold, public library
and corpus, runtime memberships, observed invocations, transcript and complete
frozen final candidate membership. A receipt, status flag or apparent hash
record cannot reconstruct this capability. Changing authority fields also
refuses: an in-process issuance registry preserves their original binding.

`ComponentQualification` requires that issued origin before any candidate code
runs. Its ordinary numerical, complete-output, source-to-ELF, host/device
ownership, synchronization and instruction gates remain separate and mandatory.
An old compiler may still be inspected as historical evidence; it cannot become
this experiment's fresh baseline by being numerically correct.
Grading and stage verification use the issued independent runtime directly.
They do not rediscover the old target backend or resolve the handwritten support
provider through the legacy source inventory.

Qualification also binds the exact private compiler copy used by grading and
the complete actual invocation roster for every mandatory member. Verification
reopens each invocation's tool, source, dependency and product pins, including
inputs outside the grade tree, and replays the stored stage witnesses against
the original member, source, output roster and required effects. Changing the
private compiler or an external build input invalidates qualification even if
the original candidate and produced grade tree remain unchanged. Saved receipt
rows confer no fresh compiler or physical runtime authority.

Phase 2 starts from the exact qualified Phase 1 candidate. Its normal broker
and author transport produce `ComponentCompilerLineage`, which binds that origin,
the frozen edit authority, actual transcript and broker receipts to all final
candidate bytes. Qualification of changed bytes additionally requires this
issued descendant authority. Feedback tools cannot redefine the baseline or
manufacture lineage from a submitted JSON object.

## Remaining target authority

The existence of a `BuildOnlyService` does not establish that its renderer or
execution provider was independently derived. An experimental runtime must
carry its own independently qualified hardware-facing support. A missing
independent ISA decoder, ABI renderer, runtime or simulator authority is an
explicit unavailable prerequisite. The handwritten support adapter cannot
fill that absence. Fresh origin receipts do not qualify those missing services
or establish matched final FireSim performance.

## Private independent runtime controls

`PreparedIndependentRuntimeContext` composes explicitly selected build and
functional execution services. Its private primitive compiler accepts a closed
subset of upstream tensor identity and addition, emits scalar LLVM, and checks
every original output expression against the resulting SSA loads and stores.
The stock translation, object, link and execution stages retain their ordinary
invocation records. Complete independent input and output values pass through
the shared native execution path and the original numerical comparison.

This primitive compiler is private evaluator support. It is absent from the
initial scaffold and all author grants. It contains no accelerator instructions,
target schedules or workload optimizations. Its schema contract contains only
the three shared ABI schemas, excluding development and validation corpora.

The prepared controls exercise concrete defects in source correspondence,
the original output roster and the original numerical gate. An explicitly
selected live instruction check adds the fixed linked-policy control: the private
negative assembly is bound to the actual public native decoder and original
source policy, linked through a scoped build service, and rejected before
execution. Source verification
also rejects operations outside its closed LLVM subset; that rejection does
not establish an audit of linked target instructions. A consistent partial
compiler ABI still fails against
the original source roster. Changing a capsule's tolerance does not change
the separately retained original policy. A rejection qualifies as a negative
control only when actual observed products establish its declared cause.

Only the explicit whole-linked-ELF check establishes its static instruction
policy; the primitive source verifier alone does not. These diagnostics grant no
accelerator effect, physical ownership,
synchronization, hardware/model equivalence or timing authority. Their stage
verifier explicitly refuses those missing mechanisms. The fixed fourteen-case
runtime qualifier therefore remains unavailable until independent producers
establish every required mechanism. A numerical pass cannot issue that runtime
authority, start a fresh authoring session, or enable Phase 2 feedback by itself.
