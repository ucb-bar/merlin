---
title: Mandatory original source-only compiler evaluation
kind: design
status: current
owner: merlin-experiments
last_verified: 2026-10-09
related: [component_compile_sources, component_scale_generalization, fresh_compiler_origin, component_launch_qualification, counted_copy_static_proof]
code_refs:
  - src/merlin_experiments/phase1/component_compile_admission.py
  - src/merlin_experiments/phase1/component_compile_roles.py
  - src/merlin_experiments/phase1/component_origin.py
  - src/merlin_experiments/phase1/component_qualification.py
  - src/merlin/targetgen/compile_only_execution.py
---

# Mandatory original source-only compiler evaluation

Fresh Phase 1 requires a live independently issued source-only roster sharing
its original hardware, minimal software and target descriptor. Both original
guard and withheld transfer members are mandatory. This selection is checked
after independent functional runtime admission and before public author grants.
The fresh input identity binds its digest; private sources and shape variants
stay outside the author view. Existing numerical v2 coverage remains separate.

## Ordinary transport

`evaluate_component_compile_roles` consumes that same live roster and actual
fresh compiler origin or observed descendant lineage. It snapshots the exact
candidate privately and compiles every original source under the existing
scoped package executor. It invokes the same ordinary frontend, lowering,
translation, object and link pipeline as numerical grading, with explicit
`BuildOnlyService` and independently derived whole-ELF instruction policy.

Large original tensor types never enter operand/golden construction, full-value
readback or runtime execution. The original source ABI still binds every input
and output to the emitted pointer entry and command buffer. Linked symbols and
all executable sections are inspected. This establishes transport availability
and instruction-policy enforcement; it does not establish output stores,
semantic coverage, index/resource legality or physical behavior.

The private `merlin.component_compile_role_evaluation.v1` report keeps complete
original member, source and ordered ABI identities. Compilation and original
static obligations have distinct denominators. A failure or exhausted explicit
compilation budget retains the original member and available attempt evidence;
it never becomes an accepted static refusal. Expected `static_refusal` cases
remain unresolved until an independent original legality checker can establish
the required reason. Saved reports cannot reconstruct the live evaluation.

## Mandatory qualification join

`qualify_component_compiler` requires the exact live evaluation for the same
original roster, origin, lineage, current candidate and grading contract. It
reopens the original source owner, compiler and private snapshot, implementation
membership, actual transport records/products, build selection and instruction
selection. Mutation or missing original membership refuses admission.

An independently selected pointer-storage policy and the fixed counted-copy
checker can establish semantic, index and complete-output obligations for its
supported scalar copy loops. The checker reopens the original source and actual
LLVM product; unsupported programs and all resource obligations remain
**UNKNOWN**. See [Conditional counted-copy static proof](../reference/counted_copy_static_proof.md).
The evaluator does not accept a proof callback or promote successful linking to static proof.
Small numerical passes may still be retained as diagnostic results, but full
compiler qualification remains refused while any original compilation or static
obligation is unresolved. Qualification verification and subsequent launch
reopen the same live evaluation. Historical receipts remain inspectable; missing
source-only authority cannot grant a new qualified launch.

Phase 2 preserves the original baseline and runtime owners when launching a
descendant. Before promotion it independently evaluates the current descendant's
original compilation and static obligations again; the baseline's live evaluation
cannot stand in for changed compiler bytes. Original numerical grading follows
that join. Each selected component execution deadline is shared across its
compile, source checks, link, execution, readback and publication boundaries;
individual subprocesses use the remaining time. Synchronous Python work is
checked at boundaries rather than forcibly preempted.

## Evidence and remaining work

Actual native tests replay independently selected RTL/software/source rosters,
including large original tensor signatures. Structural transport controls run
real stock MLIR translation, object compilation, linking and recorded package
commands; they retain every original member on deliberate compiler failure and
detect private compiler mutation. These controls deliberately isolate provenance
facets and emit no output stores. They do not seed Phase 1 or issue an independent
functional runtime, fresh origin, ISA or static proof authority.

Independent emitted-program semantic, index, resource, complete-output and
expected-refusal proof producers are still required. Tool/source pins preserve
the observed build recipe and drift attribution; they do not alone prove full
transitive toolchain/SDK closure. Numerical execution, physical effects and
performance remain separate evidence modes, with their original gates.

## Installed regression scope

`build_tools/scripts/qualify_installed.py --suite component-convergence`
archives the committed component tests, including conditional copy and container
transport controls, and verifies their imports come from built wheels outside
the checkout. Explicit native compiler selections require zero skips in the
original compile-role transport and shared-deadline rosters. Other tests retain and report missing
prerequisites separately; the suite grants no container or LLVM library tools
implicitly.

Each installed qualification owns a unique external test temporary root and
retains passing as well as failed fixtures. Original invocation records and
native products remain available for reopening after the suite completes;
pytest's passing-fixture cleanup cannot erase that evidence. Retention does not
change the declared zero-skip subset or grant execution authority.

The core-only `host-arithmetic` suite checks shared CPU arithmetic, with an
explicit host toolchain requiring zero skips throughout its original test
roster. The core-only `invocation-record` suite checks actual subprocess
environment and executable provenance. These suites establish packaging and
their stated regression scope, not fresh compiler, runtime or hardware authority.
