# Supported caller-context paper successor — UNCOMPILED

The selected-kernel theorem preserves returned values, caller/public heap and
user state for finite successful runs; it does not identify private allocation
clocks or private cells. The new candidate closes the one-call caller-context gap
by proving a constructorwise continuation frame law over those observations.

The supported normalized wrapper has one direct kernel call followed by checked
scalar/index operations and explicit public loads/stores, ending in scalar/index
returns. Argument aliases and identical raw descriptor aliases removed by shape
normalization are covered by finite checks and existing normalization roots.
Regions, additional calls, allocating prefixes and private-pointer escape remain
explicitly refused. Repeated calls require a new stability proof for related
initial heaps and different nonzero private clocks; the clock-zero theorem is not
silently reapplied.

Candidate roots: `PublicOpcode.transfer`, `suffix_transfer`,
`CallerContext.complete_iff`, `CallerContext.typed_arguments`,
`ContextBoundary.kernel_intermediate`, `CallerContext.checked_boundary`, and
`checked_context_roster`, instantiated by `GraphMemoryAfterContext.every_definition`.
Kernel preservation is derived internally from the existing actual boundary.
No whole-callee/run/world equality or global intermediate typing premise is added.
The complete source/final function roster, interfaces, calls, SSA, actual memory
addresses and returns remain bound to independently decoded raw programs.

The worked example appends a wrapper to the retained two-epoch raw modules;
**this is a source-level test extension, not an original producer capture**.
It calls the kernel, adds a bias, loads the caller cell, adds the continued value,
stores and returns it. A source-bound witness candidate reuses the immutable epoch
witness construction under its existing Nat-addition interpretation: inputs17/29,
caller91, bias7; kernel returns308 with caller183; wrapper returns/stores498.
Private cells start uninitialized and the actual initializer proof constructs them.
No successful source execution is assumed. This demonstrates an abstract nonzero
execution candidate, not native i32/numerical fidelity.

Retained actual raw source/final, full query and preparation output:
`../after-binding-deadline-v1/final-memory-after-v1/source-context-v1/evidence/positive/`.
Three expected-positive preparations (base, argument alias, descriptor alias),
six independently xDSL-valid final-binding mutations, and five explicit unsupported
variants give **14 passing SDK-free tests**. All Lean outcomes are UNOBSERVED.
There is no retained new Lean checker acceptance or qualification/replay result.

Source/review packet:
`../after-binding-deadline-v1/final-memory-after-v1/context-source-review-v1/README.md`.
It contains exact frame/domain/theorem/entry/field maps, refusal obligations,
raw-query associations and source/review patches. New immutable source cut has
11 additive files;61 with prior unchanged sources,118Lean/51Python project closure.
The prior50 files and sealed packets remain unchanged.

Proposed claim AFTER compile/type/std3 audits, concrete checks, qualification/replay
and independent review: preservation of possible finite successful return/public
observations for every checked entry of this extended boundary and supported
caller grammar, under the explicit primitive/descriptor/total-private-allocation
model. No equality of errors/divergence, allocator success, native ABI, numerical
behavior, hardware results, all pass inputs or all compiler transformations follows.
The original runtime hold, counterexamples and independent review obligations remain.
