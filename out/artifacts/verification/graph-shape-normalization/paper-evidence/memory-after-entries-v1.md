# Whole-entry paper successor — uncompiled, not qualified

The old selected-entry result alone does not cover every module entry. The new
candidate `Merlin.GraphLoopsMemorySCF.After.checked_entry_roster`, instantiated as
`GraphMemoryAfterEntries.every_definition`, covers every listed definition through
an exact source/final roster and checked per-entry route. Existing retained module:
`kernel(i32,i32,memref<1xi32>)->(i32,index)`,
`touch(i32,memref<1xi32>)->i32`, `untouched(i32)->i32`; all default-public, no declarations.

Proposed future claim: equal possible finite successful return/public-heap
observations for every supported entry's typed external-caller domain, under one
shared partial operation interpretation and the explicit private total
unobservable allocator. It is NOT yet an accepted Lean theorem, nor a claim about
all pass inputs, errors/divergence, native/hardware behavior or numerics.

Kernel preservation reuses the new selected complete chain. Unchanged entries use
finite-call-depth congruence over their complete reachable unchanged submodule.
A wrapper reaching the changed kernel is explicitly refused pending a contextual
heap/private-clock proof. Body equality alone is not used as callee equivalence.

The fixed old interpretation lacks scalar literals, making `untouched`'s outcomes
empty. The successor explicitly adds a common partial effect-free literal relation;
closed kernel support proves it cannot affect the selected kernel. No numeric
literal law is added. Typed input/caller/heap witnesses and a constructed Unit-valued
literal/binary source execution demonstrate abstract nonvacuity candidates without
assuming source execution. All remain UNCOMPILED.

Worked example and retained preparation output:
`../after-binding-deadline-v1/final-memory-after-v1/source-entries-v1/evidence/positive/`
contains the full raw source/final, `Entries.lean` and `PREPARATION.json`. The example
has two private 2×3 i32 epochs, one reused allocation, two tile-2 initializers,
ordered updates/loads, two caller-observing calls, one free and complete returns.
Twelve SDK-free preparation cases pass, including valid binding mutations and an
explicit unsupported wrapper. All Lean positive/negative outcomes are UNOBSERVED;
this is not retained checker acceptance, qualification or a semantic detection rate.

Exact source closure/root/entry/field/premise maps and reviewer scope corrections:
`../after-binding-deadline-v1/final-memory-after-v1/entries-source-review-v1/README.md`.
New immutable cut: 9 additive sources; 50 with prior sources, 111 Lean/46 Python
project source closure. Old source cuts and historical qualifications are unchanged.
`entries-topology-v1/TOPOLOGY.json` verifies shape output equals reuse input; memory
middle is AFTER reuse, and therefore intentionally differs. These byte identities
provide data provenance, not a semantic composition proof.

Remaining claim gates: justified runtime context, compile/type/std3 audits, actual
positive and binding-negative checks, new qualification/replay, and independent
Graham/Prepare review of projection, primitive/literal/descriptor/caller/allocation
assumptions. Old SDK incidents/failures/counterexamples stay retained. gSIM is separate.
