# AGENT.md — src/merlin/verification

Own independently checked program and transformation obligations. Preserve the
original source, selected semantics, verifier identity and exact checked scope.
Structural fidelity is not backend arithmetic or hardware execution evidence.
Compiler candidates cannot choose acceptance thresholds or replace their grader.

The former `verify` implementations now live here, including SMT translation,
Lean checking/toolchain intake, emitted-program obligations and their mathematical
resources in `lean/Merlin`. Historical `merlin.verify.*` imports are exact module
aliases with the same live completion state. Current callers and the two installed
verification commands select these owners; archived source profiles stay exact.
Optional historical plots/replay remain research-owned in the legacy namespace.
Preserve actual semantic source closure, tool pins and independent live completion;
moving files never admits saved results or changes acceptance policy.

`partition.py` reconstructs complete original SSA from actual outlined calls.
Keep its algorithm and refusal conditions unchanged during ownership migration.
Old imports may alias this owner; receipts must identify the actual verifier
source rather than a compatibility module. Retain historical record spellings.

`outline_ir.py` checks pure outlining by exact typed SSA beta-equivalence.
`scalar_ir.py` proves actual pure scalar-integer before/after IR with QF_BV;
unsupported operations abstain. `split_reduction.py` checks only the declared
signed-width accumulation algebra, not emitted kernels. Keep their acceptance,
refusal, parser/tool identities and explicit limitations unchanged. The legacy
`merlin.verify` modules are exact aliases and retain record/exception spellings.

`packing_types.py` owns the original typed allocation/effect premises.
`packing_ir.py` checks actual LLVM pointer uses, allocation lifetimes and alignment;
`packing.py` batches shared imports while preserving each original island budget.
`host_call_alignment.py` owns native import and exact address-alignment evidence.
Pin all actual verifier sources, not a legacy alias. Unknown effects, escaped
pointers, changed source and unproved budgets retain their original refusals.

`convolution.py` independently replays the original bounded im2col witness against
actual affine windows, reduction order, output coverage and loop-carried tensor
destinations. Shared record types and pure IR predicates live below compiler and
verifier in `compiler.ir.convolution`. Preserve every original check and source
pin; this supplies no allocation, backend or hardware qualification.
