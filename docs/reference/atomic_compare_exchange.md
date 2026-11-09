---
title: Native integer compare/exchange legalization
kind: reference
status: current
owner: llvmlower
last_verified: 2026-11-09
related: [loop_tensor_destinations, lowering_pipeline]
code_refs:
  - src/merlin/llvmlower/atomic_compare_exchange.py
  - src/merlin/llvmlower/dealloc_lifetime.py
  - src/merlin/llvmlower/loop_tensor_destinations.py
  - src/merlin/xdsl_dialects/standard_memref.py
  - merlin/tests/ir/test_atomic_compare_exchange.py
---

# Native integer compare/exchange legalization

Portable source keeps `memref.generic_atomic_rmw` with integer comparison,
conditional replacement and the unchanged old value. The explicit, default-off
`lower_integer_compare_exchange` compiler feature lowers that closed region
for the native host path. It matches source operations/types/SSA, independently
of kernel names, source IDs, references or a target accelerator.

The region must contain exactly integer equality, a select of the supplied new
value versus the current word, and `memref.atomic_yield`. Comparison and new
values come from outside the region. Qualified buffers have rank one, identity
layout, the default memory space, and i16/i32/i64 elements. Broader atomic
regions, float policies, layouts and external synchronization require separate
capabilities. This pass adds no bounds, initialization, alias or numeric proof.

The native helper takes a memref, index, comparison and new value and returns
the old scalar. Its body derives the byte address from that memref and index
and performs one standard LLVM cmpxchg with acq_rel success/acquire failure
ordering. It allocates nothing, releases nothing, and returns no pointer or
buffer. All code remains in the compiled module; no external implementation
or device runtime supplies the operation.

The pinned ownership deallocator rejects scalar-result regions without
RegionBranchOpInterface, including MemRef's generic atomic region. Its effect
checks also reject the scalar LLVM primitive inside an ordinary function body.
The legalization therefore exposes the generated helper's typed borrowing call
boundary during ownership analysis. It retains the complete actual helper IR
object, temporarily using its declaration, then restores the body before
allocation liveness, lifetime verification and final LLVM conversion. Restoration
checks the entire current declaration attribute set. A changed or missing
declaration fails; an external helper cannot survive as a successful result.
Ownership and native bufferization stay enabled.

Lifetime borrowing grants independently reconstruct the complete current private
helper body after ordinary canonicalization, along with exact argument/result
types and root symbol scope. Changed ordering, byte addressing, visibility,
extra effects or opaque bodies grant nothing. A spelling or saved record cannot
stand in for an implementation. Nested symbol tables do not inherit the root
helper's binding.

The feature inserts one explicit marker immediately before one-shot
bufferization, after any selected loop-destination reuse. The ownership prefix
must finish lowering deallocations before helper restoration, and liveness
verification must follow restoration. Unsupported/missing seams fail. Empty
selection preserves the original runner.

Loop-destination reuse may consume an already computed CAS snapshot under its
existing independent-seed/private-target and exactly-once boundaries. The
atomic region itself remains in place; only the downstream pure accumulator
destination changes. Native alias and lifetime checks still decide storage
reuse. Host execution and source numerical/effect checks remain separate from
accelerator materialization or performance claims.
