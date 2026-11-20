---
title: Zero-element tensor values
kind: reference
status: current
owner: ir
last_verified: 2026-11-20
related: [lowering_pipeline, effect_kernel_intake]
code_refs:
  - src/merlin/frontends/kernel_shape_abi.py
  - src/merlin/llvmlower/qinner.py
  - merlin/tests/ir/test_kernel_shape_abi.py
  - merlin/tests/ir/test_qinner_binding.py
---

# Zero-element tensor values

The standalone tensor shape ABI accepts nonnegative static tensor extents.
For example, `tensor<2x0x3xi64>` retains its exact rank, dimensions and dtype;
it needs no synthetic runtime dimension binding. Negative extents, Boolean or
noninteger dimensions and invalid logical grids still refuse. Grid extents
remain strictly positive. This extends the accepted value domain without a new
wire field or schema version; older positive-only readers reject static zeros.

An untagged `tensor.empty` has no unwritten elements if an actual static axis
is zero, an actual dynamic size is literal zero, or its enclosing `scf.if`
edge proves at least one of its own dynamic sizes zero. The initialization
checker follows typed equality/inequality-to-zero comparisons, Boolean
AND/OR and literal XOR negation. An impossible constant edge is also vacuous.
The Boolean proof has a bounded visit budget; unavailable proofs retain the
ordinary uninitialized-read refusal.

For example, the false arm of `(m != 0) && (n != 0)` may return an exactly
shaped `tensor.empty(m,n)`. Each alternative false comparison establishes
that an actual dimension is zero. This handles runtime empty-slice values
without executing an extraction with a clipped boundary offset. A comparison
of a different dimension, an unknown control that bypasses the comparison,
or a signed `m > 0` test supplies no zero-size fact. Source integer/index widths
remain undecided; provenance and sampled invocation sizes cannot prove vacuity.

Tagged quant-inner placeholders retain their original plan, lift and archive
binding requirements, even when empty. Skipping data initialization for an
untagged zero-element value grants no bounds, ownership, memory-effect or
physical-alias permission. Active reads from zero-sized storage remain invalid
and must meet the caller's independent access contract.

The unit tests cover static and dynamic literal zeros, Boolean branch edges,
typed serialization, unknown/wrong-dimension refusals and mandatory empty
quant-inner binding. Separate triton2MLIR integration tests execute empty
functional tensor ports through host and actual serial-target images and
confirm active zero-storage accesses abort without publishing results.
