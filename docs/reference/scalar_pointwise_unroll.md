---
title: Scalar pointwise FMA and division loop scheduling
kind: reference
status: current
owner: llvmlower
last_verified: 2026-10-06
related:
  - docs/reference/architecture.md
code_refs:
  - src/merlin/llvmlower/scalar_pointwise_unroll.py
  - merlin/tests/ir/test_scalar_pointwise_unroll.py
---

# Scalar pointwise FMA and division loop scheduling

`unroll_scalar_pointwise_fma_division_by_2` is a default-off compiler scheduling
choice. It adds an upstream LLVM factor-two partial-unroll annotation immediately
before structured loops become control flow. It grants no floating-point
reassociation, contraction, reciprocal or approximation permission.

## Recognition and fallback

Eligible loops are static, innermost `scf.for` loops with step one, at least two
iterations, no loop-carried values and one induction variable. Their flat body
must contain scalar f32 division, at least four existing scalar f32 FMA operations,
loads and stores. Remaining operations are arithmetic, affine index applications
or an empty yield. Calls, nested regions, vectors, non-f32 FMA/division, dynamic
bounds, nonempty fast-math permissions, strict floating-point scopes and existing
loop annotations are refused. Unsupported loops retain their original form.

This recognition uses generated operation semantics and types. It does not use
model names, capture provenance or device identifiers. The FMA count is an explicit
cost heuristic for amortizing partial-unroll overhead, not a numerical contract.

## Numerical and memory semantics

Only loop metadata changes. Each element retains its original operation graph,
including separate multiply/add operations and any already-fused FMA. Upstream
LLVM owns dependency analysis and its remainder loop. No no-alias attribute is
introduced; a load/store dependence must still be respected. Tensor aliasing and
any required copies are resolved by the existing bufferization pipeline before
this stage. Odd trip counts retain a scalar remainder when needed.

The scheduling choice may increase code size or register pressure. It is a search
option, not a performance guarantee. Actual scalar host assembly, whole-model
numerical gates and hardware measurements decide whether a caller enables it.

## Verification

The focused suite compares actual optimized native baseline and selected outputs
bit for bit, including odd tails, tensor input/destination aliasing, signed zero,
infinities and existing FMA cancellation. It also checks structural refusals and
that the default pipeline remains byte unchanged.
