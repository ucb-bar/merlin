---
title: "Scalar BF16 elementwise legalization"
kind: reference
status: current
owner: ir
last_verified: 2026-11-10
related: [architecture, triton_frontend]
code_refs:
  - src/merlin/llvmlower/bfloat_elementwise.py
  - src/merlin/llvmlower/bfloat_transport.py
  - src/merlin/llvmlower/pipeline.py
---

# Scalar BF16 elementwise legalization

The explicit, default-off `emulate_bfloat_elementwise` feature legalizes
scalar BF16 `arith.maxnumf`, `arith.minnumf`, `arith.maximumf`,
`arith.minimumf`, `arith.divf` and `math.sqrt`. The source program retains
its BF16 tensor/storage types and source dataflow. This is a selected compiler
implementation, not permission from a provenance tag.

The feature runs once at the initial native source boundary. It validates the
whole module before editing candidates, requiring complete scalar signatures
and neutral floating-point flags. Existing loops, branches, memory effects,
raw bitcasts and transport remain present. Unknown BF16 arithmetic, scalar
calls, vector/tensor arithmetic, nonneutral flags and explicit unsupported
rounding modes refuse. Supported transport operations are not classified as
pure; their nested operations remain inspected independently.
Shaped constants and exact transport keep their existing forms; the scalar
restriction applies to arithmetic candidates, not neighboring tensor data.

Arithmetic reuses upstream `arith-emulate-unsupported-floats` with
`source-types=bf16 target-type=f32`, which extends each operand, computes in
f32 and truncates each result back to BF16. This leaves buffer element types,
function signatures, memory effects and placements unchanged. The reference
implementation and tests are LLVM
`a47bddccec30255619bb8c37fa59700e661d4e66`,
`Arith/Transforms/EmulateUnsupportedFloats.cpp` and
`test/Dialect/Arith/emulate-unsupported-floats.mlir`.

The upstream pass does not cover `math.sqrt`. Its scalar BF16 case receives
the same explicit extend/sqrt/truncate boundary. BF16 has eight significand
bits; f32 precision resolves the relevant BF16 square-root rounding midpoint.
The selected host path uses the normal nearest-even, gradual-underflow
floating-point environment. Finite, infinite and signed-zero words are checked
exactly; computed NaN class is retained without imposing a payload choice.
This grants no alternate rounding, fast-math or physical target support.

`arith-expand{include-bf16=true}` still expands the casts before LLVM
conversion. The separate `preserve_bfloat_transport` feature then represents
only remaining scalar word transport as i16. Its original refusals for BF16
arithmetic, vector transport and by-value BF16 function ABIs remain active.
Ownership, lifetime, allocation, public signature and complete-LLVM checks also
remain active. No opaque runtime helper or changed buffer ABI is introduced.

Native tests exercise positive scalar division/sqrt, idempotence, unchanged
refusals and the explicit runner seam, alongside the existing transport
regressions. triton2MLIR integration additionally checks every BF16 input word
through clamp/div/sqrt while an adjacent uncomputed result preserves all
65,536 raw words exactly, including signaling NaNs and signed zeros.
All three tensor/memory profiles reuse actual Merlin libraries over empty,
single-element and ragged runtime domains. These are host observations,
separate from staged accelerator qualification.
