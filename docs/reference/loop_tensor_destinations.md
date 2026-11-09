---
title: Loop-carried tensor destination reuse
kind: reference
status: current
owner: llvmlower
last_verified: 2026-11-09
related: [architecture, lowering_pipeline]
code_refs:
  - src/merlin/llvmlower/loop_tensor_destinations.py
  - src/merlin/llvmlower/pipeline.py
  - src/merlin/llvmlower/impr_features.py
  - merlin/tests/ir/test_loop_tensor_destinations.py
---

# Loop-carried tensor destination reuse

`reuse_loop_tensor_destinations` is an explicit, default-off compiler feature.
It runs after tensor fusion, immediately before one-shot bufferization. Upstream
`ElementwiseOpFusion.cpp` deliberately replaces unused `outs` operands with
`tensor.empty` through `RemoveOutsDependency`. At this later seam, a complete
overwrite may instead use the corresponding serial loop-carried tensor.

The yielded value must be a direct, single-result `linalg.generic` in the loop
body with the exact carried tensor type, positive static dimensions, only
parallel iterators, an identity output map, and an unused output-init block
argument. Its complete scalar body must contain numeric arithmetic/math or
linalg index/yield operations, with no calls, opaque operations, regions or
buffer operands. Dynamic/empty shapes, live init values and library calls stay
unchanged. No operation is selected by provenance or workload identity.

Only the unused destination SSA edge changes. Every payload operation, input,
type, attribute and arithmetic order remains. Native bufferization decides
whether physical storage can be reused, including conflicts with live inputs.
Ordinary deallocation and lifetime checks remain mandatory. The feature grants
no physical alias, effect, numeric approximation or target permission.

The runner requires exactly one marker immediately before bufferization and
rejects a moved or missing seam. Empty feature selection retains the original
runner. Native proof tests cover init reads, zero/dynamic shapes and opaque
effects. Triton producer integration additionally compiles both tensor and
mixed-memory profiles and reuses one library for zero/nonzero runtime loop
counts while checking all outputs and preserved inputs.
