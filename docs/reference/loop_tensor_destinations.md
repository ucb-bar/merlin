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
argument. Its original scalar-body profile contains numeric arithmetic/math or
linalg index/yield operations. A second closed profile also permits scalar
`memref.load` observations through standard views of a `read_only`
`bufferization.to_buffer` over an actual function-entry tensor argument, with
numeric `scf.if` branches. The loop seed must then be independently synthesized:
an empty tensor, dense literal, scalar fill, or its complete pure generalized
form. Tensor-fed seeds, CFG phi arguments, unknown memory roots, writes and
opaque calls do not qualify for that read extension. Every nested branch and
operation is checked; no capture, effect or arithmetic changes.
Dynamic/empty shapes, live init values and library calls stay
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
Source-word pointer recurrences additionally exercise fused readonly gathers,
changing runtime row counts, final pointer results and zero trips through one
actual Merlin library. Destination reuse preserves that fusion while ordinary
bufferization and post-deallocation lifetime verification remain mandatory.
