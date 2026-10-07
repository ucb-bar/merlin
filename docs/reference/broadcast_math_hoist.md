---
title: Typed broadcast reciprocal square root hoisting
kind: reference
status: current
owner: llvmlower
last_verified: 2026-10-06
related:
  - docs/reference/architecture.md
  - docs/reference/scalar_pointwise_unroll.md
code_refs:
  - src/merlin/llvmlower/broadcast_math_hoist.py
  - src/merlin/llvmlower/pipeline.py
  - merlin/tests/ir/test_broadcast_math_hoist.py
---

# Typed broadcast reciprocal square root hoisting

`hoist_broadcast_source_rsqrt` is an explicit, default-off host compiler feature.
It runs after source elementwise fusion and immediately before bufferization.
Fusion can place a row operation inside a wider channel consumer. The rewrite
materializes its exact scalar dependency chain once at each distinct row point.
Later fusion cannot absorb the producer again at this seam.

## Legality

The consumer must be a single-result all-parallel tensor `linalg.generic` with
static, strictly positive extents and a complete output permutation. Every
input map must use distinct dimensions or zero for a size-one tensor dimension.
Affine expressions, dynamic domains, empty domains, reductions, index-dependent
bodies, vectors, memory operations, calls, unknown operations, strict FP scopes
and nonempty fastmath permissions refuse. No external symbol name grants purity.

Typed block arguments establish dependence on logical iterator dimensions.
Scalar SSA propagates these sets through registered pure source arithmetic and
casts. A source `math.rsqrt` is eligible when at least one repeated dimension
is absent from its dependency set. Its chain must not depend on the output-init
argument. The smaller producer is placed immediately before the consumer, after
all inputs have been defined; a preceding reduction remains in its original
place and order. Input and output layouts follow the original affine maps.

## Arithmetic, environment and ownership

Every selected source operation, scalar type, cast, operand order and attribute
is cloned exactly. No reciprocal approximation, FMA contraction, reassociation,
reduction change or precision change is introduced. The consumer uses the small
tensor through its proved projection. Backward liveness retains original chain
intermediates with other consumers, and every use of the original tensor result
receives the rebuilt consumer. Existing live output-init reads are retained.

The transform relies on registered source-operation effects and the original
nontrapping math contract; it does not attach purity to emitted library calls,
grant new exception-flag permissions, or rewrite a strict floating environment.
The smaller producer executes only when the original consumer domain is proved
nonempty. Individual repeated operations retain their source arguments and
intermediate precision. Directed compiled cases check special values; actual
target capsule qualification also checks all five rounding modes and sticky
flags against the control under its unchanged runtime.

Tensor value semantics establish source alias legality. Upstream bufferization
owns physical buffers, including any permitted output-init reuse. No `restrict`
or physical no-alias fact is inferred. The producer requires an additional small
tensor, so its allocation, lifetime, reads and writes must be included in complete
cost measurements and whole-model qualification. There is no automatic routing
or workload-name selection. Independent shape, tail, map, cast, live-use and
refusal tests exercise the shipped source implementation through actual lowering.
