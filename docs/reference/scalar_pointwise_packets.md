---
title: Scalar pointwise lane packets
kind: reference
status: current
owner: llvmlower
last_verified: 2026-10-06
related: [docs/reference/scalar_pointwise_unroll.md, docs/reference/architecture.md]
code_refs:
  - src/merlin/llvmlower/scalar_pointwise_packet.py
  - merlin/tests/ir/test_scalar_pointwise_packet.py
---

# Scalar pointwise lane packets

The explicit `packet_scalar_pointwise_fma_division_2` and
`packet_scalar_pointwise_fma_division_4` choices interleave independent scalar
chains along the last logical dimension before bufferization. Each lane retains
the original arithmetic and intermediate types. These choices form one
alternative group and are disabled by default.

`packet_scalar_pointwise_broadcast_2` is another choice in that group. It selects
an axis omitted by at least one used, nonscalar input map. Two lanes share an
extract only when the immutable tensor and every projected coordinate are
identical. Packet groups surround the other parallel dimensions, so the same
projected input can feed both lanes. An odd extent has its exact scalar tail.

Axis selection counts shared input extracts, then prefers the larger static
extent. It is a deterministic search heuristic, not a hardware cost estimate.
Scalar inputs alone do not justify interchange. Source provenance and workload
names do not participate in selection.

## Legality

Eligible bodies are static all-parallel tensor generics with at least four
existing f32 FMAs and an f32 division. Typed projected-permutation input maps and
a full output permutation establish every coordinate. The destination's initial
scalar must be unread, and the body must contain only the declared pure scalar
operations. Unknown calls, index-sensitive operations, dynamic or empty extents,
nonempty fastmath and strict floating scopes refuse rewriting.

No reassociation, reciprocal substitution, approximation or physical no-alias
permission is introduced. Tensor value semantics remain authoritative, including
live source aliases and any copies required by upstream bufferization.

## Qualification

Actual optimized native execution checks independent inputs, odd packet tails,
broadcast loads, live aliases and output permutations across multiple dimensions.
Refused selections retain the original LLVM, and an empty feature set retains
the default pipeline. Generated scalar code, copies, allocation, cache behavior
and register pressure must be included in complete timing before a caller
promotes any choice.

## Multiplication choice

`packet_scalar_pointwise_multiplication_4` is a separate, explicit experimental
choice for bodies containing at least three f32 multiplies. The body may retain
other declared scalar f32 and integer operations, including source quantization.
FMA, division, floating precision changes and strict scopes refuse this selector.
Static input maps may additionally contain zero coordinates on unit dimensions;
these coordinates are extracted exactly, and the output remains a full permutation.
Four lanes retain every original multiplication and intermediate rounding, with
bounded scalar tails. There is no CPU instruction implementation in this pass.

The multiplication and FMA/division selectors are disjoint and may be selected
together. Existing features and the empty policy keep their original behavior.
The multiplication choice supplies no profitability or floating environment
permission. Complete matched body costs and unchanged whole-model accuracy are
required before a schedule is enabled for a build.
