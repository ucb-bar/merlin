---
title: Borrowed pointwise lane packets
kind: reference
status: current
owner: compiler
last_verified: 2026-10-07
related: [reference/architecture.md, reference/prepared_model_transform.md]
code_refs:
  - src/merlin/llvmlower/scalar_pointwise_packet.py
  - merlin/tests/ir/test_borrowed_pointwise_packet.py
  - merlin/tests/ir/test_scalar_pointwise_packet.py
---

# Borrowed pointwise lane packets

`packet_borrowed_pointwise_fma_division_2` is an explicit ordinary lowering
feature for independent scalar lanes in a borrowed memref writer. It shares
the existing pure pointwise body checker with tensor packet schedules and runs
before one-shot bufferization. It selects no workload or accelerator.

The caller first binds a `BorrowedPointwiseEffects` contract to a live
`linalg.generic` writer with `bind_borrowed_pointwise_packet`. All permissions
must be explicit: immutable input spans, fresh output disjoint from every
input, stable rounding, nontrapping arithmetic and unobserved exception flags.
The caller owns the physical lifetime and effect proofs. A memref type or
function name cannot establish them. Read-only inputs may alias each other.

The pass independently checks one full output, static positive extents,
projected input maps, a complete output permutation, positive static strides
and an injective output address layout. It admits a pure scalar binary32 body
with at least four source FMAs and a division. Output-init reads, unknown
operations, dynamic layouts, constrained floating contexts and fastmath refuse.
Unsupported source remains on its ordinary lowering path.

Two lanes interleave independent scalar operations while retaining every
source arithmetic operation and its order within each lane. A bounded scalar
tail covers odd extents. The output map, strided parent storage and enclosing
`prov.*` source trace remain; `prov.transforms` records the packet rewrite.
The explicit contract is consumed by the selected rewrite.

Empty feature selection preserves the original pipeline. This option composes
with the existing tensor packet feature; neither implies physical alias or
floating-effect permission for the other. Upstream bufferization continues to
own tensor storage semantics. Actual generated copies, preparation, consumers,
stores, stack pressure and waits must be priced together before promotion.
Source tests cover independent shapes, row pitches, output permutations,
read-only input aliasing, odd and singleton tails, supported rounding modes,
source trace preservation and conservative refusals.
