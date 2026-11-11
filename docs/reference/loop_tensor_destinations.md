---
title: Loop-carried tensor destination reuse
kind: reference
status: current
owner: llvmlower
last_verified: 2026-11-11
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

Direct destination retargeting requires a single-result `linalg.generic` in the
current block with the exact carried tensor type, positive static dimensions, only
parallel iterators, an identity output map, and an unused output-init block
argument. Its original scalar-body profile contains numeric arithmetic/math or
linalg index/yield operations. A second closed profile also permits scalar
`memref.load` observations through standard views of either a `read_only`
`bufferization.to_buffer` over an actual function-entry tensor argument, or an
actual private `memref.alloc`, with numeric `scf.if` branches. The loop seed must then be independently synthesized:
an empty tensor, dense literal, scalar fill, or its complete pure generalized
form. Complete unused-init generics may establish seeds from numeric tensor
inputs after recursively checking their independent destination. Init-dependent
seeds, CFG phi arguments, unknown memory roots, writes and
opaque calls do not qualify for that read extension. Private allocations supply
disjointness from that independent seed, not a readonly fact or release authority.
Every nested branch and
operation is checked; no capture, effect or arithmetic changes. Loop-local
`tensor.extract` captures may read actual function-entry tensor values through
standard tensor views, restricted imports from the qualified private roots,
or recursively checked complete pure tensor producers/dense literals/fills.
Computed read roots retain their original evaluation and grant no ownership of
their input buffers. Unknown/effectful/library-backed and init-dependent
producers remain outside this read profile.
The independent seed remains required; function CFG phi arguments and unknown
snapshot owners do not qualify. This recognizes tensor views recovered by native
canonicalization without weakening lifetime or effect checks. Loop-local
scalar capture producers also close recursively, so moving a scalar load outside
the generic cannot bypass its root/seed checks. Pure padding/conditional values
and restricted tensor imports from the qualified roots remain in that closure.
Serial `scf.while` after-regions also qualify when the condition forwards every
before-argument unchanged, in the same position and type. Computed, permuted or
heterogeneous condition state remains unchanged because it cannot inherit the
corresponding initial seed facts. The original condition, iteration order and
zero-trip behavior stay executable; no iteration bound is invented.
Dynamic/empty shapes and library calls stay unchanged. No operation is selected
by provenance or workload identity.

Conditional state may follow nested `scf.if` results. Every selected result
position in both regions must qualify before any path changes. A pass-through
of the carried value needs no edit. A captured immutable tensor needs a pure
parallel value copy on that path, with the carried value as its destination.
Loop-local arithmetic/tensor values and pure tensor linalg producers may also
use a complete value copy after closing their producer DAG against hidden
effects/captures. The same readonly-entry/fresh-seed gate applies to scalar
memory captures; unknown producers, writes and opaque calls do not qualify.
Reduction bodies and live original init arguments remain intact; the separate
copy moves their complete computed value. Library calls grant no permission.

An additional dependency profile accepts an already-computed integer atomic
snapshot. Its standard `scf.execute_region no_inline` must contain exactly one
tensor generic and its result yield. The generic's output seed is independently
synthesized, its output-init argument is unused, and its closed numeric body
may contain i32/i64 `memref.atomic_rmw` only on actual private allocations through
standard views. The loop seed must also be independently synthesized. This
lets an atomic old value feed a loop accumulator without allocating a new
accumulator generation on every iteration. The original atomic, destination,
mask, exactly-once boundary, ordering and control position remain intact; only
the downstream pure value destination/copy changes. External or unknown atomic
roots, shared seeds, extra effects and opaque calls remain outside this profile.
Native alias and post-deallocation lifetime checks remain authoritative.
Closed integer generic atomic compare/exchange regions also qualify as such
snapshot dependencies. Their separate [native legalization](atomic_compare_exchange.md)
runs after destination reuse and retains a complete checked borrowing boundary
through ownership analysis, with every helper body restored before liveness.

Original payload operations, inputs, attributes and arithmetic order remain.
Direct retargeting changes only an unused destination. New copies use identity
maps and an unused init, preserve source IDs/location on the selected edge, and
introduce no arithmetic, predicate change or branch speculation. Native bufferization decides
whether physical storage can be reused, including conflicts with live inputs.
Ordinary deallocation and lifetime checks remain mandatory. The feature grants
no physical alias, effect, numeric approximation or target permission.

Destination timing follows original liveness. If the old carried value remains
in use after a producer, retain that producer and copy its complete value only
at the selected control edge, after those reads. Nonidentity carried-input maps
also require a complete computed value before any carried element is overwritten.
Ancestor inspection stops at the owning loop; mutually exclusive sibling
branches do not become later uses. This keeps coupled online-softmax states
executable without forcing fresh ownership on every iteration. Native alias
analysis may still refuse or introduce copies; lifetime checks remain enabled.

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
The same integration now also retains pointer/accumulator `while` state and
final pointer observations over changing runtime counts, including zero.

When atomic legalization is selected, destination reuse precedes its one
immediately pre-bufferization marker. Both feature application orders retain
that contiguous group: destination analysis, atomic preparation, bufferization
and ownership, then restored atomic helpers before liveness. Duplicate, moved
or multiple atomic seams refuse. The registered scalar pipeline and explicitly
composed pipelines share this ordering; no feature registration order can split
atomic preparation from its ownership suffix.
Nested pointer branches, pointer-returning helper inlining, `for`/`while` state,
and conditional row reductions now also execute through the actual Merlin
library path over independent empty/tail domains, choices and iteration counts.
Computed/permuted before-state and broader ownership profiles remain separate.
Tiled online attention additionally retains two contractions and running maximum,
normalizer and accumulator through one native library, including empty keys,
runtime tails and independently changing query/key dimensions. This is measured
finite-f32 numerical coverage, not all attention variants or device execution.

The upstream boundary is LLVM `a47bddccec30255619bb8c37fa59700e661d4e66`:
`mlir/lib/Dialect/SCF/Transforms/BufferizableOpInterfaceImpl.cpp` requires
equivalent before/after state or inserts a before-region allocation/copy.
`mlir/test/Dialect/SCF/one-shot-bufferize.mlir` exercises while equivalence,
non-equivalent conditions and mismatched state. The conditional read-only/fresh
case also appears in `Dialect/Bufferization/Transforms/one-shot-bufferize-allow-return-allocs.mlir`.
This feature exposes value destinations before that analysis; it does not
replace upstream alias handling or relax post-deallocation lifetime verification.

Complete model captures additionally exercise scalar reads of a preceding
computed model tensor feeding coupled integer loop state. Recursively checking
that readable value lets the existing destination/copy mechanism retain the
original recurrence without fresh ownership on every iteration. Two i32/i64
models preserve learned weights and all six results through actual native
execution, including empty calls; the i64 model also qualifies through the
public binding CLI. Sixteen
focused positive/refusal cases cover for/while state, views, nested producers,
constants/fills, init reads, effects, calls and library markers. The 107-case
destination suite passes with the native toolchain; post-deallocation and alias
checks remain enabled.
