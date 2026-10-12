---
title: Capability roadmap for a verified general compiler
kind: design
status: current
owner: ir
last_verified: 2026-10-12
related: [verified_compiler_tooling, compiler_verification, lowering_pipeline, phase2_test_justification, component_compiler_convergence]
code_refs: [src/merlin/transforms, src/merlin/verify, src/merlin/llvmlower, src/merlin/targetgen]
---

# Capability roadmap for a verified general compiler

This roadmap separates the first delivered subset from remaining capabilities.
The [usage guide](../guides/verified_compiler_tooling.md) states the callable APIs,
experiment boundaries and proof scope. Existing shared and upstream components
can satisfy part of a row; a row is complete only when the ordinary compiler
actually invokes it and its declared assurance and feedback are exercised.

## Ownership and phases

- **model2MLIR:** faithful capture/export, frontend representation and quantization
  recipes. Exported semantics remain explicit through downstream transforms.
- **Merlin/upstream:** generic analyses and transformations, host code, packing,
  requantization, buffer ownership, dispatch and compilation orchestration.
- **Target OOT:** target instruction definitions, instruction selection, physical
  layouts, device allocation and hardware-specific scheduling policies.

Phase 0 resolves reviewed software declarations and hardware facts and derives
independent development tests. Phase 1 establishes a correct general compiler
for the admitted domain. Phase 2 changes its implementation and policies using
identity-bound evidence, while retaining Phase 1's exact inputs and tooling.

The first delivery implements static view/scalar normalization, ordered loop
segmentation, initialized local buffer reuse and conservative sequential
dependencies. It generates optional typed ISA definitions and checks modeled
obligations through Lean. General tensor bufferization, all numerical policies,
arbitrary effectful scheduling and whole-machine equivalence are not established
by this delivery.

## Remaining capability groups

The rows below are requirements and exploration surfaces, not a statement that
all listed optimizations are implemented. Phase 1 first needs a legal scalable
path; Phase 2 chooses profitable alternatives from measured bottlenecks.

| Group and passes | Owner | Phase | Verification obligation | Useful feedback |
| --- | --- | --- | --- | --- |
| Frontend/numerical normalization: shape/rank/index normalization, quantization boundaries, contraction/window/composite normalization, constant folding, CSE/DCE | model2MLIR + Merlin/upstream | P0 facts/tests; P1 semantic intake | Preserve exact dtypes, scales, rounding, overflow, initialization and source indices; isolate approved approximation changes | Typed operation coverage, unresolved forms, integerized work, numerical mismatches and source provenance |
| Shared analyses: dominance/use-def, alias/effects, dependence, shape/access/ranges, liveness, reuse/working sets and resource pressure | Merlin/upstream; OOT supplies constraints | P1 foundation; P2 recomputation | Analysis results must conservatively contain possible accesses, dependencies and lifetimes; unknown effects constrain rewrites | Reasons for unknowns, conflicting accesses, peak live bytes, repeated materializations and invalidated analyses |
| Whole-graph planning: legalization/placement, region formation/outlining, producer-consumer and epilogue fusion, layout/representation propagation, materialization elimination, graph scheduling | Merlin mechanisms + OOT capability/policy | P1 complete paths; P2 global optimization | Complete inputs/outputs, preserved representation, ownership, dependency and placement legality | Source-to-region ownership, host/device work, conversions, transfers, boundaries and connected-program costs |
| Loop/compute transforms: multilevel tiling, tails, strip mining, ordered reductions, interchange, vectorization, unrolling/rerolling, LICM, specialization/multiversioning | Shared machinery + OOT legality/policy | P1 arbitrary supported sizes; P2 tuning | Index coverage, exact carried state, safe bounds/overflow, valid guards and numerical-order permissions | Work/byte counts, utilization, code growth, index overhead, register/local-memory pressure and tail cost |
| Storage: bufferization, lifetime shortening, allocation hoisting, arena planning, buffer reuse, local resource allocation, spills/reloads/recomputation and bank/layout assignment | Merlin generic ownership; OOT device resources | P1 bounded legal storage; P2 locality | Alias and initialization discipline, nonoverlapping live assignments, capacity/alignment and spill value preservation | Peak storage, lifetimes, spill bytes, allocation frequency, bank conflicts where observable and compile memory |
| Movement/residency: transfer insertion/coalescing/batching, immutable preprocessing, copy/packing fusion, output materialization and cross-operation residency | Merlin mechanisms + OOT physical formats/transfers | P1 correct crossings; P2 reuse | Logical-to-physical index correspondence, valid windows, synchronization, visibility and complete readout | Unique/repeated bytes, transaction count, packing/setup costs, device round trips and measured reuse |
| Device lowering/scheduling: instruction selection, operand/address legalization, configuration-state management, issue scheduling, software pipelining, multiple buffering and synchronization simplification | OOT; reusable algorithms in Merlin | P1 legal instructions/state; P2 overlap | ISA/effect conformance, encoded ranges, resource hazards, state prerequisites and completion ordering | Executed instructions, setup commands, occupancy, stalls, observed overlap and legality failures |
| Host/device compilation: ABI/descriptors, generic host conversion/requantization, host loop optimization, dispatch consolidation/preparation hoisting, dependencies, build/link/cache and code/stack validation | Merlin/upstream + OOT execution interface | P1 complete execution; P2 overhead/scaling | Caller/callee contracts, ownership, representation, observable effects and identity-correct artifact reuse | Host work, dispatch/setup cycles, temporary bytes, stage times, cache hits, object/code size and stack pressure |

## Extend proofs without introducing an answer

Instruction declarations can mechanically generate signatures, local checks,
encodings and effect/resource records. They cannot establish which tensor
decomposition or schedule is profitable. Hardware declarations and semantic
primitives remain reviewed inputs; the candidate cannot redefine them to make a
proof succeed.

Extend formal coverage by adding a shared semantic model, a general theorem or
checked obligation, structural projection of actual emitted IR and negative
controls. Keep data-dependent aliasing, floating approximation, asynchronous
protocols and target state explicitly outside coverage until those requirements
are met. Host memory/LLVM/RTL execution still needs independent conformance.

CPU register allocation remains upstream LLVM work. Local accelerator storage,
register banks and queue allocation stay in the OOT. Shared liveness and
dependency algorithms can serve both with distinct admitted resource models.

## Require evidence before promotion

For each implemented transformation record its semantic preconditions, actual
before/after identities, affected interfaces, invalidated analyses and declared
proof scope. Development tests exercise the specific property it changes:
isolated arithmetic/encoding, tails and reductions, connected regions, resource
boundaries or whole-program composition. Public workload-derived tests retain
their source meaning and never substitute convenient shapes for unresolved work.

Use cheap structural checks, static bounds and qualified estimates to screen
candidates. Use numerical execution and identity-matched timing at the required
fidelity to accept improvements. An analytical prediction remains a prediction;
roofline distance does not establish correctness or achievable whole-program
speed. Track positive and negative outcomes with compiler and input identities.

Selective development retesting follows changed pass dependencies and interfaces.
Unknown impact requires broader checks. Complete promotion gates remain required,
and an immutable compiler revision owns each in-flight job. This allows parallel
exploration without editing the compiler being evaluated or copying entire
datasets and build trees per candidate.

Completion means a general invoked capability with its declared gates, rather
than a capsule count or a pass name. Fresh Phase 1 success, calibrated Phase 2
ranking and final independent whole-program execution/performance remain
separate experimental results.
