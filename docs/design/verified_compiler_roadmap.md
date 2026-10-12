---
title: Capability roadmap for a verified general compiler
kind: design
status: current
owner: ir
last_verified: 2026-10-12
related: [verified_compiler_tooling, compiler_verification, lowering_pipeline, phase2_test_justification, component_compiler_convergence]
code_refs: [src/merlin/transforms, src/merlin/verify, src/merlin/llvmlower, src/merlin/targetgen, src/merlin/perf, packages/merlin-experiments/src/merlin_experiments/phase1/brokers/common_transforms.py]
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

The common substrate supplies static view/scalar normalization, ordered loop
segmentation, initialized local buffer reuse and conservative sequential
dependencies. Version 2 additionally supplies guarded scalar CSE/DCE and LICM,
bounded ordered unrolling, actual IR analyses and an explicit resource allocator.
These capabilities have different invocation and assurance boundaries below.
Their presence does not prove whole-model compilation or performance.

## What actually runs

The ordinary shared host compiler and the common authoring service are distinct
routes. A candidate can request the service and compile its returned standard IR;
`lower_model` does not silently insert that service or its allocator into every
host or OOT compilation. Reuse the already invoked upstream passes rather than
adding another implementation with the same name.

| Route and selection | Actual invoked implementation | Established scope and remaining qualification |
| --- | --- | --- |
| Selected logical host layout and dispatch fusion | An explicit source-pinned `GraphOptimizationContract` selects `layout_propagation.rewrite_module` and/or the existing outlined logical-host fusion planner/emitter before ordinary `lower_model` preprocessing. | The first selected domain is static pointwise tensors and permutations. Typed interfaces, ordered scalar regions, symbolic logical coordinates and expanded actual-source correspondence are checked independently. Unsupported forms refuse; byte/dispatch savings and code-growth limits are structural cost proxies. This route supplies no Lean, physical representation, hardware or timing proof. |
| Ordinary host preparation and scalar lowering | `llvmlower.lower.lower_model` calls `passes_xdsl.preprocess_text` and `pipeline.lower_to_llvm_ir`. The default upstream pass list runs canonicalization/CSE, elementwise fusion and named-op generalization, one-shot bufferization, out-parameter conversion, ownership-based deallocation, allocation/loop hoisting, linalg-to-loops, SCF-to-CF, strided metadata expansion and LLVM conversion/translation. | These are existing executed passes, not missing compiler mechanisms. Numerical execution and ordinary build gates remain their assurance; the new Lean obligations do not prove this complete pipeline. An upstream pass being scheduled does not establish a profitable rewrite for every input. |
| Selected CPU vector or multicore lowering | `llvmlower.pipeline.lower_to_llvm_ir` selects `build_rvv_pipeline` and its transform schedule for the supported CPU vector path, or `_parallel_pipeline` for scalar OpenMP; serial lowering remains the default. | Existing host scheduling/vectorization machinery is reusable where the selected host ISA/runtime permits it. It does not generate an accelerator schedule, cover every CPU, or inherit a complete transformation theorem. |
| Ordinary host object generation and linking | `llvmlower.lower.lower_model` uses `codegen.compile_ll` for selected objects or `codegen.build_host_shared` for a host library, including the shared MLIR C runtime. Wide C-interface boundary repair runs before code generation. | Stock LLVM owns CPU instruction selection, physical register allocation and its selected optimizations. Do not duplicate its register allocator in the OOT. Device artifact lowering, target ABI details and final deployment remain separate obligations. |
| Existing optional host lifetime/arena optimization | The `sink-deallocs` option inserts canonicalization followed by upstream `optimize-allocation-liveness`, with a structural placement check. `lower_model(static_arena=True)` calls `arena_bind.bind_arena` after LLVM translation; that route reuses `xdsl_dialects.lowering.arena_plan.pack_disjoint`. | Both are opt-in, not defaults. The binder checks the admitted allocation/free, escape and CFG conditions and conflicting placement; unsupported cases remain unbound or refused. It is not the new SSA resource allocator and is not covered by the new Lean lifetime receipt. It does not establish asynchronous device residency. |
| Host-owned common transformation request | `CommonTransformService.apply` invokes `transforms.pipeline.apply_common_pipeline` on the request's actual standard xDSL IR. Default selectors enable supported shape/integer simplification and initialized buffer reuse; loop segmentation, dependency permutations and version-2 scalar redundancy/LICM/unrolling require explicit options. | Changed covered transforms require actual host-issued receipts bound to before/after IR. Structural correspondence is trusted; Lean checks modeled bit-vector equality, pure scalar reuse/speculation, ordered loop arithmetic/folds and disjoint lifetime obligations. Opaque effectful schedule changes without a supported semantic model refuse proof. This is not a full SCF/memory/LLVM/RTL theorem. |
| Version-2 host pass observations | The common request automatically invokes `perf.pass_observation.observe_changes`, recomputing storage/dependency planning and `analyses.analyze_ir` on connected actual before/after IR with the reviewed index width. | Reports static operations/loops, known integer allocation bytes, reservation plans, dependency edges, unresolved aliases and block-scoped logical liveness/pressure. It does not run resource allocation. Dynamic work, physical traffic, cycles and capsule impact remain unknown; these observations have no ranking authority. |
| Version-2 callable structural analyses | An OOT or shared consumer explicitly calls `transforms.analyses.analyze_ir` and can validate its source-bound result with `validate_analysis`. | Supplies conservative alias/effect/dependency facts, scoped captured uses/liveness and known logical footprint/pressure subtotals. Region/CFG memory arguments remain unknown until forwarding is proved. No full CFG, access-range, cache, physical representation or asynchronous lifetime claim is made. |
| Version-2 callable instruction resource allocation | An OOT explicitly supplies `ResourceBinding`, banks/units and `InstructionConstraints` to `allocate_instruction_resources`; `emit_resource_allocation` emits typed assignments and `validate_allocated_ir` independently checks those actual recorded assignments using the existing checked allocator. | Sequential single-result SSA and result-free state events are supported. Unknown effects, pointer/dynamic/opaque representations, nested control flow and simultaneous results remain refusals. The annotation is not terminal operand lowering, automatic host arena planning, a Lean receipt or proof that the supplied constraints match hardware. |
| Optional ISA and interactive proof authoring | The minimal reviewed ISA can drive the generic definition/encoding/local-verifier generator in the EL5 treatment. EL6 additionally admits interactive Lean authoring. Automatic common verification is host-owned for every EL. | These tools generate declarations or check scoped obligations; they do not choose tiling, placement, layouts or schedules. The target OOT must author its compiler policies. Phase 2 inherits the admitted Phase 1 source/tool profile exactly; it cannot add a treatment tool after the fact. |

Versioned public source membership makes a callable helper available and pins its
identity; it does not establish that the candidate used it. Before claiming an
OOT capability complete, inspect its actual compilation path, emitted operands
and independent numerical/hardware checks. Before claiming an optimization
complete, also establish its measured effect on the relevant original tests.

## Remaining capability groups

The rows below describe the full capability requirements and their remaining
qualification, not a backlog of missing pass implementations. The existing
routes above already supply portions of several rows. Phase 1 first needs a
legal scalable target compiler; Phase 2 chooses profitable alternatives from
measured bottlenecks. Target constraints and policies remain OOT authoring work.

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
