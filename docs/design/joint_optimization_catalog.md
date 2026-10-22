# Architecture-aware compiler optimization guidance

This catalog supplies optimization hypotheses and compiler entrypoints. It does
not supply target schedules, tile sizes, reference kernels, private timings or
benchmark-specific answers. Capability facts and selected numerical/instruction
contracts determine applicability; an architecture label alone grants nothing.
The automatic author still discovers, checks and measures each candidate.

## Method and fairness

Record the exact catalog and compiler-library version in each experiment. This is
compiler-assisted exploration, so comparisons must disclose the assistance.
A later ablation can compare identical workloads, source/reference grants, model,
compute budget and timing evaluator with and without this catalog or editable
passes. Keep private reference results out of candidate ranking and fitting.
Do not call development-exposed workloads unseen validation.

The catalog may explain a mechanism, its legality premises, available analyses,
pass interfaces and tunable dimensions. It must not prescribe winning constants,
copy private implementation choices, provide golden outputs, or disclose private
comparative scores. Allowed source examples retain their separate explicit grants
and attribution. More guidance does not relax independent correctness checks.

## Tightly coupled, CPU-issued accelerators

The current Gemmini campaign selects this category. Interpret CPU, cache, DMA,
command queues and accelerator resources through the selected RTL facts. Rocket
is scalar and in-order; accelerator reservation/dependency queues are distinct.
All module names below are under `merlin.compile.scheduling` unless qualified.
Availability is not proof that a pass applies to an input or improves performance.

| Opportunity | Existing mechanisms to inspect | Choices the agent must discover | Legality and feedback |
| --- | --- | --- | --- |
| Command and dispatch amortization | `command_compaction`, `compact_dense`, `llvm_program.emit_index_program` | Counted-loop structure, work grouping, dispatch granularity | Actual emission, declared instruction policy, code size and host issue/completion costs; never unroll large shapes merely to simplify analysis. |
| Configuration reuse | `configuration.eliminate_redundant_configuration`, `configuration_loops.hoist_loop_configuration` | Hoist scope and invariant state | Decoded reads/writes/clobbers and loop invariants; preserve unknown state changes. |
| Load/compute/store overlap | `copy_pipeline.copy_pipeline_motion`, `schedule_pipeline.compile_schedule`, `Kernel` effects/stages | Prefetch distance, buffering, queue admission and pass order | RAW/WAR/WAW, consumption versus readiness versus safe reuse, bank/capacity limits, completion at host boundaries; two buffers plus per-operation waits do not prove pipelining. |
| Reuse and tiling | `block_schedule`, `compact_dense`, `dense_source` | Tile sizes, traversal, grouping and retained operands | Derive geometry from input/facts; include tails, padding, accumulator capacity, command overhead and spill traffic. |
| Placement and storage | `temporary_storage.allocate_temporary_storage`, `device_spills.allocate_with_spills` | Buffer sharing, bank placement, copy/spill decisions | Cross-layer live ranges, aliasing, actual emitted storage and asynchronous completion; planned lifetimes are not measured reuse. |
| Transfer layout and packing | `panel_transfers.plan_packed_panels`, `merlin.compile.host_optimizations`, shared host codegen | Panel layouts, layout propagation, preparation reuse | Alignment, stride, physical transfer width, source ownership and quantization; distinguish one-time weights from per-inference conversion. |
| Connected fusion | `fusion.plan_fusion`, `shared_input_groups`, `shared_input_schedule`, pointwise lowering | Producer-consumer groups, shared-input groups and legal epilogues | Preserve every live-out, reduction/rounding, side effects, aliases and state; compare spills and whole-chain timing with unfused execution. |
| CPU/accelerator handoffs | Shared host codegen, stream/arena planning, global implementation choices | Partitioning, retained intermediates, synchronization placement | Count conversions, normalization, packing, dispatch and waits in the reported execution scope; do not add overlapping engine busy times. |
| Shape policy | Existing pass manager, `compile_schedule` prerequisite/invalidation checks and target adapters | Alternative pass orders and shape/resource-based selection | General compiler rules and explicit unsupported regimes; no capsule-name or model-name dispatch. |

Each experiment should state the bottleneck hypothesis, changed passes/policies,
legality premises, source/result IR identity, relevant correctness subset, larger
shape result, timing epoch/scope/confidence and any regression. Prioritize measured
benefit and time to useful performance over broad pass coverage. The target backend
owns target legality and instruction mapping; generic host work belongs in Merlin.

## Other architecture categories

These are separate optional catalogs, not capabilities claimed for the current
Gemmini target or for a named external chip without its own facts.

| Capability category | Additional optimization hypotheses | Required compiler/model support |
| --- | --- | --- |
| Decoupled accelerator with local instruction memory and vector execution | Local program reuse; instruction-memory footprint; autonomous DMA/compute schedules; vectorization/masking; vector tails; local register/storage allocation; matrix/vector fusion; host launch amortization | Explicit program-memory limits and ISA, vector/reduction numerics, DMA/event semantics, target code-size and register/spill feedback, cross-engine dependency model. Existing generic passes may help; target adapters and missing passes must be implemented and checked. |
| SIMT-style execution | Thread/block mapping; coalescing; shared-memory tiling; bank conflicts; synchronization; register pressure/occupancy; divergence; asynchronous copies where supported | Actual lane/warp/workgroup contracts, memory consistency and barrier legality, register/shared-memory limits and a target-specific execution model. Gemmini queue or roofline assumptions do not transfer automatically. |

The user's Atlas-like and Muon-like examples motivate these categories; this
document does not establish their exact hardware features. Dispatch guidance by
verified capabilities rather than brand names. Expose only the relevant compact
menu to an author; retain the broader catalog for on-demand operator review.

## Current implementation evidence and remaining work

The v13 compiler constructs shared `Kernel`/`Loop` programs for dense, convolution
and pointwise paths. Dense lowering consumes `result.kernel` through
`emit_index_program`. Its sparse shared `compact_dense` change implements two-slot
buffering; the current adapter retains completion fences and does not yet establish
asynchronous overlap. Most cross-operation fusion families remain unavailable.
These are real integration steps, not demonstrated large-model performance wins.
The next continuation must expose working scoped cycle feedback, retain numerical
versus structural-versus-RTL distinctions, and prioritize useful schedule/fusion
variants. Final physical validation remains separate.

## Detailed mechanism inventory

Canonical machine-readable inventory: [joint_optimization_catalog.json](joint_optimization_catalog.json). The 45 entries below map mechanisms to actual repository symbols. **Available** means selected public-library access, not demonstrated applicability or speedup. **Extension** means operator integration is needed before author import. Repository paths in the JSON are relative; they are not host filesystem grants.

### Storage

| ID and mechanism | Compiler entrypoint | Access | Evidence and remaining work |
| --- | --- | --- | --- |
| `capacity_tiles`: Choose tile geometry from operand and accumulator capacities | `merlin.compile.capacity.capacity_fit` | Extension | Chosen tile, capacities, actual allocated bytes and edge work Remaining: Target geometry and allocation obligations must be supplied; fit is not performance selection |
| `resident_rhs`: Retain reusable RHS across output tiles | `merlin.compile.scheduling.dense_rhs_residency.plan_dense_rhs_residency` | Available | Loads eliminated versus decoded emitted loads Remaining: Plan alone grants no overlap or synchronization |
| `sibling_input_reuse`: Share an input stripe across sibling contractions | `merlin.compile.scheduling.shared_input_schedule.schedule_shared_input` | Available | Source identities, load bytes and live ranges Remaining: Needs target primitive and storage bindings |
| `temporary_reuse`: Reuse disjoint temporary lifetimes | `merlin.compile.scheduling.temporary_storage.allocate_temporary_storage` | Available | Peak live bytes, placements and emitted object offsets Remaining: Finite straight-line scope; loop lifetimes require separate proof |
| `device_spills`: Insert exact spills and reloads when local capacity is exceeded | `merlin.compile.scheduling.device_spills.allocate_with_spills` | Available | Spill bytes, inserted transfers and preserved numerics Remaining: Spill pricing and target transport adapter required |
| `operand_vs_accumulator_lifetimes`: Keep operand staging lifetime separate from partial-sum lifetime | `merlin.compile.scheduling.staged_dense.ComputeConsume` | Available | Independent operand and partial-sum live intervals Remaining: Target adapter must not release staging or accumulator storage on descriptor issue alone |

### Host Memory

| ID and mechanism | Compiler entrypoint | Access | Evidence and remaining work |
| --- | --- | --- | --- |
| `host_stack_arena`: Move oversized static stack temporaries to a module arena | `merlin.llvmlower.stack_arena.bind_stack_arena` | Available | Pre/post LLVM hashes and frame/arena bytes Remaining: Arena capacity is not runtime peak or cross-layer reuse |

### Packing

| ID and mechanism | Compiler entrypoint | Access | Evidence and remaining work |
| --- | --- | --- | --- |
| `preparation_cse`: Deduplicate equal preparation views and exact casts | `merlin.llvmlower.tensor_provenance_cse.deduplicate_tensor_preparation` | Extension | Preparation count and byte count before/after Remaining: Does not permit float reassociation or mutable storage reuse |
| `persistent_prepack`: Prepare fixed weights once across calls | `merlin.llvmlower.weight_prepack_runtime.fixed_arguments` | Extension | Preparation lifetime, setup cost and amortization horizon Remaining: Manifest weight role alone insufficient; invalidation adapter required |
| `borrowed_preparation`: Reuse prepared storage through checked borrow leases | `merlin.llvmlower.prepared_operand_schedule.install_prepared_borrows` | Available | Borrow scope, producer identity and overwritten-byte checks Remaining: Target layout and lifetime witness required |
| `direct_padding_gather`: Pack windows directly from logically padded inputs | `merlin.llvmlower.im2col_padding.rewrite_padding_gathers` | Extension | Removed intermediate bytes and packing instructions Remaining: Need physical buffer correspondence and measured host cost |
| `cold_warm_state_policy`: Separate one-time setup, cold invocation and warm steady-state cost | `merlin.llvmlower.weight_prepack_runtime.fixed_arguments` | Extension | Setup cost, first-call cost, repeated-call distribution, persistent bytes and invalidation events Remaining: Existing fixed-argument support is not a complete cold/warm benchmark or cache-state contract |

### Movement

| ID and mechanism | Compiler entrypoint | Access | Evidence and remaining work |
| --- | --- | --- | --- |
| `dma_coalescing`: Merge adjacent private copies | `merlin.compile.scheduling.copy_coalescing.coalesce_private_copies` | Available | Requested versus actual transfer count and bytes Remaining: No transfer legality inferred from adjacency alone |
| `strided_tile_loads`: Coalesce contiguous columns into strided local tiles | `merlin.compile.scheduling.tile_load_coalescing.TileLoadCoalescing` | Available | Decoded extents, stride and local overlap checks Remaining: Target encoding/capability binding required |
| `rhs_load_elimination`: Eliminate redundant resident RHS reloads | `merlin.compile.scheduling.staged_dense.eliminate_redundant_rhs_loads` | Available | Retained RHS epochs and emitted load counts Remaining: Invalidation and physical address proof required |

### Overlap

| ID and mechanism | Compiler entrypoint | Access | Evidence and remaining work |
| --- | --- | --- | --- |
| `copy_prefetch`: Prefetch private copies before their consuming work | `merlin.compile.scheduling.temporary_storage.prefetch_private_copies` | Available | Issue/completion ordering and time-window evidence Remaining: Unknown DMA dependency must remain blocked |
| `copy_pipeline`: Overlap transfer stages using existing private buffer slots | `merlin.compile.scheduling.copy_pipeline.pipeline_private_copies` | Available | Slot ownership, fences, queue occupancy and completed-call timing Remaining: Does not establish compute/load overlap without its own adapter |
| `double_buffer_ownership`: Express load/descriptor/consume/readout lifetimes explicitly | `merlin.compile.scheduling.staged_dense.rewrite_staged_dense` | Available | Planned versus emitted stages, live intervals and dependency waits Remaining: Target must realize legal alternating storage and stage ordering |
| `pipeline_phases`: Separate prologue, steady state and drain while rotating buffer ownership | `merlin.compile.scheduling.staged_dense.StagedDenseProblem` | Available | Stage count, slot identity, prologue/drain costs and decoded final completion Remaining: Existing staged representation is a foundation; generic modulo scheduler and emitted phase/rotation verifier remain an explicit implementation obligation |

### Scheduling

| ID and mechanism | Compiler entrypoint | Access | Evidence and remaining work |
| --- | --- | --- | --- |
| `tile_traversal`: Choose output-tile traversal to expose reuse | `merlin.compile.scheduling.staged_dense.order_output_tiles_n_before_m` | Available | Per-shape packing, traffic and whole-call cost Remaining: No universal traversal winner; shape/cost policy must choose |
| `block_order`: Schedule tile load/preload/compute/store order | `merlin.compile.scheduling.block_schedule.schedule_contraction` | Available | Schedule IR, physical mapping and decoded sequence Remaining: Schedule must lower through target adapter and completion validation |
| `convolution_tiles`: Tile convolution without a full materialized image matrix | `merlin.compile.scheduling.block_schedule.schedule_tiled_convolution` | Available | Window reuse, buffer sizes and packing cost Remaining: Host packing/accelerator gather tradeoff needs complete-cost feedback |

### Issue

| ID and mechanism | Compiler entrypoint | Access | Evidence and remaining work |
| --- | --- | --- | --- |
| `configuration_cse`: Remove redundant setup instructions | `merlin.compile.scheduling.configuration.eliminate_redundant_configuration` | Available | Config counts, state checks and issue cost Remaining: Candidate assertions alone cannot establish state equivalence |
| `dead_private_writes`: Delete unobserved private writes | `merlin.compile.scheduling.dead_writes.eliminate_dead_writes` | Available | Removed writes and original observable outputs Remaining: Observable stores, MMIO and fault semantics excluded without proof |
| `command_loops`: Compact static command streams into counted host loops | `merlin.compile.scheduling.command_compaction.compact_command_stream` | Available | IR/text size, loop bounds, emitted command equivalence Remaining: Code-size reduction not a dynamic command-count reduction |
| `encoded_reroll`: Recover periodic counted command nests | `merlin.sched.reroll.find_nests` | Available | Static bytes versus actual dynamic instructions Remaining: Loop-aware dynamic trace adapter needed for unresolved induction |
| `host_address_arithmetic`: Reduce repeated integer address work while preserving exact bounds and pointers | `merlin.sched.ir.expr.evaluate` | Available | Host integer instructions, pointer materialization and command-submission window Remaining: Expression representation exists; general emitted-host strength-reduction/hoisting policy and instruction-cost feedback not established by this API |

### Synchronization

| ID and mechanism | Compiler entrypoint | Access | Evidence and remaining work |
| --- | --- | --- | --- |
| `barrier_cleanup`: Propose redundant barrier/config cleanup | `merlin.compile.scheduling.effect_cleanup.cleanup_indices` | Available | Fence counts and retained dependency paths Remaining: Proposal is not execution authority; queue completion proof required |

### Host Compute

| ID and mechanism | Compiler entrypoint | Access | Evidence and remaining work |
| --- | --- | --- | --- |
| `scalar_loop_outline`: Outline large generated scalar LLVM loops | `merlin.llvmlower.llvm_loop_outline.outline_loops` | Extension | Compile latency, code size and runtime cost Remaining: May change compile latency and call overhead; must measure |
| `owned_scalar_host`: Lower scalar tensor regions using Merlin-owned host code | `merlin.llvmlower.host_tensor_codegen.HostLinalg` | Available | Host operation coverage and host/accelerator seam costs Remaining: Unsupported host signatures require narrow reviewed closure |

### Host Interface

| ID and mechanism | Compiler entrypoint | Access | Evidence and remaining work |
| --- | --- | --- | --- |
| `descriptor_transport`: Construct exact ranked-descriptor ABI transport | `merlin.llvmlower.host_descriptor_call.HostDescriptorTransport` | Extension | Descriptor overhead, copies and pointer/lifetime checks Remaining: No copy/ownership inference from descriptor construction |

### Fusion

| ID and mechanism | Compiler entrypoint | Access | Evidence and remaining work |
| --- | --- | --- | --- |
| `epilogue_fusion`: Fuse per-output epilogue into producing reduction loop | `merlin.llvmlower.epilogue_fusion.fusion_stage` | Extension | Intermediate bytes, launches and exact output comparison Remaining: Device fusion still needs selected numeric/effect capability |
| `connected_multioutput`: Compile connected pointwise composition with two live outputs | `merlin.compile.scheduling.multioutput_fusion.compile_source_multioutput` | Available | Source coverage, emitted regions and boundary tensors Remaining: Not general arbitrary graph fusion; target capability must support operations |
| `physical_composition`: Select an independently qualified physical composition | `merlin.compile.scheduling.fusion.plan_fusion` | Available | Original versus fused arithmetic and decoded instruction image Remaining: No capability inferred from matching names or approximate numerics |
| `independent_sibling_epilogues`: Retain independent output scale, activation, bias and readout state for each sibling | `merlin.compile.scheduling.shared_input_groups.find_shared_input_groups` | Available | Per-result epilogue provenance and decoded state before each readout Remaining: Grouping is only a source proposal; independent epilogue-state and clobber adapter still required |

### Numeric Formats

| ID and mechanism | Compiler entrypoint | Access | Evidence and remaining work |
| --- | --- | --- | --- |
| `requant_proof`: Synthesize/prove exact scale and output saturation | `merlin.llvmlower.requantization.synthesize_store_scale` | Available | Scale proof, overflow bounds and exact boundary values Remaining: Sample agreement is not full-domain proof; target rounding must match |
| `ordered_reductions`: Preserve seed and floating reduction order | `merlin.llvmlower.scalar_ordered_reduction.edit_pipeline` | Extension | Reduction order, accumulator width and exceptional behavior Remaining: Parallel reassociation requires separate numeric permission |
| `row_normalization`: Schedule chunked complete-domain softmax/normalization | `merlin.compile.scheduling.row_softmax.compile_row_softmax` | Available | Row working set, passes over data and numeric error contract Remaining: Target arithmetic adapter and final complete-domain checks required |
| `partial_sum_initialization`: Initialize partial sums once and preserve accumulate-versus-overwrite semantics | `merlin.compile.scheduling.staged_dense.AccumulatorZero` | Available | Initialization count, accumulation epochs, bias identity and tail checks Remaining: Initialization permission must be decoded; zero input data is not proof of an initialized accumulator |

### Legality

| ID and mechanism | Compiler entrypoint | Access | Evidence and remaining work |
| --- | --- | --- | --- |
| `sync_legality`: Check synchronization against machine-specific semantics | `merlin.sched.check.sync.check_sync` | Available | Happens-before paths, outstanding work and visibility boundary Remaining: Noncoherent host cache maintenance cannot be invented by a fence checker |
| `placement_legality`: Check live local operands against placement/clobber hazards | `merlin.sched.check.placement.check_placement` | Available | Address ranges, live operands and refused overlaps Remaining: Bank conflicts and throughput need separate performance model |
| `schedule_tokens`: Represent stages, dependencies and bounded counted loops | `merlin.sched.ir.kernel.check_structure` | Available | IR plus selected pass order and actual lowering receipt Remaining: Structural legality alone does not prove target semantics |

### Compiler Control

| ID and mechanism | Compiler entrypoint | Access | Evidence and remaining work |
| --- | --- | --- | --- |
| `checked_pipeline`: Choose pass order through final independently decoded bytes | `merlin.compile.scheduling.schedule_pipeline.compile_schedule` | Available | Transformed IR consumption and per-pass validation receipts Remaining: Applicable passes must execute; a registration list is not realization |

### Feedback

| ID and mechanism | Compiler entrypoint | Access | Evidence and remaining work |
| --- | --- | --- | --- |
| `cost_kind`: Distinguish exact/estimated/lower-bound schedule costs | `merlin.sched.cost.price` | Available | Scope/model hashes, uncertainty and non-additive resources Remaining: Queue waits, noncoherent cache traffic and host costs may be unknown |
| `compiler_observation`: Expose packing/lifetimes/fusion planned-versus-emitted evidence | `merlin.compile.compiler_observations.write_receipt` | Available | Bounded diagnostics with artifact references; UNKNOWN stays UNKNOWN Remaining: Runtime waits and savings absent until independently observed |

## Explicit gaps

- **host_cache_visibility**: Make CPU clean/invalidate/flush and DMA visibility part of owned execution ABI. Selected target cache-maintenance protocol and physical visibility witness; ordinary architectural fence alone does not establish noncoherent-cache correctness.
- **queue_pressure_feedback**: Choose legal issue distance and overlap using queue occupancy/backpressure. Independently validated occupancy/stall attribution; static counts do not measure queue waits.
- **shape_policy_selection**: Select tile/reuse/packing/pass variants using shape classes and complete cost. General policy trained/validated across connected source-derived cases; no kernel-specific magic constants or retrospective private reference tuning.

## Compact additional capability starters

These are design directions, not implemented target support or claims about a vendor product. Select traits from actual hardware and runtime contracts.

### Mixed Precision Fp4 Fp6 Fp8

- Define exact exponent/mantissa/bias/subnormal/NaN/rounding encodings before instruction selection.
- Choose tensor/channel/block scaling under explicit error bounds.
- Track scale ownership and metadata traffic with payload tiling.
- Pack/unpack non-byte-aligned FP4/FP6 without padding or endianness ambiguity.
- Select accumulation/widening and reduction order independently of storage format.
- Fuse casts only when exact or permitted by the source error contract.

Missing support: Integer requantization is not an FP4/6/8 numerical implementation; Need format-specific arithmetic/packing dialect adapters and exceptional-value reference contracts; Need metadata-aware bandwidth and conversion cost feedback.

### Vector Processing Unit

- Vectorize supported scalar regions under exact alias and numeric contracts.
- Tile by vector length and handle tails with proven masks.
- Choose gather/scatter versus contiguous layout conversion.
- Balance register pressure, unrolling and explicit spills.
- Preserve reduction order or bind permitted reassociation.
- Schedule vector/scalar transfers and avoid lane extraction churn.

Missing support: Need concrete scalable/fixed-vector dialect, target ABI and mask semantics; Existing generic scheduling does not establish VPU instruction or throughput support; Need lane utilization and vector-register spill observations.

### Asynchronous Runtime

- Bind buffers to explicit queue/event dependencies.
- Overlap copies and kernels only under memory-visibility guarantees.
- Amortize host launch overhead through legal batching or captured graphs.
- Manage reusable allocations and cross-kernel lifetimes.
- Choose stream/queue placement without assuming independent hardware resources.
- Separate launch latency, enqueue duration and completed execution timing.

Missing support: Need concrete runtime calls, context ownership and error/teardown adapter; Need pinned/pageable host-memory policy and transfer-cost measurement; No claim of existing ROCm binding or vendor-specific graph API support.

## Grow the catalog through exploration

This is an open inventory, not an optimization allowlist or required pass order. Record a new mechanism with capability predicates, a bottleneck hypothesis, legality premises, compiler entrypoints, changed passes, relevant numerical evidence, comparable timing, counterexamples and unknowns. Distill successful discoveries into generic passes or shape/resource policies. Preserve failed approaches with their tested scope. Never promote a planned optimization to a demonstrated speedup from a pass receipt alone.

The static symbol references and hashes were checked; this catalog introduces no target execution or new simulator qualification. Private implementation audits remain outside this inventory.

## Select guidance from the software/target specification

Use orthogonal capability fields rather than a chip-name category. The software
specification describes the execution stack; selected RTL facts and execution
contracts constrain which declarations are valid. Separate host (scalar/in-order),
submission (CPU-issued/instruction-issued), compute (systolic), numeric (integer
quantization/storage/accumulation), memory (scratchpad/accumulator/DMA/coherence),
scheduling (queues/dependencies/completion/overlap), and runtime (bare-metal or a
concrete runtime such as ROCm). ROCm is not selected for this bare-metal Gemmini run.

Use supported, unsupported, and unknown states. Offer a mechanism only when all
its hardware/runtime prerequisites are supported. Hide mismatches. Unknown
prerequisites become focused discovery questions, not optimization prescriptions.
Then filter by actual workload IR (shared inputs, convolution, repeated calls,
etc.) and available compiler APIs. Do not infer workload properties from hardware
tags or union whole category menus. Queue depth, memory capacities, transfer
granularity, visibility and numerical rules remain parameterized facts.

This policy is recorded in the catalog and continuation instructions. It does not
yet add a new enforced software-spec schema or automatic runtime menu selector.
The current continuation gets the selected CPU-issued mechanisms; mixed precision,
VPU and ROCm starter catalogs stay outside its active optimization menu.
