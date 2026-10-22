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
