"""Author obligations for the selected shared APIs, never a pass-use verdict."""


def prompt():
    return """

## Scheduling integration and shared-pass development

Use the exported scheduling IR in the production compiler: `merlin.sched.ir.kernel`
provides Kernel, Call, Loop and explicit dependency/stage fields; `ir.expr` supplies
symbolic indices and `ir.parse.parse_kernel` reopens retained schedules. Do not
replace that integration with an unused scheduling import or a side demonstration.
Keep symbolic counted loops for large problems; the finite `compile_schedule`
route is for bounded instruction regions, not unrolling a full large tensor.

Represent buffer ownership and lifetimes in that shared scheduling IR: host input,
DMA destination, resident compute operands, accumulator and output each need an
identified owner, read/write effects, completion and safe reuse boundary. Construct
legal load/compute/store interleaving across tiles, with prologue, steady state,
epilogue and tails. Double buffering and prefetch may overlap independent transfers
with compute only while their storage slots, banks, command resources and hardware
dependencies permit it. Preserve preload/compute accumulator dependencies, drain
stores before host consumption and fence before reuse where required. Model the
accelerator's actual command queues and dependency tracking; a CPU reorder buffer
is not a substitute. If the timing model rejects a legal overlap, retain the exact
schedule and rejection as an unsupported-model report for operator review; do not
silently serialize the intended production schedule or invent an overlap estimate.
Keep a legal serial schedule available for numerical diagnosis and comparison.

The exact readonly source inventory above is the API authority. Its module paths,
signatures and source hashes are inspectable. Implement target adapters and policy
in your editable compiler. The following are available mechanisms, with independent
legality requirements; they are not a mandatory sequence of all passes:

| Layer / entrypoints | Applicability and tunable choices |
| --- | --- |
| `block_schedule.schedule_contraction`, `schedule_tiled_convolution` | Declared contraction/convolution semantics and geometry; tune tile dimensions, loop order, operand residency, load grouping, lookahead and placement through Geometry/Knobs; retain tails and reduction order. |
| `dense_source.schedule_source_dense`, `compact_dense.schedule_compact_dense`, `llvm_program.emit_index_program` | Recognized typed sources and admitted index/encoding adapter; keep large nests compact and derive dimensions from the input. |
| `shared_input_groups.find_shared_input_groups`, `shared_input_schedule.schedule_shared_input` | Actual shared operands and legal ordered reductions; choose groups/tiles under storage bounds, verify with `shared_input_verification`. |
| `panel_transfers.plan_packed_panels` | Actual source layouts, bounds, transfer widths and physical limits; tune packed panels and coalescing without assuming contiguous physical storage. |
| `schedule_pipeline.compile_schedule` | Finite Kernel plus effect/storage adapters, actual-byte decoder and resource contracts; use ScheduleBudgets and pass_order with validated prerequisites, current context stamps and bounded repair. |
| `temporary_storage.allocate_temporary_storage`, `device_spills.allocate_with_spills` | Private ownership, alias lifetimes, storage banks and admitted spill classes; choose allocation, copy elimination and spill limits, preserving completion. |
| `copy_pipeline.copy_pipeline_motion` | Explicit independent resources, in-flight limits, slot lifetimes and wait/completion facts; tune CopyPipelinePolicy and prefetch distance, preserving a legal serial fallback. Double buffering requires two actually nonoverlapping live slots. |
| `fusion.plan_fusion`, `pointwise_source.compile_source_pointwise` | Exact scalar/reduction correspondence plus effects and resource bounds; select legal fusion groups and bounded recomputation rather than model-name rules. |
| `configuration.eliminate_redundant_configuration`, `configuration_loops.hoist_loop_configuration` | Decoded reads/writes/clobbers and loop invariants; preserve unknown state changes. |
| `staged_dense.rewrite_staged_dense`, `command_compaction.compact_command_stream` | Admitted staged operand/descriptor contracts; select reuse, legal output-tile ordering and command compaction with their independent checks. |
| `program_limits.finalize_program`, `row_softmax.compile_source_rows` | Actual encoded bounds / continuation contracts; row numerics, ordering and storage remain explicit. Unsupported row arithmetic uses an admitted host route. |
| `host_optimizations`, `host_tensor_codegen.HostLinalg`, `host_epilogue.HostEpilogueEmitter`, `requantization` | Reuse shared host codegen and exact numerical policies. Tune graph partitioning, packing reuse and host/device handoffs; do not regenerate generic host arithmetic case by case. |

Maintain an input-derived pass applicability record for every compiler invocation:
family, legal premises, applied/no-match/unsupported/unknown, precise reason,
selected policy and actual source/result schedule digests. Unsupported means a
specific absent capability; unknown means an unresolved implementation or proof.
Do not declare everything unsupported to avoid implementing scheduling. Prioritize
the applicable mechanisms identified by the first small and large Spike results.

Persist the real Kernel text and the library's `merlin.schedule_compilation.v1`
report when using compile_schedule, including pass records, library/context
identity, resources, final schedule/image digests and completion checks. Feed its
actual result into emission. The host's V2 target transport receipt proves file
custody only; your JSON claims do not independently prove optimization use or
semantic equivalence. Do not report scheduling qualification until an operator
checks the selected schedule against actual emitted operands/completions. A valid
no-match is acceptable; an ignored transformed result is not. Numerical checks
and performance evidence continue while that qualification remains pending.

Generic improvements are permitted through a reviewed library epoch. Put a patch,
its source revision, changed API/policy, applicability, a small failing example
and an independent check in your submission's `shared-library-proposals/`.
Continue target compiler work while the operator reviews it. The operator can
request an epoch stop at the next native author turn boundary, drain owned jobs,
freeze your current compiler, prepare a new versioned library and start a new
native run with `--joint-epoch REVIEW_JSON --fresh-backend`. That continuation
retains your authored compiler, original empty origin, original 12-hour clock
and aggregate gSIM ledger. It grants no historical certificate. Never edit the
mounted library in place or import a prior backend to implement a proposal.
"""
