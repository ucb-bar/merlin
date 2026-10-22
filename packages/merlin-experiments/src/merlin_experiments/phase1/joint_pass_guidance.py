"""Author obligations for the selected shared APIs, never a pass-use verdict."""


def prompt():
    return """

## Performance-driven compiler exploration

Strong performance across the admitted shape families is the objective once
correctness and legal execution are established. Pass counts, dialect coverage,
buffer counts and instruction counts are not the score. Keep a working baseline
and compare a few hypothesis-driven variants on relevant small checks and a
representative large Spike case under one timing epoch. Integrate wins as reusable
passes or shape/resource-derived heuristics; do not branch on capsule/model names.
A library edit inside the grant needs no operator transition. Spend effort on the
current dominant cost rather than implementing every available transformation.

For CPU-issued, tightly coupled accelerators, consider this mechanism menu only
when selected facts and observations support the premise:

| Suspected cost | Compiler/schedule options | Evidence needed |
| --- | --- | --- |
| CPU issue/configuration overhead | `configuration` elimination, `configuration_loops` hoisting, `command_compaction`, compact `Kernel` loops | Actual emitted commands/host work and complete-call cycles; keep unknown state clobbers. |
| Serialized transfers/compute | `copy_pipeline`, `compile_schedule` pass order, legal queue interleaving and buffer lifetimes | Dependencies, slot capacity, completion and same-epoch timing. Two buffers with per-operation fences do not demonstrate overlap. |
| Repeated operand traffic or spills | `compact_dense` blocking/traversal, `temporary_storage`, `device_spills` | Actual placement, live ranges, requested traffic, padding and a large case in the relevant capacity regime. |
| Boundary/packing overhead | `panel_transfers`, shared host codegen, `fusion`, `shared_input_schedule` | Numerical/layout/alias premises, complete live-outs, one-time versus per-call costs and whole-chain timing. |

This menu supplies hypotheses and API locations, not selected schedules or winning
parameters. Discover choices from typed IR, hardware facts and your own evaluations.
Preserve an independent serial/unfused alternative where useful. Report numerical
status, structural evidence and withheld RTL coverage separately; withheld gSIM is
not an instruction to simulate the whole corpus. Unknown timing is not zero or a
performance win: save the reproducer and continue independently measurable work.

## Model deployment and numerical boundaries

Maximize accelerator coverage over ALL operations and connected regions admitted
by the selected hardware facts and numerical contract, not only contractions.
Inspect standalone and fused forms separately: residual/elementwise addition,
bias, scaling/requantization, activation, pooling and movement may be available
through different legal device forms. A readout-only epilogue capability does
not imply a standalone instruction. Derive eligibility from actual operand and
result types, shape/layout, numeric order and selected capabilities; an operator
name alone is insufficient. Preserve every live output when grouping or fusing.

Do not silently send a supported legal domain to the CPU because its compiler
adapter is unimplemented, inconvenient, or has an unmeasured cost. Hardware
unsupported, compiler unimplemented, numerically invalid and unresolved proof
are different outcomes. Missing target lowering is a compiler task. Profitability
guides device grouping, schedules and fusion choices; it is not blanket permission
to evade required accelerator coverage. Initially establish reusable lowering for
the supported domain. Subsequently compare legal CPU/device placements under the
same numeric and complete-call cost scope; a measured or supported-model CPU win
may justify CPU placement where the selected admission policy permits that choice.
If a current mandatory device gate prevents that comparison, report the precise
policy limitation for review; do not silently bypass it or label the hardware
unsupported. Retain real hardware-unsupported floating or nonlinear/control seams
through Merlin host codegen with explicit evidence.

Maintain source-bound placement diagnostics: source region/node IDs, standalone
versus fused-only capability, selected device form, host reason and remaining
compiler/proof gap. Report required/covered/eligible-but-host/unknown counts and
static boundary bytes separately from measured host time and transferred bytes.
Unknown measurements stay unknown. The fast model-check contraction census is
only a contraction census, not evidence that every supported operator is offloaded.
Inspect the native complete source-demand report for all-region obligations.

The selected deployment for ResNet50, TinyLlama and SmolVLA is W8A8:
signed INT8 weights and activations at accelerator contractions, with the selected
integer accumulation and output ABI. This does not require every intermediate or
model output to be INT8. Keep raw i32 outputs when the typed source requires them.
Preserve captured floating-point normalization, nonlinear functions, scaling and
control through Merlin host lowering; do not replace accelerator contractions with
CPU fallback or convert an entire mixed graph to integer arithmetic.

Use the selected public `target_contract.yaml`, typed source and numeric policies
as the authority. Its scaled i8 readout contract specifies f32 scale, half-even
rounding and saturation to [-128, 127]. That contract does not make arbitrary host
casts, raw accumulator stores or other source rounding operations equivalent.
Track operand/output dtype, scale identity and granularity, zero points, bias and
activation order, rounding, saturation, accumulation bounds and host/device
placement at each graph boundary. Fusion, packing and scheduling must preserve
these properties; never manufacture scales or loosen comparison tolerances.

The existing visible model-check endpoint accepts `resnet50`, `tiny_llama` and
`smolvla`: `python agent_model_check.py check --model MODEL`. ResNet has `model`;
TinyLlama has `prefill` and `decode`; SmolVLA has `prefix_encode`, `flow_denoise`
and `action_decode`, including its declared recurrent state. Early fast checks
replay selected kernels; they do not establish full-chain numerical correctness.
Use unchanged independent numerical contracts for capsule, connected and final
model checks. A format-compatible external signature alone does not establish
model calibration, boundary semantics or numerical equivalence. Missing evidence
stays unknown. The compiler must generalize through these source contracts, not
model-name dispatch.

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

The selected baseline inventory gives inspectable module paths, signatures and
source hashes. When the editable generic fork is selected, improve its pass
implementations alongside target adapters and policy in your submission. The
following are available mechanisms, with independent
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

For modules in the selected editable fork, implement generic improvements directly
in `submission/compiler_library/`; each evaluation snapshots those edits together
with your backend. You need no operator epoch for an ordinary edit inside that
grant. Keep useful failure examples and independent checks with the change.
For a new shared module or protected implementation outside the editable grant,
put a patch and applicability example in `shared-library-proposals/` and continue
independent work while the operator reviews the extension. The operator can
request an epoch stop at the next native author turn boundary, drain owned jobs,
freeze your current compiler, prepare a new versioned library and start a new
native run with `--joint-epoch REVIEW_JSON --fresh-backend`. That continuation
retains your authored compiler, original empty origin, original 12-hour clock
and aggregate gSIM ledger. It grants no historical certificate. Never edit the
immutable baseline in place or import a prior backend to implement a proposal.
"""
