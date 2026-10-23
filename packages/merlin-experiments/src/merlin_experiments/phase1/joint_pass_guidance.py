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

## Construct capabilities for the whole public corpus

Read PUBLIC_CAPABILITIES.md once, then query PUBLIC_CAPABILITIES.json only for
relevant families and members. Use all public operations, shapes, layouts,
numerical boundaries and connected patterns to plan ONE general compiler.
Capsule IDs identify tests and provenance; never use them as production dispatch.
Assign capability families across compiler boundaries: graph/pattern legality,
numeric/layout, schedule/storage, ISA lowering, host/runtime, and integration.
Agree on typed contracts and disjoint source ownership; do not create a compiler
per capsule. Readiness gates admit only the relevant tests, never stop independent
construction. Build reusable capabilities rather than repairing cases independently. Validate
each new capability through one tiny emission/target-consumption/link/Spike/output
canary, then admit its eligible cases to the rolling queue while construction
continues elsewhere. Do not wait for the entire compiler before first execution.

Original correctness, calibration prerequisites and scalability can advance
concurrently. Preserve every declared Spike/gSIM obligation; performance effort
primarily targets the scalability cohort once executable. Small original cases
remain regression checks and model evidence, not the main performance objective.
Use compact family-level feedback and retrieve details on demand. For reusable
construction/check sequences use the staged `construction_actions.py` with explicit
commands, semantic capability, boundary and local dependencies. Automatic and
offline studies execute the same client and self-check broker; its command success
is not numerical certification. Keep source/request identities in the action log.
Prefer a bounded
machine-run search over legal pass choices to one agent turn per parameter trial.
If exported in the selected library, `pipeline_policy.evaluate_variants` evaluates
applicable policies on separate IR clones and returns transformed artifacts and
receipts. Feed those artifacts into real emission and independent checks; a search
receipt is neither a correctness certificate nor a timing measurement. When the
selected version provides `host_pass_profile`, inspect action_wall_ns separately
from framework_wall_ns to find expensive transformations versus orchestration.
These are host pass costs, not target cycles or capsule numerical evidence.

## One evolving compiler, parallel bounded work

Act as one compiler coordinator and delegate bounded, independent tasks to the
available parallel workers. Choose tasks from the current bottlenecks: frontend
operator coverage, connected-region legality and fusion, scheduling/pass policy,
shared host boundaries, or focused correctness/performance diagnosis. Every worker
contributes to the SAME reusable compiler; do not create separate benchmark backends.
Use the supported isolated lightweight git/task mechanism for edits, with narrow
file ownership and clear commits, rather than copying the whole compiler per worker.
The coordinator owns integration, compiler health and the next useful experiment.
Integrate compatible completed changes promptly; keep conflicting variants separate
until a focused same-scope comparison supports a choice.

Submit focused checks asynchronously against immutable compiler revisions. Continue
independent implementation while they run; neither the coordinator nor all workers
must wait for source quiet or each other's tests. A result belongs to its recorded
compiler/input/runtime hashes, never automatically to newer edits. Name the revision,
focused checks, first actionable failure, numerical/performance/model confidence and
artifact paths in each short handoff. Reuse an existing equivalent request/result
only when those relevant identities match. Avoid full-cohort checks on every edit;
reserve them for declared milestones/final validation. Report untested coverage.
All 20 selected CPUs are dynamically shared through the existing pool: no fixed
CPU assignments to author, compilation, grading, or calibration roles. Available
worker count is a concurrency ceiling, not a requirement to launch redundant work.
A protected evaluator/library upgrade may request a checkpoint; ordinary compiler
and editable-pass development continues while immutable evaluations run.

## Model deployment and numerical boundaries

Maximize accelerator coverage over ALL operations and connected regions admitted
by the selected hardware facts and numerical contract, not only contractions.
Inspect standalone and fused forms separately: residual/elementwise addition,
bias, scaling/requantization, activation, pooling and movement may be available
through different legal device forms. Every must-accelerate supported region needs
a legal fused OR unfused device path; a failed fusion attempt does not justify
silent CPU substitution. A readout-only epilogue capability does
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

Try the exported scheduling IR where useful in the production compiler:
`merlin.sched.ir.kernel` provides Kernel, Call, Loop and explicit dependency/stage
fields; `ir.expr` supplies symbolic indices and `ir.parse.parse_kernel` reopens
retained schedules. These are useful abstractions, not a dialect-use score or a
requirement to run every pass. Honor the selected shared-pass contract; an unused
import or a side demonstration does not establish production integration.
Keep symbolic counted loops for large problems; the finite `compile_schedule`
route is for bounded instruction regions, not unrolling a full large tensor.

Represent buffer ownership and lifetimes explicitly in the chosen scheduling IR: host input,
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

Use a source-to-realization fusion workflow where applicable. Distinguish adjacent
pointwise fusion, contraction epilogues, shared-input siblings, connected
producer/consumer residency, and preparation reuse. Their legality is different:
source adjacency alone proves neither scalar equivalence nor buffer lifetime.
Preserve all live-outs and independent output/epilogue state, including observers
outside the proposed region. Keep a correct unfused alternative. Record source IDs,
numeric/capability pins, legality premises, storage/lifetimes, selected schedule,
and applied/no-match/unsupported/unknown; do not call an unimplemented route
hardware unsupported. Compare complete-region costs, including boundaries, only
when scope and numerical contracts match; unknown model cost is not a gain.

If `merlin.compile.fusion_workflow` is present in the selected public API inventory,
`discover` records these source opportunities and native fanout-reader refusals.
`realize_adjacent` returns the checked Kernel and remaining temporary objects;
feed both to ordinary allocation/emission. `realize_shared_input` returns actual
logical effects with independent sibling outputs; target encoding/effect checks
remain required. `realize_multioutput` returns the existing guarded pointwise
compilation images. Consume returned products, not just receipts. Counted loops
stay on the compact route; finite fusion never unrolls them. Source epilogue proofs
still use `output_fusion.decisions`; general connected residency is not implemented
by this facade. Use existing `compiler_observations.install_prepared_observed` and
`lifetimes` for preparation installation and source lifetime evidence. Save pinned
connected_fusion sections with `compiler_observations.write_receipt` for the
existing diagnostic feedback path; planned/checked schedules do not prove runtime
traffic reduction or complete numerical execution. If the optional facade is not
exported, use its listed native APIs from the actual inventory; no protected import.

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
