"""Author obligations for the selected shared APIs, never a pass-use verdict."""


def prompt():
    return """

## Performance-driven compiler exploration

Strong performance across the admitted shape families is part of the work from
the start, not a phase after correctness: build on the scheduling representation
early and measure what each change does to cycles. Correctness and legal execution
still decide whether a variant is accepted. Pass counts, dialect coverage,
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
not an instruction to simulate the whole corpus. Unknown complete timing is not zero
or a win: decide on the case's calibrated fast estimate meanwhile, and revert if a
complete estimate or measurement later disagrees.

## Construct capabilities for the whole public corpus

Read PUBLIC_CAPABILITIES.md once, then query PUBLIC_CAPABILITIES.json only for
relevant families and members. Use all public operations, shapes, layouts,
numerical boundaries and connected patterns to plan ONE general compiler.
Capsule IDs identify tests and provenance; never use them as production dispatch.
Before construction, make one compact capability DAG covering ALL declared public
families, with an owner, input/output contract, local dependencies and a fallback
or explicit missing capability for each node. Use these compiler boundaries:

- Frontend/source graphs and legal fusion -> typed semantic interface with numerical
  order, layouts, aliases and every live output -> reusable target lowering.
- Public RTL/ISA declarations -> generic ISA generation and typed operations -> the
  same target lowering. Generate neutral scaffolding inside the empty backend;
  do not import an existing target compiler. Encoding alone proves no memory effects,
  completion, legal fusion or instruction numerical semantics.
- Source ownership and live buffers -> Merlin host codegen/dispatch -> complete
  host/device execution ABI joined to target lowering.
- Shared scheduling IR, transformations and pass policy connect these layers: tune
  pass order and branches from source shapes, layouts, numeric contracts and resources;
  consume transformed products in actual emission rather than logging unused calls.

Construct independent nodes concurrently. Do not develop the compiler one capsule
at a time or wait for one family to pass before implementing another. Agree on typed
contracts and disjoint source ownership; every helper contributes to ONE compiler.
When CONSTRUCTION_PLAN.example.json is staged, adapt it after mandatory discovery
into submission/CONSTRUCTION_PLAN.json (and each independently tested variant).
Preserve its inventory_sha256 and complete family roster; assign real owners,
planned module paths and local depends_on edges, with integration_node depending
transitively on all capability nodes. Planned modules need not exist yet. This is
construction ownership, not evidence of implementation or a correctness certificate.
If the selected policy requires the plan, ordinary broad simulation requests check
it; editing, compile diagnostics and the policy's bounded tiny bootstrap remain
available. Missing code or a failed capsule never globally stops construction.
Existing construction_actions validates local action dependencies when used.

Write executable entrypoints early and prove one tiny end-to-end
emission/target-consumption/link/Spike/all-output path. Then immediately submit broad
eligible sets from the entire public roster through the rolling queue; retain
unimplemented families as active construction tasks. Further tiny canaries are for
changed boundaries, not a mandatory serial check for every case. Fix shared ABI
failures before repeating affected work; independent capabilities keep progressing.
Consume each result as it arrives, without waiting for a batch or source quiet.
Inspect the returned public source -> target IR -> LLVM/object artifacts and their
complete pointer/output ABI, actual commands, loops, readout modes and completion.
Trace the first incorrect boundary; a successful import, pass receipt or object
build alone does not establish executable semantics.

The goal is ALL cases in the current selected public inventory complete AND strong
performance. Use PUBLIC_CAPABILITIES.json and the launch contract for current counts,
roles and required tiers, including admitted supplements; historical corpus sizes
are not coverage limits. Correct declared refusals remain refusals.
There is no whole-correctness-cohort barrier before calibration or large-shape work.
Preserve each original Spike/gSIM obligation and use the normal calibration policy.
Begin scalable implementation and legal optimization during construction; use the
selected provisional model with its limits, then the selected tuned model when
available. Optimize chiefly the selected primary performance roles and connected
workloads; small correctness cases have low performance priority and supply
regression/calibration evidence. Prioritize broad Spike numerical coverage before
spending discretionary gSIM work. Every gSIM candidate needs matching passing
numerical evidence; original required L3 obligations and selected calibration
tiers remain intact. A runtime forecast does not waive a required physical tier.
As soon as informative passing Spike witnesses exist, automatic calibration can
progress while other compiler families are repaired; do not impose a global pass
barrier. Aim for below 5% error on independent held-out measurements within the
model's claimed domain, including large-shape regimes. Small held-outs alone do
not validate large-shape accuracy. Use a qualified selected model when available;
otherwise retain explicit provisional uncertainty and missing coverage. Keep
FireSim as independent final validation, never coefficient-fitting feedback.
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

## Learn reusable schedule regions

Explore both the optimization mechanisms and their selection: edit or introduce
shared passes, transformations, schedules, knobs and flags when the existing menu
is insufficient. Keep semantics/lowering shared; specialize actual pass pipelines
using source geometry, layout, precision and hardware-derived resource facts.
Start with coarse regions, then refine where paired complete-call cycle evidence
reveals a crossover. Consider interacting dimensions and aspect ratios, not only
one global size cutoff. Storage-fit legality is independent of profitability.
Use cheap cycle-model comparisons first; spend additional simulations near uncertain
crossovers or on a specific unresolved mechanism, without rerunning the whole corpus.
Compare both sides of each proposed boundary and aligned/ragged neighbors. Freeze
separate geometries for validation; a holdout used to revise a rule becomes development
data and needs a fresh replacement. Preserve each region's incumbent where evidence
is missing, noisy or regresses. Avoid finer rules without useful performance support.
When exported, `shape_policy` assesses proposed typed guards and executes selected
registered pipelines; the agent still discovers the guards and supplies real paired
feedback. Extend shared predicate tooling for missing generic features rather than
encoding capsule identities. Record the rule, selected pass order/options, evidence
domain, gains/regressions and tested coverage; no universal generalization claim is
implied. This work runs alongside construction and adds no launch prerequisite.

Optimize across compiler levels when a local schedule plateaus. Compare graph
grouping and fusion/fission, producer/consumer layouts, immutable preparation,
storage lifetimes and reuse, instruction selection and command batching, as well
as device schedules. Preserve structured batch/head/reduction axes and explicit
effects until the relevant transformations have consumed them. Keep every chosen
alternative connected to ordinary host/device emission; annotations alone do not
change execution. Fusion can lose through spills or lost parallelism, so retain
unfused alternatives. Layout choices include downstream conversions and packing.
Only hoist preparation for truly immutable/versioned inputs; report cold setup
separately from repeated execution under the declared workload scope.

Use hardware-relative features where meaningful (tile counts, tail fractions,
live bytes/capacity, reuse distance and transfer/compute demand) alongside raw
dimensions. Do not invent measured activity from those static features. Explore
small interacting bundles when one-pass-at-a-time search misses enabling effects,
then ablate useful winners to explain their benefit. Rank next work by expected
whole-model time saved, occurrence count and evaluation cost; estimate only from
compatible timing scopes, and check complete chains for shared/overlapping work.
Screen legality, changed emitted code and duplicate programs cheaply; reuse valid
prior observations and evaluate informative boundary cases before broad sweeps.
Keep compiler search cost separate from target runtime, and prefer simpler rules
when timing uncertainty cannot distinguish alternatives. Record missed optimization
reasons (legality, missing representation/adapter, unsupported timing or measured
regression) so a helper improves the actual bottleneck rather than adding flags.

## One evolving compiler, parallel bounded work

Act as one compiler coordinator and delegate bounded, independent tasks to the
available parallel workers, up to the selected helper ceiling.
Start independent capability construction promptly; use fewer workers when ownership,
CPU or token budgets would make more redundant. Choose tasks from the current bottlenecks: frontend
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
compiler/input/runtime hashes, never automatically to newer edits. Name the revision, touched files, consumed pass/IR APIs, local dependencies,
focused checks, first actionable failure, numerical/performance/model confidence and
artifact paths in each short handoff. Pass relevant contracts and a compact capability
brief to helpers; retrieve detailed source or logs only for their current boundary. Reuse an existing equivalent request/result
only when those relevant identities match. Avoid full-cohort checks on every edit;
reserve them for declared milestones/final validation. Report untested coverage.
All selected CPUs are dynamically shared through the existing pool: no fixed
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

If selected, `selected_contracts.scheduling_spec_draft` and
`inspect_scheduling_draft` expose missing declaration fields through the ordinary
resolver. Supply explicit authored facts/software paths; keep unknown effects
unknown until established. Raw ISA encodings alone do not provide logical memory
effects or completion. The current compact dense family requires synchronous
logical operations; queued acceptance is not completion. Existing block schedules
remain available while developing a sound asynchronous adapter or extension.

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
compiler change may belong in semantic admission, graph/region construction,
numeric/layout contracts, allocation, instruction lowering or host execution,
not only a pass. Identify the missing boundary and retain a working route.
When the selected `DenseSourceResult` provides `emit_llvm`, it connects the
checked compact schedule to the ordinary `FnBuilder`: the target callback emits
instructions while the shared schedule owns loops, tails, accumulation and final
stores. Inspect that seam before reconstructing those semantics manually. This
emitter preserves the chosen schedule; it is not by itself a performance win.
Compare meaningful scheduling/policy alternatives through actual emitted code.
Use only available APIs and extend the editable library when a real gap remains.
The
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

When the selected facade provides resumable discovery, bound each `discover`
page with `max_candidates` and retain its returned cursor for the same source and
contracts. `evaluate_fusions` evaluates isolated `FusionAlternative` products
alongside an unfused baseline; pass the actual product to the emitter. Use
`realize_candidate(..., connected=options)` with the selected
`connected_regions.compile_connected_pointwise` bridge for supported ordered
pointwise stages. Its explicit tensor bindings and runtime continuation preserve
live buffers across launches; this is not proof of on-chip retention or a speedup.
When selected, `compile_connected_region` also composes checked dense and pointwise
products with explicit live buffers. Its `ordered_invocations()` supplies the
actual products and bindings to ordinary target code generation; `execute()` is
a separate synchronous buffer-runtime interface. Mixed dense/pointwise composition
still requires the target's multi-stage pointer/launch ABI and completion checks.
Neither API alone establishes fused device execution or on-chip retention.
Implement missing reusable composition mechanisms through editable compiler
code; retain a correct route and all live outputs while exploring alternatives.

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
