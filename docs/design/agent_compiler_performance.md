---
title: "Design: compiler abstractions and tools for performance convergence"
kind: design
status: draft
owner: core
last_verified: 2026-10-06
related: [architecture, perf_phase2_wiring, capsule_phase_split, performance_budget_unit]
code_refs:
  - src/merlin/kernels/cca.py
  - src/merlin/kernels/cca_mlir.py
  - src/merlin/kernels/action_catalog.py
  - src/merlin/xdsl_dialects/schedule.py
  - src/merlin/xdsl_dialects/lowering/global_plan.py
  - src/merlin/xdsl_dialects/lowering/outline.py
  - src/merlin/xdsl_dialects/lowering/dispatch_program.py
  - src/merlin/xdsl_dialects/lowering/outlined_plan_emission.py
  - src/merlin/llvmlower/device_catalog.py
  - src/merlin/perf/global_planner.py
  - src/merlin/perf/agent_guidance.py
  - src/merlin/perf/phase2_edit_contract.py
  - src/merlin/perf/phase2_analytical_provider.py
  - src/merlin/perf/cost_terms.py
  - src/merlin/perf/placement_census.py
  - src/merlin/targetgen/model_coverage.py
  - src/merlin/targetgen/eligibility.py
  - src/merlin/perf/activity_schedule.py
  - src/merlin/perf/fast_estimate_validation.py
  - src/merlin/llvmlower/lowering_recipe.py
  - src/merlin/llvmlower/compilation_recipe.py
  - src/merlin/llvmlower/broadcast_math_hoist.py
  - src/merlin/llvmlower/segmented_input_acceptance.py
  - src/merlin/llvmlower/ordered_fma_groups.py
  - src/merlin/llvmlower/ordered_fma_group_outline.py
  - src/merlin/llvmlower/ordered_bf16_group_binding.py
  - src/merlin/llvmlower/closed_group_writer.py
  - merlin/runtime/c/f32_interval_endpoint.h
  - src/merlin/llvmlower/enclosed_readout.py
  - src/merlin/llvmlower/llvm_loop_metadata.py
  - src/merlin/runtime/host_provider.py
  - src/merlin/runtime/backends/spike_model.py
  - packages/merlin-experiments/src/merlin_experiments/phase2/authoring.py
  - packages/merlin-analysis/src/merlin/agentreport/tokens.py
---

# Compiler abstractions and tools for performance convergence

The agent needs an inspectable path from source semantics to selected schedules, emitted work,
and measured costs. Extend the existing CCA, schedule dialect and global planner at their
current boundaries. Keep the optimization algorithm in Merlin and hardware facts, instruction
selection and device implementation in the target's OOT dialect repository.

The normal whole-model operation profiler now reads typed operations rather than
requiring a custom function print form. Effect-only calls/stores receive their
own intervals, and source identity arrays survive in the table. Generic public
entry functions retain their C interface through textual preprocessing. Actual
native marker order and original outputs, plus an ordinary upstream RV64GC
build/run, qualify this measurement path. This does not assign costs to an
uninstrumented model: paired timing remains necessary because markers can change
fusion, allocation placement and code layout. Complete instrumentation of a
3,434-operation/70-call prepared model restores the original IR structure after
removing marker calls; its full numeric and hardware timing gates are separate.

## Current foundations

### Phase 0, 1 and 2 integration backlog from accelerator coverage

The manual campaign found a concrete integration failure: an accepted
segmented-input optimization was absent from a newer winning compiler recipe.
Contraction coverage also failed to expose a remaining host global reduction.
The reference's unbracketed timer is not a CPU utilization measurement: its
layer routines contain CPU address calculation, command issue and waiting.
Its separately timed global average pool is device work. These failures require
the following changes at the existing compiler and experiment boundaries.

| Phase | Required change | Existing owner / foundation | Status and admission evidence |
| --- | --- | --- | --- |
| 0 | Census all source compute regions, including reductions, preparation, nonlinear maps and epilogues; distinguish native capability, legal composite representation, missing lowering, numerical refusal and unknown evidence | `model_coverage`, `placement_census`, target eligibility contract | Foundation exists; mandatory connection to every golden build and complete physical placement remains unfinished. Contraction-only catalog completeness must never be displayed as whole-model coverage. |
| 0 | Freeze original inputs, accuracy policy, target facts, source semantics, allowed compiler edit surfaces and measurement boundaries | Existing provenance, recipes, package and experiment contracts | Reuse these owners. The evaluated optimization agent cannot change the trusted oracle or grant itself new floating-point permissions. |
| 1 | Expose general device reductions and fused consumer alternatives, as well as GEMM schedules | Source group/consumer proofs in Merlin; typed instruction and resource lowering in OOT | New candidates require explicit dtype, overflow, rounding, lifetime and live-use proofs, independent shapes and source-equivalent fallback. An accelerator call is not itself a profitable transformation. |
| 1 | Price the complete producer and consumer, including packing, allocation, readback, certificate and refinement work | CCA emitted work, complete capsules, ordinary compiler recipe, cost providers | Several qualified local improvements regressed in full capsules or whole models. Retain negative evidence; do not promote from one convenient synthetic fixture. |
| 2 | Persist compatible accepted transformations in each candidate recipe and explain every omitted winner | Normal `DeviceRouting` hooks, explicit features, source-bound implementation identity, global plan | Actual recipe composition is not automatic today. Compare effective features, source/ABI proofs and selected objects before running the new candidate. A changed source can invalidate a proof and must cause explicit requalification rather than silent omission. |
| 2 | Select and emit a whole-program plan for representation, residency, fusion, dispatch and buffer lifetime | `GlobalPlan`, schedule dialect, global planner and existing model build hooks | Per-contraction selection is enabled; shared whole-program search is not connected end to end. A selected-plan-to-final-object witness is required. |
| 2 | Measure conserved current host gaps and device callback intervals; retain unknown pure CPU/device utilization | Typed operation profiler, final-image identity, hardware measurer | Old profiles cannot be subtracted from new champions. Callback intervals include CPU issue and synchronization; array busy time needs separate hardware evidence. |

The candidate-ranking model should use source-derived useful work, actual
issued work, moved bytes, host evaluations, dispatch and dependency constraints.
Calibrate across independent sizes and workloads, report uncertainty and
unpriced terms, and validate the known fast reference as held-out evidence.
Fitting a total to the reference's target cycle count does not validate the
component model or schedule ordering.

Production selection must depend on semantic, layout, numerical and resource
facts. Workload names, capture identifiers and expected output bytes cannot
select an implementation. Source hashes identify proof and calibration
bindings; they grant neither profitability nor algorithm selection. Finish
with the original whole-model gate, all executable-section instruction checks
and actual hardware identity before promotion.

The shared empirical cycle-screen validator now takes arbitrary provider feature
JSON pointers and content identities for executable, workload and the combined
target/timing-scope/memory-regime domain. It fits each fold without the held-out
workload group's labels, refuses missing or extrapolated features, reports absolute
errors and uses the existing schedule-rank gate. Overlapping cycle intervals remain
undecided. Equal-feature schedules with different observed costs produce a collision
receipt; they do not silently become distinguishable. An optional nonnegative linear
screen explicitly selects a fixed term, requires two distinct points per parameter
and refuses collinear features. Its coefficients are statistical screening terms,
not physical service-rate calibrations. Existing mechanism calibration and resource
composition remain authoritative for those claims.

A known fast implementation should be evaluated using its own emitted/executed
features with its timings excluded from coefficient fitting. Compare its section
costs and whole-program total: a coincidental total can hide opposite errors in
compute, host and movement. Keep additional workload groups for generalization
checks. Matching one known program does not qualify schedule ordering by itself.

CCA describes observed implementations. Its ten facets cover compute, vector, memory,
envelope, spatial, SIMT, dispatch, layout, coverage and communication, with scope and provenance.
The complete MLIR serializer preserves every facet field, including tuples, unknown values,
zeros and false values. It retains legacy compute/vector/memory views and rejects disagreement
between those views and the complete record. Preservation does not make an unknown field known:
the analyzers must still supply the evidence.

CCA is descriptive; compiler decisions belong in `schedule` and `GlobalPlan`.
The existing schedule dialect describes placement, lifetime, resident packed tensors and dispatch
grouping. The planner already represents value/buffer representations, demands, resource occupancy,
region and transition alternatives, cycle intervals and plan verification. The analytical provider
already accepts content-bound calibration, prices declared features and physical movement, composes
resource costs and retains unknown inputs. These are useful foundations for the extensions below.
`ActivityTimeline` already schedules event dependencies and resource/serial-group constraints;
reuse it when binding a concrete implementation's compute and movement events.

Agent guidance resolves manifest optimization surfaces to actual Python AST symbols and package
components. The phase 2 edit contract requires owners for the decisions being studied and rejects
missing owners. This identifies where an agent may edit; an observed bottleneck does not itself
grant permission to change an excluded host compiler or grader. Current phase 2 context does not
require whole-model FireSim for every iteration. A hardware performance goal therefore needs an
explicit qualification policy beyond a successful analytical iteration.

The current ahead-of-time model route in `zephyr_model` invokes `DeviceRouting` preparation,
catalog and post-offload callbacks before lowering the remaining host graph. That seam does not
invoke `global_planner.optimize_program`. A target catalog can select its own legal schedules,
but use of a catalog alone does not establish shared whole-program search. Similarly, an existing
timeline class does not establish that a package constructs a timeline from its emitted operations.
Agent guidance already reports this distinction as `global_planner_wiring`, leaving shared search
unknown without a selection/emission witness. The first task is integration and evidence at these
existing seams; class existence is not an enabled optimization.

The current OOT catalog now supplies a narrower, verified integration. The ordinary model build
accepts exact prepared-source contraction calibrations, invokes the existing shared optimizer for
each explicitly covered contraction and links the selected compiled objects into its real catalog.
An independent complete 17 by 73 by 65 model preserves all 1,241 original integer outputs in native
and target execution; the selected full-fixture implementation measures 3,234 to 2,357 GSIM cycles.
Catalog selection, actual symbols, objects and final executable are closed. This is per-contraction
measured selection; it neither prices whole-model boundaries nor enables whole-graph search.

Source identity admission and profitable selection must remain separate receipts. The normal
outlined model identity gate preserves the original prepared source and closes required external
implementations. The optional calibrated catalog changes actual device code beneath those bindings.
An agent can now edit the catalog integration owner as well as the singleton selector and device
schedules through the package inventory. Missing prices remain unknown; a capsule's duration must
not be transferred to a different source, shape, layout or whole-model context without evidence.

The first real model calibration also exposed and fixed a shared dispatch bug: equal tensor shapes
could collapse calls selected to use different device implementations. Dispatch identity now
retains the source-bound implementation, while identical implementations preserve existing sharing.
The complete normal model executes all155 calls and passes every256,000 original compiled output
words plus its original framework gate. The selected implementation's matched capsule is separate
performance evidence; that gate does not establish whole-model hardware improvement.

The shared outliner and outlined-plan emitter now accept explicitly permitted, typed external
declarations through `external_symbols`. External calls remain in source order as graph nodes,
including calls with no results. They do not consume an outlined-kernel table entry. The emitter
preserves their call ABI and declaration attributes in its expanded-computation proof. Missing or
unpermitted declarations still refuse, and the receipt leaves external implementation closure
unknown. The target adapter must bind every required symbol to its actual compiled catalog and
linked executable. This enables an existing model route to use shared emission without inventing
function bodies; it does not establish profitable plan selection or calibrated timings.

## Recommended additions, in order

Ordinary upstream lowering now records its resolved command/pipeline/gates, selected
features, prepared input/runner/generated schedule identities, executable identity,
redacted project environment and successfully returned LLVM hash. Every new invocation
removes the previous success receipt before validation. This closes an observed missing
lowering-recipe seam; imported pass implementations, runtime compiler flags/ABI, final
codegen/link and staging still need their own closure. It leaves emitted IR and existing
build identities unchanged.

The ordinary bare-metal whole-model builder additionally records owned model,
runtime, harness and explicit math-policy compilation plus final linking. Each
command retains the effective flags, executable hash, redacted project environment,
explicit source/input hashes and returned object identity. Ordered link inputs bind
actual provider objects and the final executable is completed after existing audits.
Actual RV64GC execution of a minimal complete model, matching helper compiler ABI,
exact relinking and failed/refused reused-workdir cases qualify this observation seam.
Headers, resolved libraries and provider-owned compile commands remain outside its
closure. Existing flags, emitted bytes and build-marker identities are unchanged.

The bare-metal model builder also accepts an explicit `host_provider_builder`.
A mixed host/provider implementation can call retained source fallbacks, numerical
helpers and target primitives; such an object needs a separate compilation boundary
from a device catalog whose kernels must have no unresolved symbols. The hook
keeps that catalog rule intact. It binds the final prepared source and model LLVM/object,
each imported object and companion LLVM, explicit compiler commands and working
directory, declared dependency hashes, numerical/effect witness identities, and
defined/required C function signatures. It rechecks those pins at the ordered final
link and verifies that every required symbol resolves in the actual executable.
New compile commands can use the ordinary compilation recorder supplied by the hook.

These checks close the declared imported compilation and linkage evidence. Dependency
completeness and numerical/effect proofs remain the emitter's obligations; recording
their hashes does not prove them. LLVM signatures compare simple scalar/pointer C
machine boundaries, including integer extension attributes. Opaque pointers do not
encode logical tensor dtype, descriptor rank, strides or ownership: the typed source
binder, provider contract and actual ABI/numeric gates must close those separately.
Unsupported aggregate, address-space or calling-convention boundaries refuse.
The hook is inert by default and grants no implicit approximation, target capability
or source-fallback permission. Actual native and normal upstream RV64GC executions
exercise a ranked provider calling its retained source fallback, with immutable input
and output guards, exact final relinking and import/link mutation refusals.

| Priority | Addition | Phase | Concrete result |
| --- | --- | --- | --- |
| 1 | Numeric and coverage contracts | 1, retained in 2 | Every route states supported semantics, reduction order, rounding, intermediate precision, output ownership and layout. Unsupported cases have a reasoned refusal. Original accuracy criteria remain pinned. |
| 2 | Complete compilation identity and ABI checks | 1 and 2 | A receipt connects source IR, passes, runtime/toolchain ABI, device objects, linked binary, staged binary and hardware revision. Minimal target executions exercise conversion helpers and host/device calls before expensive whole-model runs. |
| 3 | Source-to-decision-to-emission explanations | 1 and 2 | CCA facts link to the selected alternative, responsible pass/AST symbol, legality proof and emitted work. A pass reporting success is checked against the resulting code. |
| 4 | Bind and verify temporal resource plans | 2, legality checked in 1 | Bind emitted events to existing ActivityTimeline/schedule/global-plan representations; extend them with buffer read/write intervals and next-panel slots where needed. Verify lifetime and hazard constraints before costing a candidate. |
| 5 | Region and representation alternatives | 2 | Enumerate legal fusion, packing, gather, requantization, residency and host/device partition choices. Price intermediate materialization and boundaries as well as arithmetic. |
| 6 | Calibrated models with uncertainty | 2 | Estimate host issue, arithmetic, conversion, transfer, dispatch and synchronization costs using measured ranges and resource timelines. Return unknown for missing rates; select the next measurement when uncertainty changes candidate ordering. |
| 7 | Explicit host and device edit grants | 2 | Supply separate source-bound packages/contracts for reusable Merlin transforms and OOT implementation. Required decisions must name an editable owner, including host code generation when it is part of the optimization objective. |
| 8 | A fidelity ladder and representative portfolio | 1 and 2 | Run legality and native accuracy, minimal target ABI/ISA checks, relevant full-output capsules, then whole-model hardware qualification. Add held-out shapes, tails, layouts and numeric boundary cases before general promotion. |
| 9 | Durable experiment and queue lifecycle | 2 tooling | Record intent and immutable pins before submission; collect staged identity and results with restart-safe workers. Distinguish queued, running, failed, unverified and qualified outcomes. |
| 10 | Optimization and token ledger | 1 and 2 tooling | Attribute each change to its owner, hypothesis, emitted difference, gates and matched measurements. Keep regressions and timeouts. Record token counters per agent/request window without counting cache or reasoning subsets twice. |

These are target-independent contracts and tools. A provider supplies target capacities, address
legality, pipeline and bandwidth facts, instruction semantics, ABI details and execution adapters.
Core code must not infer those facts from the name of a target.

### Keep call and stack boundaries in the calibration domain

`merlin.perf.execution_boundaries` accepts typed provider-decoded function extents
and instruction records with explicit execution counts, call kinds, stack access
widths and known or unknown frames. It checks complete, nonoverlapping coverage
before summing or normalizing to a selected entry invocation. Symbol identities
bind evidence only. The target provider owns instruction/register/ABI decoding
and verifies the recorded program, census and provider artifacts.

The summaries distinguish per-call prologue accesses from repeated stack sites,
direct and indirect calls, unresolved callees and unknown classification. They
do not recover chronological memory events, physical traffic, cross-call/loop
dependencies or peak simultaneous stack usage. A training-only boundary envelope
can refuse a newly outlined helper or new spill pattern even when the numerical
features used by a cycle fit are inside its scalar ranges. Passing this unpriced
subdomain neither prices those boundaries nor approves a ranking: the ordinary
held-out coverage and ordering gates remain required. Missing facts are never
treated as zero cost.

### Bind memory experiments to their byte representation and runtime

A border-initialization experiment reduced isolated instructions by27% but
added666instructions in the complete model with its actual runtime. The capsule
memset used a scalar byte fallback at its odd byte length; the whole model's
libc used word transfers and a tail. General scalar copy expansion then removed
copy helpers while increasing whole instructions15.92%. These results show why
phase2 needs runtime identity, alignment, byte extent and emitted transfer
strategy in a memory experiment's execution regime.

The useful shared alternatives now include private uniform-fill copy folding
and contiguous-suffix specialization. The first proves a fresh private full
uniform store, complete direct-copy users and source lifetime/order. The second
proves distinct allocation roots and a longest common contiguous suffix, then
retains outer traversal with ordinary memcpy for alignment and tails. Neither
assumes a device or a workload. Both are explicit default-off pipeline choices
with required application-count receipts; the original full model remains the
numeric and performance qualification scope.

A semantic strided f32 view is also distinct from a later quantized i8 storage
view. A consumer capability must name element representation, element versus
byte offsets, owning allocation, valid lifetime and exact producer placement.
Borrowing an i8 representation requires proving it already exists and remains
live, or generating the source's exact strided quantization/packing. It cannot
reinterpret an earlier f32 root. This belongs in the existing representation,
buffer, demand and transition contracts; device DMA segmentation and address
legality remain the provider's responsibility.

### Explain the actual CPU code produced by a numeric alternative

Canonical integer radix products now have an explicit shared reconstruction
alternative. An absolute-prefix bound proves signed-i64 weighted updates and
one final binary64 conversion exact, including positive zero cancellation.
Scratch allocation, initialization, traffic and finish must be priced with the
producer. A variable-weight switch initially produced a runtime multiplication
per output; putting each proved positive constant inside its own case lets
ordinary CPU codegen emit constant shifts while the C source retains defined
negative multiplication semantics. The emitted instruction witness explains
this difference; a valid mathematical rewrite alone cannot predict its cost.

Agent guidance should link each optional transform to the responsible shared
AST symbol, numeric/layout proof, application count, produced IR/objects and
complete matched measurement. The existing edit contract and compilation
recipe are the owners for these links. Generic emitted-work categories and
counter provenance belong in CCA; a guessed CPI or unobserved overlap does not.

### Check semantic parameters through instruction lowering

A source-stride convolution experiment exposed a target lowering that discarded a
nondefault execute-stride property. The operation existed and default tests passed,
while the requested layout could not execute correctly. Phase 1 should exercise each
declared semantic parameter through actual lowering and decoded emitted instructions,
including nondefault values, interacting settings and refused resource cases. The OOT
provider owns parameter/encoding/legality facts; Merlin can own the generic coverage
contract and witness collection. An unconsumed semantic property must be refused or
explained, rather than silently receiving a default. This coverage extension is proposed;
the concrete OOT stride fix has its own target tests and qualification.

### Carry numerical requirements with optional algorithms

Reusable certificate code now has distinct full-norm and representation-only metadata
types. A caller with an independent absolute-product bound can request only the fields
it consumes, avoiding square/sqrt work; the reduced type cannot feed the full Holder
consumer. This is an example of a target-independent requirements abstraction. Numeric
scope must also state finite/RNE assumptions, nontrapping arithmetic and whether FP
exception flags are observable. Stable rounding alone does not prove flag equivalence.
Neither local error bounds nor local capsule tolerance establish whole-model acceptance:
an approximation still has to pass the original complete accuracy gate.

### Close the complete consumer frontier before offloading floating contractions

A contraction's result type does not describe the complete numerical obligation.
An f32 result can feed a pointwise scale, a maximum reduction, a nonlinear map,
a denominator reduction and a later output conversion through several live paths.
A certificate for a rounded contraction result cannot replace those paths unless
it proves every observable consumer. A final BF16 result alone does not authorize
early BF16 rounding of an intermediate that still has live f32 consumers.

The optional shared ordered-FMA analysis now identifies supported typed source DAGs
and their live frontier: operand maps, reductions and order, scalar precision,
conversion boundaries, source contexts and escaping uses. The source-group outliner
moves the original operations into ordinary functions, retaining arithmetic and
initialization semantics. Source-bound preparation uses the regular DeviceRouting
callback and validates function bodies, contexts and typed call boundaries before
binding. This is source partitioning, not permission to approximate or offload.
Unsupported reductions, effects or conflicting maps refuse while preserving the
source fallback. See [source groups](../reference/ordered_fma_groups.md).

A certified region can include softmax and normalization only when its certificate
covers the actual arithmetic and denominator uses. Generic interval endpoint helpers
provide distinct exact and explicitly bounded BF16-bin policies; they require the
declared floating environment and complete operand bounds. A bounded local result
still needs the original complete workload accuracy gate. The provider supplies
device partials and resources; Merlin owns source legality, precision requirements
and fallback contracts. Numerical primitives and source grouping do not establish
an enabled device implementation or a performance gain.

An optional certificate boundary now retains a closed quantized consumer.
The typed frontier analysis enumerates all live
uses, coordinate transformations, source extrema, scale calculations and
conversions. If it proves every escaping integer value and scale unchanged,
unused floating differences need not force source replay. Any additional
floating escape invalidates that smaller obligation. Leaving the original
consumer in place can preserve the existing writer ABI, provided the complete
use context is immutable and the numerical witness proves observational
equivalence. The explicit observed-writer binder validates complete source,
context and observation witnesses before delegating the existing ABI and buffer
checks. The supplied numerical theorem remains a separate obligation; endpoint
helpers or frontier analysis alone do not establish it. Agents should compare
ambiguous output counts and actual
refinement work, then price preparation, readback and the complete consumer.

CCA should report source work, actual device coverage and the remaining host work
separately. A fast standalone kernel is not coverage evidence for a whole model.
Bind a selected region to its prepared source, emitted calls and final implementation
before using its measured cost to prioritize the next optimization.

### Verify preparation, allocation and emitted CPU structure

The typed broadcast-math pass demonstrates an existing smaller-domain source
alternative: a pure operation can execute before a broadcast when its full scalar
body, maps, live users and numerical scope prove equivalence. Price its actual
evaluation count after upstream lowering. Likewise, accepted segmented inputs
require both an exact owning byte representation and an explicit read-only consumer
contract. Merely declaring a target view capability does not remove a host copy.

Optional device ABIs must also survive source preparation and bufferization.
Check the rewritten scratch element type, fresh owner, allocation extent and
selected host object, including when the input is an already rewritten capture.
A numerically successful execution with a legacy oversized allocation does not
qualify the proposed memory reduction. Closed device catalogs must explicitly
account for any host helper dependencies; a compiler byte-copy intrinsic and a
runtime memcpy call are separate compilation policies with distinct costs.

An ordinary CPU loop selected by a target schedule may need explicit shared
no-unroll metadata. Confirm the loop survives actual upstream translation and
CPU code generation, then measure the complete command sequence. Smaller text
does not prove fewer cache misses or shorter hardware execution. These witnesses
belong in the existing compilation recipe and CCA emitted-work evidence; target
address encodings, command legality and memory resources remain in OOT.

## Where the agent should modify code

| Decision | Editable owner | Protected comparison inputs |
| --- | --- | --- |
| Device tiling, panel placement, target instruction lowering and target legality | OOT dialect, transforms and schedules | Target facts and hardware identity used for the comparison |
| Host fusion, exact scalar scheduling, packing, quantization and requantization | An explicitly granted Merlin compiler package | Original numeric policy and source inputs |
| Global region selection, physical representations, dispatch and buffer lifetime | Merlin schedule/global planner | Source dependencies, alias and ownership obligations |
| Compilation scalability, consistent helper ABI and runtime orchestration | Merlin build/runtime infrastructure | Declared target ABI and linked-artifact provenance |
| Grading, hidden oracle, accuracy budget and baseline selection | Trusted experiment owner | Excluded from the evaluated agent's edit grant |

The package contract should name both the decision and its owning symbol, plus allowed extension
directories. If the agent cannot express a needed legal optimization through the granted owners,
report the missing surface and extend the contract in a reviewed experiment setup. Do not encourage
the agent to compensate with workload-name switches or weaker accuracy checks.

## Temporal plan details

Use the existing activity event DAG for compute, transfer, conversion, configuration, dispatch and
fence events. Bind inputs, outputs and storage hazards to those events and extend duration handling
where intervals are required. Buffers declare address-space roles, size/alignment, lifetimes and access modes.
The target provider maps those roles onto concrete memories, banks, queues and instruction sequences.

Verification checks read-before-write, overwrite-before-consume, capacity, disjoint required intervals
and completion before reuse. A next-panel transfer is eligible for overlap only after its producer
and storage constraints are proved. A cost model can then distinguish fewer commands from shorter
execution: issuing fewer instructions can still increase stalls or reduce transfer/compute overlap.
Add these capabilities to the existing plan rather than creating a second optimizer-only schedule IR.

## Logical bindings and physical implementation identity

Keep source semantics, compiled implementation and calibration context as separate identities.
A logical alternative carries exact source identity, ordinal, operand/result types, numeric and
layout contracts. A physical implementation identity derives from its actual generated operations,
numeric parameters, resource schedule, ABI and compiled bytes. Source labels alone should not force
duplicate copies of an otherwise identical implementation, while equal shapes alone must not merge
different schedules or arithmetic.

The target provider canonicalizes and compiles its instruction IR; Merlin retains the logical
bindings and verifies that every dispatch resolves to the selected actual implementation. Before
sharing a symbol or object, require implementation equivalence and ABI closure. Changing emitted
operations invalidates that identity. Price executable footprint and dispatch alongside arithmetic
when duplicating implementations could change layout, cache or branch behavior.

Implementation equivalence does not establish timing equivalence. Calibration additionally carries
actual buffer addresses/alignment, memory and cache policy, input/context scope, engine/hardware pins
and measurement boundary. A shared device object used by many calls still needs evidence for its
use in that full-program context. Preserve unknown whole-model costs until measured.

## Analytical models and convergence

Derive useful arithmetic, issued arithmetic, tail waste, physical traffic, host evaluations,
dispatch count and conversion work from the actual emitted implementation. Record which features
are exact counts, calibrated estimates or unknown. Preserve source-level and emitted-work counts
separately so algorithmic expansion remains visible.

Calibrate one term at a time with controlled pairs on the declared hardware. Fit resource-specific
ranges across sizes and working sets, including saturation, cache and contention transitions.
Use temporal dependencies and measured overlap to compose durations; neither an unconditional sum
nor an unconditional maximum describes every implementation. A geometric array count alone cannot
establish an end-to-end lower bound when wave lengths, numerical algorithms or host work differ.

The agent loop should:

1. Seal the original program, numeric gate, functional coverage and comparison artifacts.
2. Inspect source topology, CCA, emitted work and a conserved host/device profile.
3. Identify the dominant priced term and its editable compiler owner.
4. Propose a general alternative with an explicit legality proof and predicted cost interval.
5. Run the cheapest gate that can reject it, including allocation, conversion and readback costs.
6. Measure surviving candidates and update calibration with the matched control.
7. Qualify a composed whole-model binary on the required hardware and check the original outputs.
8. Promote the general pass only after independent and held-out cases; retain the losing evidence.

Choose experiments by expected benefit and uncertainty reduction per wall time and token cost.
Use existing measured champions as references for schedules and resource utilization; preserve
their program and measurement boundaries when comparing. Do not claim a goal from instruction
counts, a capsule result, an analytical interval or a terminal UART lacking staged identity.

## Accounting semantics

Use explicit owned session paths or driver accounting files. Record cumulative counters and
completed-request window deltas, with the source identity and observation time. Input includes
cached input in a raw session total; reasoning output is already part of output. Store those subsets
for analysis without adding them again. Reset or contradictory counters remain unavailable.

A campaign mixes OOT, host compiler, correctness and infrastructure work. Record per-thread totals
and event windows, and state when an exact per-optimization allocation is unavailable. The goal
tracker is a separate observed counter until its accounting semantics are reconciled with the
driver's raw counters. Token traffic does not establish monetary billing.
