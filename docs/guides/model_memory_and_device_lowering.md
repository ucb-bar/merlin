---
title: Model memory planning and device lowering
kind: guide
status: current
owner: compiler
last_verified: 2026-10-20
related: [architecture, model_lowering, static_arena_wiring]
code_refs: [src/merlin/runtime/program.py, src/merlin/xdsl_dialects/lowering/arena_plan.py, src/merlin/semantic_compiler/buffer_allocation.py, src/merlin/semantic_compiler/ordered_mnk_tiling.py, src/merlin/compile/c_tiling.py, src/merlin/compile/semantic_regions.py, src/merlin/compile/semantic_application.py, src/merlin/compile/application_c.py, src/merlin/compile/layout_c.py, src/merlin/compile/c_conversions.py, src/merlin/semantic_compiler/contraction_precision.py, src/merlin/targetgen/native_application.py, src/merlin/compile/dispatch_c.py, src/merlin/compile/session_dispatch.py, src/merlin/compile/object_exports.py, src/merlin/llvmlower/session_bundle.py, src/merlin/perf/global_planner.py]
---

# Model memory planning and device lowering

Merlin chooses model regions, tracks tensors between regions, and compiles host
computation and calls. The selected backend owns device registers, instruction
selection, scratch layout, spills and instruction timing. Platform support owns
loading, launch, completion, cache maintenance and the final executable build.

## Passes and compilation stages

Some stages below are library functions rather than registered MLIR passes. The
names in backticks are existing entrypoints. Required backend work is marked
separately; a row does not imply that the entire model path is implemented.

| Stage | Owner and entrypoint | Why it is needed |
| --- | --- | --- |
| Capture and source accounting | Merlin frontend and `application_graph_inventory` | Preserve inputs, weights, outputs, state and exact SSA uses before placement. |
| Numerical lowering | Merlin precision transformation with backend numerical rules | Make conversions, scales, initialization and reduction order explicit. A new precision policy needs its own quality evaluation. |
| Outline computation | Merlin `lower_model_to_dispatch_program` | Expose compute regions and all captured values while preserving source dataflow. |
| Partition and select implementations | Merlin `optimize_program` with `TargetPlanningAdapter` | Choose supported host/device regions and explicit representation transitions. Missing capability or cost information must stay visible. |
| Group/fuse regions | Merlin `plan_dispatch_fusion` and `OutlinedGlobalPlanEmitter`, or a selected target emitter | Reduce boundaries where semantics and target resources permit it. Grouping alone does not prove fewer transfers or a fused device kernel. |
| Plan application buffers | Merlin `plan_bounded_arena`, used by `build_program(memory_requirements=...)` | Check temporary storage, completion lifetimes, declared views, persistence, padding and reserved space. |
| Tile and generate loops | Shared shape policy plus backend lowering | Cover M/N/K tiles and tails; preserve reset/continuation and the admitted reduction order. The compact `LoopedMNKTiling` plan describes packed operands and output tiles; the selected backend emits loops. |
| Normalize completed MXU state | Atlas `--spill-atlas-mxu-state` | Materialize explicit weight/accumulator block arguments as raw tile values before strict verification and scheduling. It does not snapshot in-flight operations. |
| Schedule virtual device operations | Atlas `--schedule-atlas-virtual` | Order ready operations using dependencies, resources and register pressure. This is device scheduling, not model placement. |
| Insert register spills and reloads | Atlas `--spill-atlas-virtual` | Recover tensor/scalar pressure, including block arguments and entry controls. Use shared VMEM placement for spills and scratch; retain the existing scheduler and register allocator. Completed MXU handles on explicit CFG edges become FP8/BF16 tile values before allocation. In-flight MXU state is not spillable. |
| Final register allocation and device lowering | Atlas `--lower-atlas-virtual-to-machine` and its allocation plan | Assign physical registers, pairs, MXU slots and DMA staging locations after spilling. Application arena placement does not assign these resources. |
| Insert DMA waits | Atlas `--insert-atlas-dma-waits` | Ensure transfers complete before dependent accesses or unsafe reuse. |
| Schedule/time physical instructions | Atlas `--insert-atlas-delays` or `--schedule-atlas-stream` | Respect functional-unit availability and physical hazards after allocation. |
| Verify the final device stream | Atlas `--verify-atlas-machine-stream`, `--verify-atlas-generated-schedule`, `--verify-atlas-timing` | Check the actual placed/timed stream before publishing its bytes. |
| Encode and measure program size | Atlas `atlas-emit` | Produce raw instruction words and count the final image after waits, delays and branch handling. |
| Select safe program boundaries | Merlin orchestration plus backend safe-boundary information; required integration | Sequence complete programs when one image does not fit. A split must preserve live data and drain effects; cutting binary bytes is invalid. |
| Lower host computation/calls | Merlin `llvmlower.lower_model` and `codegen.compile_ll` | Produce CPU objects through LLVM. Device loading stays in the shared target driver. |
| Package and link | Generated device data/interfaces plus the platform build | Link CPU code, constants, startup, driver and linker script into one platform executable. |

The Atlas entrypoints describe the inspected prototype, not a claim about newer
backend branches. No backend pass is added by the graph-memory API below.

## Explicit placement

`GlobalPlanPolicy.allowed_placements` restricts the implementations the global
planner may select. `required_placements` fixes the placement of individual
dispatch nodes, including nodes covered by a fused region. Both use placement
names supplied by the selected `TargetPlanningAdapter`.

For example, if an adapter names its routes `cpu` and `accelerator`, an
accelerator-only search uses
`GlobalPlanPolicy(allowed_placements=("accelerator",))`. A mixed search can use
`required_placements=((3, "accelerator"),)` to keep node 3 on the accelerator.
An unavailable required route produces a refusal; a cheaper CPU alternative
cannot replace it. The physical lower bound still includes all implementations,
so restricting placement does not make the reported hardware baseline easier.

These fields constrain selection among implementations the adapter actually
provides. They do not add missing lowering, qualify a host implementation, or
establish whole-model Atlas coverage. Default placement remains unrestricted.

## Outline selected semantic regions

`semantic_dispatch(request, entry=...)` exposes every pure computation in a
parsed semantic graph as a logical dispatch node. Its node indices are the
indices used by the existing `RegionAlternative` and `GlobalPlan` interfaces.
Declared constants retain their own buffers and exact source bytes. Input
order follows the request boundary; ordered results can include shared values,
repeated results and inputs returned unchanged.

After the target adapter selects legal alternatives,
`outline_semantic_regions(request, entry=..., plan=...)` checks full coverage
and returns one source `KernelRequest` per region. It derives inputs and all
escaping results from actual uses. It preserves computation attributes, index
maps and numerical policies. A cut does not authorize a conversion or a
precision change. Changed arithmetic or constant bytes invalidate the source
binding even when operation names and shapes are unchanged.
Region inputs are retained: a region's last read cannot authorize overwriting
an application input that a later region may still use.

Fusion can combine non-adjacent operations, but the resulting region dependency
graph must remain acyclic. Calls are ordered by those dependencies. Missing,
overlapping or cyclic choices are rejected. Representation changes require an
explicit target emitter; this outliner does not insert hidden conversions.
The output is a checked `GlobalPlanEmission`, consumed by the existing APIs:

```python
outlined = outline_semantic_regions(request, entry=entry, plan=selected_plan)
program = build_program(
    outlined.logical,
    global_plan=outlined.plan,
    global_plan_emitter=outlined,
    memory_requirements=application_memory_requirements,
)
```

`application_memory_requirements` supplies the real capacity, alignment,
completion events and reserved ranges for the emitted dispatch graph. Each
region still needs a compiled implementation matching its ordered source
slice. Outlining proves wiring, not the target implementation's arithmetic.
It does not supply a complete target planning adapter or model compiler.

For independently compiled region contexts, use
`DispatchCall(symbol, placement, context_index=i)` with `emit_program_c`.
The generated header declares a context table; callers provide one slot per
selected index. Invalid tables are rejected before launching work. Calls
without an index retain the original single-context interface. Buffer planning,
error propagation and timeout lifetimes use the same serial dispatch driver.

## Compile a staged application

The optional `application_adapter` on a native target binding supplies source
region legality and compilation through the existing target backends. The
shared `compile_semantic_application` checks source coverage, outlines regions,
plans the emitted tensor arena and generates C dispatch. It does not require
recapturing each region or exporting through another compiler.

```sh
merlin-targetgen native-compile --engine merlin_native --support "$SUPPORT" \
  --snapshot "$SNAPSHOT" --compile-inputs "$COMPILER_INPUTS" \
  --application --application-name model --application-memory-bytes "$TENSOR_MEMORY_BYTES" \
  --abi auto --mode diagnostic --target-source "$TARGET_SOURCE" --out "$RUN/compiled"
merlin-targetgen native-verify --engine merlin_native --support "$SUPPORT" \
  --snapshot "$SNAPSHOT" --compile-inputs "$COMPILER_INPUTS" --application \
  --mode diagnostic --target-source "$TARGET_SOURCE" \
  --artifact "$RUN/compiled" --out "$RUN/verification.json"
```

This path admits static captures under an explicitly selected diagnostic
precision policy and the provider's supported operation signatures. Application
C transports dense fixed-width tensor bytes, with explicit dtype and shape
metadata for non-f32 ports. State binding supports raw f32, f64, BF16 and FP8
storage, and signed-i64 tensors. Initialization copies exact bytes; it does not
perform numerical conversion or admit arithmetic on those types. Caller-owned state requires
the corresponding compiled routes; lifted mutable buffers require an explicit
`--application-state` declaration. By default, declared immutable weights become
exact semantic constants. For large models, pass
`--application-parameters external` to both commands. Parameters then retain
typed source ports during compilation and bind to offsets in one checked
`handoff/parameters.bin` payload. They are excluded from the public runtime
input array; `application-c.json` and `handoff/abi.json` list their byte ranges,
types and hashes. Small captured constants and state initialization keep their
existing roles. Runtime samples and goldens are never loaded by compilation.

Compile the exported C/H pair in the platform build. Load `parameters.bin` into
aligned, caller-owned memory and call the generated parameter-binding function
before invoking the model. Binding checks byte count, alignment and SHA256.
Keep the descriptor at the same address and keep the weight bytes unchanged
through all invocations and recovery; binding performs no allocation and does
not repeat the full hash on every invocation. A failed bind preserves the
previous descriptor. The platform still owns startup, memory placement, cache
maintenance and linking. No host main function or ELF is required by this
interface. External weights do not establish that model data fits device memory
or that its operations can all compile.

Missing computation routes fail before publishing a successful program set.
Source fusion matching does not authorize different arithmetic. For example,
`recognize_silu` identifies a private sigmoid feeding `x * sigmoid(x)`, retaining
the original source nodes. A provider must separately establish its numerical
and storage conditions before selecting a fused implementation; an escaping
sigmoid result prevents this fusion.
Providers can expose `application_region_alternatives(source)`: legal source
regions with explicit placements and boundary representations. Merlin's
`select_application_plan` chooses a complete compatible partition under a fixed
work budget, rejecting overlapping or cyclic fusions. Larger regions are tried
first with stable tie breaking; unknown costs are never ranked or reported as
measured performance. Providers with a legacy `plan(source)` method remain
supported explicitly. A failed shared selection does not call the legacy path.

### Target adapter availability

The application adapter is optional. Atlas's published prototype at
`37372987d3dabee895d100495f22d7b19503fab3` does not yet provide this full
application interface. Its spilling and existing single-region examples remain
separate supported paths. The additional Atlas application/codegen adapter is a
companion change, so these commands require a provider implementing the API
above. Missing support returns `unsupported_semantics`; it does not invoke a
framework interpreter or silently route device work to the host.

`emit_layout_c` provides byte-preserving reshape, copy, transpose, static affine
gather, slice insertion and concatenation. Gather maps are checked over the full
declared shape before C emission. Slice insertion first copies its initialized
destination, then writes the selected positive-stride window. It does not treat
empty storage as zero. Concatenation preserves input order, including repeated
inputs. All routes reject overlap between outputs and inputs, and target-packed boundary data;
they consume dense logical buffers and preserve raw f32 bit patterns.

The provider must explicitly admit each CPU layout operation in its placement
policy. Availability of a C implementation does not authorize moving failed
device arithmetic to the CPU. Device-required regions retain their own coverage
and execution requirements.

`PointwiseTiling` retains each source tensor's logical rank and affine input
maps. For tile iteration, it combines the leading output dimensions into rows
and keeps the last dimension as columns; a vector has one row. Python and C
packing recover the original coordinates before applying each source map.
Broadcasts, reversed views and channel-wise operands therefore retain their
logical indexing. Output assembly writes the original dense element order.
The target supplies tile geometry and numerical semantics, and selects the
two-dimensional body. This does not add a reshape to the source computation or
change its arithmetic. Padding values and every read bound remain explicit.

`emit_application_c` binds constants and source input order to the existing
dispatch driver. It stages computed results and copies every ordered output,
including repeated and passthrough values, only after all calls succeed. Contexts
and scratch remain caller-owned. The application byte budget covers tensor
temporaries and output staging; per-region packing/conversion scratch is reported
separately. This does not establish general device or persistent-state allocation.

The publication also combines the selected region sources and one shared runtime
into a standalone C/H pair under `c-package/combined/`. The application interface
records its source/header paths, and `link_sources` selects only that C file.
Region sources remain available for inspection; they are not extra link inputs.
The platform still supplies callbacks, startup, linker placement and the final ELF.

`native-verify --application` reopens the source and snapshot, reconstructs the
partition, rechecks native publications, regenerates region exports, and replays
buffer/C generation. Rehashed changes to code or source still fail replay. These
are structural and artifact-identity checks, separate from numerical execution,
original-model quality and selected-system qualification.

For explicitly scaled contractions, `FloatBoundarySpec` records independent
operand encoding exponents and a result decoding exponent. The contraction
checks that the latter restores the sum of operand scales. The C BF16 decoder
implements the result conversion with integer bit operations and binary32 RNE,
including subnormal results and overflow. This is numerical boundary work on
the host, not an equality rewrite or a model-quality guarantee.

## Completion-aware application buffers

`ArenaRequirements` extends the existing dispatch-program memory plan. It names:

- The exact emitted dispatch digest and arena capacity in bytes.
- Launch and completion events for every node, in dispatch order.
- Required allocation alignment and optional physical byte sizes, including padding.
- Declared intermediate views as `(view, parent, byte_offset)`.
- Buffers reserved for the whole invocation and byte ranges owned by other work.

A buffer remains live until its last consumer completes. An asynchronous reader
can therefore keep an input alive after later independent dispatches launch.
A view extends its root allocation's lifetime. Different views of the same
backing store cannot be allocated as independent memories. Reuse at the same
event is conservatively forbidden for separate allocations.

The selected backend supplies physical footprints and legal views. The planner
checks their bounds and retention; it does not establish numerical equivalence
of a view or infer that an operation is safe in place. Graph source types must
already reflect any selected precision transformation.

`allocate_buffers` uses the existing Z3 dependency for finite placement and checks
its witness using direct interval arithmetic. It returns `sat`, `unsat` or
`unknown`; `unsat` concerns the supplied storage/events, not the hardware's
ability to implement the original model. Addresses are selected deterministically
within a fixed instance. All sizes use an explicit storage unit.

`plan_bounded_arena` adapts this to the existing `MemoryPlan` and `MerlinProgram`
representation. Arguments, constants and caller-owned results remain outside the
temporary arena. Persistent intermediates remain reserved during one invocation;
state initialization and retention across invocations remain caller obligations.
Aliases to external buffers are refused by this first adapter.

Use the graph digest after global-plan emission. A schedule prepared for another
graph is rejected. A completion event later than a dependent consumer's launch is
also rejected; the planner does not insert a missing wait on the caller's behalf.

This API is opt-in. `emit_program_c` consumes the bounded plan and emits a serial
C driver that binds intermediates to `arena + offset`. It calls explicitly
compiled implementations for every node, including support operations, and
requires completion before continuing. See [C dispatch](dispatch_c.md).
The existing `plan_arena` and monolithic LLVM arena binder retain their paths;
this does not redirect an existing monolithic host build through dispatch.

`plan_program` connects `optimize_program`, a selected `GlobalPlanEmitter`, and
bounded buffer placement. Its memory-requirements callback sees the emitted
graph after fusion and inserted conversions. A plan without a complete route
fails instead of selecting an implicit host fallback. Performance bounds remain
in the returned planner report; a structurally valid route is not a timing proof.

## Compact contraction loops

`LoopedMNKTiling` derives M/N/K trip counts, packed buffer sizes and tile order
from one typed, ordered contraction. It does not construct an e-graph or kernel
request for every output tile. The target supplies tile geometry, element widths,
the numerical policy and separately qualified pad values.

`materialize_looped_mnk_inputs` packs A in M/K order and weights in N/K order,
so each operand is stored once. A short K segment comes first, padded before its
real elements. `assemble_looped_mnk_outputs` trims invalid M/N lanes after the
backend has removed its physical lane layout. These are boundary transformations;
they do not authorize changing reduction order or relaxing arithmetic.

The Atlas provider uses this plan for an admitted FP8/BF16 contraction, with an
optional terminal ReLU, and emits nested device loops with explicit output-buffer
bounds. A looped contraction alone does not establish whole-model coverage or
full-system deployment.

`render_looped_mnk_c` emits byte-preserving C pack/assembly helpers from the same
plan. The caller validates buffer sizes and aliasing; the target provider handles
its physical lane layout and calls its existing driver. This keeps shape/index
packing shared without moving target instructions or platform services into Merlin.

`plan_streamed_mnk` handles packed buffers that exceed a selected device window.
It chooses fixed M/N panels under the byte budget, including alignment, and
retains the complete K reduction in each launch. Its deterministic search
minimizes launch count, then padded output work and operand transfers.
`render_streamed_mnk_c` packs panels and scatters valid cells; it does no tensor
arithmetic. The target supplies a legal padding encoding and executes the same
selected body for each panel. Generated dispatch waits before reusing staging
and publishes the output only after all panels succeed. Host staging is counted
separately from the device allocation. A minimum full-K panel that cannot fit
still produces a resource-limit diagnostic; cross-launch K continuation and
device-resident KV storage remain open.

For a captured f32 contraction, `native-compile --compile-inputs ...
--diagnostic-region ... --whole-contraction` preserves the complete source window
and one target semantic reduction. The selected provider checks the declared
precision policy and target support. `precision-plan.json` records operand axis
order, conversion specifications, source values and result shape; it remains an
explicit precision change. `native-verify` replays the same source-bound request
and parameter packing. Selecting a tile index at the same time is an error.

This is region compilation. It does not account for the rest of the captured
graph or generate its dispatch order. Source conversion, caller-owned encoded
operands, declared parameters and target layout must agree before execution.

`render_f32_operand_c` emits the checked rank-two operand conversion in its
recorded axis order. It reuses the finite float codebook and integer-only C
encoder, validates all values before changing the encoded destination and does
not allocate storage. The target's generated source-call wrapper binds these
helpers to its region interface and caller-owned conversion scratch.

## Row softmax across tiles

`decompose_row_softmax` consumes an explicitly selected numerical policy and
primitive descriptors from the existing target catalog. Each row group retains
all column tiles, one global maximum, and one global denominator. Tile sums
combine left to right under the selected primitive's rounding. A tail mask is
an initialized constant after exponentiation; padding does not add terms to the
denominator. This decomposition does not reassociate an arbitrary source f32
reduction.

`materialize_row_softmax_inputs` handles logical boundary tiles; the backend
adds its physical lane layout and checks numerical input bounds.
`assemble_row_softmax_outputs` preserves row/column order and trims invalid
lanes. The native CLI's `--row-softmax-tile-index` selects one row group and
records the source-to-tile decomposition for `native-verify` to replay. Whole
model orchestration and automatic conversion/placement remain separate stages.
The shared `render_row_softmax_c` helper emits the same logical packing and
tail trimming for generated C calls. The provider supplies physical lane layout
and launch services; failed multi-group calls retain outputs in private scratch
until every group succeeds.

`recognize_row_softmax` checks the parsed source max/subtract/exp/sum/divide
sequence, including its initializers, broadcasts and intermediate users. An
escaping intermediate prevents fusion; the result may still have multiple
consumers. Recognition records unresolved source reductions and math rather
than granting target equivalence. A separately selected precision policy can
bind that region to the target's row implementation. Staged graph lowering
rechecks the compiler-input signature, source identity and policy before native
selection. Shared C conversion supports unscaled BF16 RNE with explicit code
ranges, including signed zero and rejection before output writes.

## Backend handoffs for the remaining capabilities

| Capability | Merlin supplies or consumes | Backend work remains |
| --- | --- | --- |
| Temporary storage | Complete graph uses, physical footprints, completion events and memory budgets | Atlas places typed slots and spills together outside the existing PACK, boundary and DMA windows. General strided/aliased device buffers and application-plan integration remain separate work. |
| Register spilling | Resource failures can trigger a different legal region/tile choice | Tensor/scalar results, block arguments and entry controls have spill routes. Mutable CFG slots remain invocation-reserved; completed MXU state has raw save/restore, including explicit CFG edges. In-flight state remains unsupported. Model execution and selected RTL qualification are separate evidence tiers. |
| Looped tiling | Shapes, valid extents and numerical policy | Parameterized addresses, legal tails, accumulator continuation across the chosen loop representation. |
| Multi-tile attention | Explicit whole-row reductions and policy-preserving decomposition | Qualified reduction primitives, masks, scaling and execution evidence beyond one tile. |
| Numerical support | Preserve policy identity and reject incompatible routes | Qualified arithmetic and conversion implementations; no silent precision substitution. |
| Program size | Consume final image sizes and sequence separate programs through the agreed driver | Branch-range handling, safe split points, state materialization and a checked emitter per segment. |
| Persistent state | Account for state inputs/outputs and invocation ownership | DMA address windows and physical transfers for weights and KV buffers. |
| Model placement | Cover every source node/result with a supported route | Faithful device lowering for selected regions and explicit failures for missing support. |

For an IMEM boundary, require completed effects and materialized live values.
Never infer that MXU accumulator readout can be restored without changing the
computation. First use serial program replacement; instruction preloading and
concurrent replacement require separate hardware support.

## Repeated compiled stages

`session_bundle.load(root, runtime_inputs=False)` validates the existing
version-2 session description without requiring runtime tensor samples. Supply
a source-only directory containing the declared MLIR, weight metadata and child
state descriptions. This option does not call the evaluator-oriented
`session_bundle.generate` function.

`compile.session_dispatch.emit_session_dispatch` connects already compiled
stages through the ordinary bounded arena and C dispatcher. A
`SessionImplementation` identifies a synchronous wrapper and its runtime ports
by captured argument index; parameters may remain bound in the wrapper context.
Check those ports against the stage compiler's public interface before selecting
the implementation. The wrapper uses the normal C dispatch ABI:

```c
int stage_call(void *context, const void *const *inputs, void *const *outputs);
```

The emitter preserves stage order, all declared cross-stage edges, output
order and recurrent input/output mappings. Each iteration produces a new buffer
version; ordinary lifetime allocation decides which intermediate storage can be
reused. Repeated calls use one wrapper/context. A stage without an implementation
must contain only a typed return of its arguments; it forwards those buffers and
does not introduce a device launch. Missing computation is an error.

The emitted C/H driver and buffer description expose the last stage's outputs
and the final carried states. The caller supplies compiled wrappers, contexts,
aligned buffers and the arena. Failure handling follows `emit_program_c`: drain
or recover a failed endpoint before reusing its state or buffers. Results can be
partial after a failed call.

This is host orchestration. It does not fuse stages into one device image, prove
the stage arithmetic, load runtime samples, or implement a missing stage. Its
finite expansion budget bounds the number of calls in the host driver; device
loop formation and IMEM residency remain backend compilation decisions.

### Linking stage objects

Generated standalone stages may contain region helpers or bundled runtime
functions with the same names. Compile each stage C file separately. Before
linking them together, `compile.object_exports.localize_object_symbols` can
keep an explicit list of application entry points public and make the other
definitions local to that object. Select LLVM `nm` and `objcopy` from the host
toolchain. Keep every entry point used by the application, including binding,
state initialization and recovery functions.

The helper checks that selected exports exist and that unresolved library
dependencies remain unchanged. It records input/output and tool hashes, and
publishes a fresh object only after those checks. It does not resolve external
libraries, verify the stage arithmetic or build a platform ELF. The platform
still supplies its public driver interface, startup and linker script.

## Checks

The focused graph-memory tests cover asynchronous lifetimes, declared views,
reserved scratch, physical padding, persistent buffers and stale graph identity.
Two hundred deterministic tiny placement problems are also compared against an
independent exhaustive enumerator with three buffers, capacity 2–6 address units,
extents 1–3, alignment 1 or 2, and short bounded event ranges. Those are component
checks, not an Atlas or full-model execution result.

```sh
PYTHONPATH=src python -m pytest -q \
  merlin/tests/ir/test_buffer_allocation.py \
  merlin/tests/runtime/test_completed_arena.py \
  merlin/tests/runtime/test_arena_plan.py \
  merlin/tests/dse/test_global_planner.py
```
