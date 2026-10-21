---
title: Enable shared passes in model builds
kind: guide
status: current
owner: core
last_verified: 2026-10-21
related: [shared_pass_phase2, ../reference/shared_compiler_passes]
code_refs:
  - src/merlin/compile/baremetal_model.py
  - src/merlin/llvmlower/source_model.py
  - src/merlin/llvmlower/source_dense.py
  - src/merlin/transforms/dense_ir.py
  - merlin/tests/ir/test_dense_capture_metadata.py
  - merlin/tests/ir/test_dense_zero_splat.py
  - src/merlin/transforms/dense_row_batching.py
  - src/merlin/transforms/dense_row_batching_check.py
  - merlin/tests/ir/test_dense_row_batching.py
  - src/merlin/llvmlower/source_dense_group.py
  - src/merlin/compile/scheduling/compact_dense_group.py
  - src/merlin/compile/scheduling/compact_dense_group_check.py
  - src/merlin/compile/scheduling/compact_dense.py
  - src/merlin/compile/scheduling/compact_verification.py
  - merlin/tests/infra/test_compact_pe_weights.py
  - merlin/tests/infra/test_compact_pe_native.py
  - merlin/tests/infra/test_compact_zero_initialization.py
  - src/merlin/llvmlower/source_pointwise.py
  - src/merlin/llvmlower/source_multioutput.py
  - src/merlin/llvmlower/source_recomputation.py
  - src/merlin/llvmlower/source_recomputation_policy.py
  - src/merlin/runtime/source_recomputation_build.py
  - merlin/tests/runtime/test_source_recomputation_build.py
  - src/merlin/transforms/recomputation.py
  - src/merlin/transforms/recomputation_check.py
  - merlin/tests/gemmini/test_public_recomputation_execution.py
  - merlin/tests/gemmini/recomputation_host_work.py
  - src/merlin/compile/scheduling/multioutput_fusion.py
  - src/merlin/compile/scheduling/multioutput_fusion_check.py
  - src/merlin/llvmlower/source_rows.py
  - src/merlin/llvmlower/dealloc_lifetime.py
  - src/merlin/llvmlower/borrowed_buffer_calls.py
  - src/merlin/llvmlower/fresh_tensor_writer.py
  - src/merlin/llvmlower/device_build.py
  - src/merlin/llvmlower/device_shim.py
  - src/merlin/llvmlower/descriptor_result_identity.py
  - src/merlin/llvmlower/descriptor_result_preparation.py
  - src/merlin/llvmlower/lower.py
  - src/merlin/llvmlower/logical_borrowed_abi.py
  - src/merlin/llvmlower/convolution_routes.py
  - src/merlin/llvmlower/im2col_pack.py
  - src/merlin/verify/host_packing_storage.py
  - src/merlin/compile/scheduling/schedule_pipeline.py
  - src/merlin/compile/scheduling/temporary_storage.py
  - src/merlin/compile/scheduling/copy_elimination.py
  - src/merlin/compile/scheduling/dead_write_pipeline.py
  - src/merlin/compile/scheduling/dead_write_check.py
  - src/merlin/compile/scheduling/dead_writes.py
  - src/merlin/runtime/backends/spike_model.py
  - src/merlin/runtime/host_provider.py
---

# Enable shared passes in model builds

Use `spike_model.build(..., shared_source_lowerings=(...))` to let Merlin
discover and optimize supported computations during ordinary model compilation.
The backend supplies declared capabilities and instruction implementations;
Merlin selects source regions, runs the shared checks and inserts calls using
the existing fresh-result ABI.

## Select a supported family

| Selection | Accepted computation | Optimizations |
| --- | --- | --- |
| `SourceDenseLowering` | Static rank-two integer contractions with checked signed widening, modular arithmetic and zero initialization | Resource-derived blocking, operand reuse and compact counted loops |
| `SourcePointwiseLowering` | Static, contiguous, total integer elementwise operations whose scalar semantics appear in the selected software specification | Qualified fusion, temporary allocation, exact private-copy elimination, persistent configuration cleanup and exact device spills |
| `SourceRowLowering` | Canonical complete-row softmax under its original numerical policy | Resource-bounded chunks, retained intermediate state and complete-domain reductions |

These are optional backend integration interfaces. Existing targets need their
own qualified mappings and runtime bindings before selecting them. A target name
or an imported library does not enable these families.

Read the [pass contracts](../reference/shared_compiler_passes.md) for the exact
supported arithmetic, layouts, completion rules and numerical boundaries.

## Supply capabilities and implementations

Each selection needs current `SelectedSchedulingContracts`, explicit ownership
premises, resource limits, alignment and an execution domain (`host` or `device`).
Pointwise lowering additionally needs a storage adapter, instruction encoding
and final-byte decoder. Optional fusion, configuration cleanup and spills need
their declared semantics and independently qualified implementations.

For pointwise operations, `lower(source)` receives only the immutable shape,
scalar IR and scalar graph. Return `PointwiseProgram(kernel, objects, bindings)`
to map that computation to original target instructions and physical buffers,
or `None` when no mapping is available. Merlin independently checks the original
instruction sequence and the final decoded program against the source.
The mapping cannot change the selected capacities, decoders or budgets.

For dense contractions, supply a `DenseLoopStorage`, qualified `DenseLoopAdapter`
and static command-site budget. Merlin derives dimensions and the full ordered
reduction from the source, then uses the existing compact dense planner.
Assigned windows must account for live storage, packing and staging reservations.
Optional `DenseResources` controls restrict the search and cannot widen declared
capacities. Missing addressable storage, qualification or exact synchronous
continuation keeps the original route. Named matmul, floating arithmetic,
transposes, batches and nonzero seeds remain outside this source family.

### Retain compute-array weights across output tiles

The compact scheduling API accepts `reuse_pe_weights=True` when the original
qualified `DenseLoopAdapter` supplies `retained_weights=True`. Both default to
false. The capability must cover weight retention through operand loads,
stores and descriptor-only preloads. Constructing the Boolean data does not
qualify an implementation or enable asynchronous execution.

The selected schedule keeps a weight tile in the compute array while consuming
several M tiles into separate accumulators. The adapter must encode and project
the existing `Preload(weight_row=None, ...)` and `Compute(fresh_weights=False,
...)` forms. The symbolic checker establishes the exact latest resident weight
owner, rejects intervening fresh-weight clobbers, and retains the checks for
tails, ordered reductions, physical storage and complete outputs. It checks
counted loops without enumerating scalar products.

This option reduces compute-array weight reloads. It does not itself reduce
external DMA requests. The emitter starts each output reduction with an
overwriting first compute. The checker also accepts actual explicit
`AccumulatorZero` effects when the original qualified adapter supplies
`zero_initialization=True`. An accumulating first compute must read the exact
latest complete zero tile. Later reduction steps and output stores reject any
intervening zero reset. A zero also conservatively clobbers retained PE weights;
retention across it requires separate semantics. These checks do not describe
an accumulating instruction as an overwrite or establish target completion.

The source reader accepts explicit zero initialization through `linalg.fill`
over `tensor.empty`, or a static `tensor.splat` of an earlier local integer-zero
constant with the exact accumulator type. A zero tensor value does not establish
fresh storage, disjointness or permission to overwrite it. Existing observers of
that value remain live; lowering must provide its own qualified destination and
completion checks.

Known, typed capture annotations for source ancestry and tracing are retained
without using their values to choose transformations. Unknown attributes,
dispatch hints, numerical-policy hints and unknown effects remain barriers.
The native regression below exercises two contractions, three carried calls and
an additional observer of the original zero tensor through ordinary, mixed and
CPU lowering. It establishes source-level compatibility and native execution,
not a target instruction mapping:

```sh
python -B -m pytest merlin/tests/ir/test_dense_zero_splat.py \
  -k native_ordinary_source_model_preserves_observed_zero_splat \
  -o tmp_path_retention_policy=all -p no:cacheprovider \
  --basetemp "$TMPDIR/dense-zero-observer"
```

Dense checking covers source and schedule IR. Its pinned C emission does not
establish arbitrary target C or final machine-code semantics. The backend must
independently qualify instruction bodies and ABI effects and supply applicable
final-image checks. Static command sites are not final machine instructions.

An original software specification can select `merlin.dense_schedule.v2` for
explicit external pointer transport. It must declare the address operand and
address space for each LHS load, RHS load and output store, with matching
full-width address fields in the selected ISA. Its logical pointer ABI must
declare synchronous, bounded, dense row-major accesses: read-only inputs and a
fully written output. The existing `merlin.dense_schedule.v1` contract remains
closed and accepts integer coordinate operands only.

For V2, the ordinary dense scheduler binds each pointer from the current
contraction's dimensions and element widths. It checks the actual buffer role
and byte offset returned by the adapter, including a pointer supplied by the
encoder. The existing complete-domain checker still verifies ranges, tails and
reduction coverage. This correspondence check supplies neither executable
instruction semantics nor runtime ownership qualification. V2 currently refuses
output readout fusion. Selecting a
software contract does not change existing authoring exports or permissions.

All families use two implementation callbacks:

- `prepare(bindings, directory)` supplies implementations for the checked
  bindings in an invocation-private directory.
- `build(bindings, context)` compiles their objects and returns the existing
  `HostProviderBuild`, bound to the current prepared model, LLVM and model object.

Each binding contains a checked compilation and its fresh-result writer ABI.
The backend remains responsible for instruction bodies, descriptor handling,
strides, final pointer alignment, ownership, complete writes and completion.
Qualification identities record independent evidence; a hash does not establish
that evidence.

### Batch contiguous rows sharing a right operand

`plan_source_dense_row_batching(module, context, budgets)` proposes a checked
tensor-level alternative for several exact integer contractions. Their left
operands must partition one immutable, contiguous source into complete rows;
their right operands must have the same proven value and coordinate mapping.
The planner creates one ordinary matmul per admitted batch and exact result slices. It introduces
no concatenation or packing copy and preserves each original reduction order,
output observer and zero-initializer observer.

```python
from merlin.transforms.dense_row_batching import (
    DenseRowBatchingBudgets, plan_source_dense_row_batching,
)

plan = plan_source_dense_row_batching(
    original_module, original_context,
    DenseRowBatchingBudgets(max_members=4),
)
if plan is not None:
    plan.revalidate()
    candidate_module = plan.candidate
```

The budgets bound source elements, candidate members, batches, index range and
constructed IR. Unsupported views, unknown annotations/effects, incompatible
arithmetic and unsafe intervening operations retain the original route.
The independent checker reconstructs the actual batch, its complete output
partition and unchanged surrounding SSA and metadata. Source, context, budgets
and checker changes invalidate the plan.

This explicit planner does not install itself in saved-model builds. An original
integration must consume its candidate before device outlining and establish
placement for the new contractions. Native paired tests cover complete outputs,
modular overflow, input guards and carried state. They establish tensor algebra
and ordinary CPU compilation, not physical operand retention, Gemmini execution,
or profitability. Actual target loads and complete-model comparisons are still
required before claiming accelerator reuse.

### Reuse an activation across independent contractions

An original host integration can supply `SourceDenseLowering.lhs_grouping`:

```python
from dataclasses import replace
from merlin.llvmlower.source_dense_group import SourceDenseGrouping

selection = replace(
    original_dense_selection,
    lhs_grouping=SourceDenseGrouping(
        storages=original_member_storage_assignments,
        encode_member=original_member_encoder,
        project_member=original_member_decoder,
        retention_law=original_qualified_retention_law,
        max_members=2,
    ),
)
# Pass selection through the ordinary shared_source_lowerings argument.
```

Discovery groups adjacent admitted integer contractions that read the identical
activation SSA value. It preserves each complete ordered reduction, separate
accumulator and result, and every downstream observer. The existing planner
selects counted loops from current dimensions and resources; the group loads an
activation panel once, then completes every member using that retained panel.
It introduces no concatenation buffer. The fresh-result bridge makes one
borrowed call with a separate allocation for each output.

This first domain requires the complete K dimension to fit in the assigned LHS
window. Member RHS and accumulator windows must be disjoint, including physical
backing aliases, bank limits and live reservations. The original V2 pointer
contract must expose actual external buffer operands. Missing full-K capacity,
retention qualification, synchronous completion or exact continuation retains
individual contractions. Intervening operations are barriers. Logical-ABI,
offload-plan and convolution compositions currently retain the singleton route;
unsupported arithmetic and shapes retain their source route.

The `DenseLHSReuseLaw` must independently establish that member operations do not
change the retained LHS, that independent members can interleave, and that omitted
loads have no faults or extra observable effects. Bind it to the original
hardware, numerical policy and adapter qualification. RTL capacity alone cannot
establish these software rules. `encode_member` emits existing actual instruction
operands, including member buffer pointers; `project_member` reconstructs those
effects. The shared checker verifies the complete physical stream and each
member's reduction, with one persistent model of shared storage. A member label
or chosen plan is insufficient proof of retention.

Source/SSA, selected contracts, original callback identities and code, source
files and the issued checked program remain bound across preparation and build.
Changes require replanning. Final instruction implementation, descriptor effects,
linked execution and target qualification remain provider obligations.

Run the ordinary connected native comparisons and the refusal/mutation checks:

```sh
python -B -m pytest merlin/tests/infra/test_shared_lhs_source.py \
  merlin/tests/infra/test_shared_lhs_provider.py \
  merlin/tests/infra/test_compact_dense_group.py \
  merlin/tests/infra/test_source_dense_group_freshness.py \
  merlin/tests/infra/test_source_dense_group_custody.py \
  merlin/tests/infra/test_source_group_context.py \
  -o tmp_path_retention_policy=all --basetemp "$TMPDIR/shared-lhs-check"
```

These tests use two neutral rectangular primitives, different integer widths,
separate and shared physical memory, complete recurrent outputs and preserved
inputs. Runtime instruction operands and byte effects establish requested
payloads and command counts. They establish no physical accelerator support,
bus traffic or latency. This optional host entry changes no frozen V6/V7 SDK
inventory or authoring permission. Phase 2 exposure requires a reviewed release
fixed before authoring, with the same library and checks for every treatment.

### Retain one producer for two outputs

An original integration can select an optional composition on its pointwise
lowering:

```python
from dataclasses import replace
from merlin.llvmlower.source_multioutput import SourceMultiOutputLowering

selection = replace(
    original_pointwise_selection,
    multi_output=SourceMultiOutputLowering(
        lower=original_composed_instruction_mapping,
        composition_law=original_qualified_composition_law,
    ),
)
```

Merlin discovers three adjacent static integer computations: one producer and
two independent consumers of its exact result. Both consumer results and every
later observer are preserved. The producer must have no other observer or
escape. A published producer, intervening operation, unavailable capability or
unsupported computation retains individual lowering. `multi_output=None`
retains the existing discovery route.

The mapping receives `MultiOutputSource`, including the original ordered inputs,
outputs, scalar computations and input indices. Return
`MultiOutputProgram(kernel, objects, bindings)` or `None`. The first domain uses
existing synchronous single-result instructions: compute each producer element
once into private storage, then retain each fragment's physical generation for
both consumers.
It requires no new multi-result instruction or concatenation buffer. Both final
buffers use the existing plural fresh-result bridge in one borrowed call.

`MultiOutputCompositionLaw` binds the selected configuration, numerical policy,
adapter qualification and exact instruction/scalar roles. Its explicit premises
cover complete projected effects, synchronous private composition and the
absence of faults or observations from suppressed intermediate transfers.
These require independent implementation evidence; RTL capacities and passing
numerical samples do not establish them. Runtime ownership, input disjointness,
complete writes and borrowed-buffer effects remain separate requirements.

The shared checker follows actual byte provenance through physical backing
aliases, verifies complete producer and output coverage, and checks the final
decoded program after allocation and scheduling. Recomputing the producer,
copying it to another backing between consumers, omitting an output, changing
addresses or replacing scalar arithmetic cannot satisfy this check. Source SSA,
selected context, callback code and issued program remain bound through build;
changes require replanning. Unknown arithmetic and asynchronous arithmetic
remain barriers. Unsupported allocation retains the individual source route.

Run the ordinary paired native cases and mutation checks with:

```sh
python -B -m pytest merlin/tests/infra/test_multi_output_source.py \
  merlin/tests/infra/test_multioutput_fusion.py \
  merlin/tests/infra/test_source_multioutput_freshness.py \
  merlin/tests/infra/test_source_multioutput_early_callbacks.py \
  -o tmp_path_retention_policy=all --basetemp "$TMPDIR/multi-output-check"
```

The two neutral byte-machine models use different integer widths and separate
or aliased private memory. Each comparison preserves all nine outputs across
three calls, including state, position and read-only inputs. It checks final ELF
program bytes, runtime addresses, producer generations and descriptor-copy
requests. Arithmetic requests remain unchanged; intermediate publication,
descriptor copies and dispatches can disappear. These tests qualify neither a
physical accelerator nor latency. A backend without the consumed-program and
qualified composition contracts remains an open integration. This host entry
does not extend frozen SDK inventories or authoring permissions; Phase 2 needs
a separately reviewed release shared by every treatment.

### Trade an intermediate for bounded recomputation

`plan_source_recomputation` proposes a separate alternative for a pure integer
producer with two adjacent pointwise consumers. It clones the producer's exact
ordered scalar operations into each consumer and removes its tensor result.
Both consumer results, surrounding operations and later observers remain.
This deliberately increases arithmetic; it is different from the one-producer
physical composition above.

The original host supplies `RecomputationPermissions` and
`RecomputationBudgets`, along with the current `PipelineContext`. The five
explicit permissions cover unobserved, fault-free repeated evaluation and
suppressed materialization, and stable inputs throughout the region. A label,
RTL capacity or apparently pure backend operation cannot supply these premises.
Budgets limit candidate discovery, tensor elements, repetitions, producer work,
added scalar evaluations, complete cloned IR construction and scalar IR bytes.
The first supported domain has two consumers, static identity indexing and
total integer scalar operations. Unknown effects or reductions inside that
region, observed producer results, intervening operations and unsupported
layouts preserve the original route by returning no candidate. Other source
operations must retain their existing observable expressions and opaque effect
epochs. Invalid or stale issued plans refuse.

```python
from merlin.transforms.recomputation import plan_source_recomputation
from merlin.llvmlower.source_recomputation import SourceRecomputationSelection
from merlin.llvmlower.lower import lower_model

plan = plan_source_recomputation(module, context, permissions, budgets)
selection = None if plan is None else SourceRecomputationSelection(
    plan,
    verify_current=original_live_owner,
    comparison=original_complete_cost_decision,
)
result = lower_model(
    source_text, workdir, targets=("host",), vectorize=False, parallel=False,
    source_recomputation=selection,
)
```

`comparison` is an optional original `OffloadCostDecision` from
`compare_complete_offload_cost`. It must bind this exact source, context and
both alternative identities. Missing evidence retains the byte-exact original
input. Inconclusive or unfavorable complete-cost bounds also retain it;
selecting the candidate requires its upper bound below the original's lower
bound in the same declared unit and regime. This adapter invents no costs or
unit conversions. A checked candidate can be compiled separately for
calibration; that trial does not authorize automatic selection.

`original_live_owner` receives keyword arguments `stage`, `source`, `plan`,
`settings`, `comparison` and `result`. It must return `None`, raising on any
missing or changed premise. `stage="entry"` identifies checks without a completed
result; these can repeat throughout compilation. `stage="return"` supplies the
actual `LowerResult` before completion. The owner must bind current
tools, numerical policy, runtime, ABI, compiler options and measurement scope.
At return it must also establish whether costs observed for an earlier image
apply to the actual emitted image. Callback presence is not qualification.

The ordinary path retains the input, selected source and selection report as
`source-recomputation-*` artifacts. Source/SSA identity, context, implementation
bytes, callback code and complete-cost ownership are reopened through emission;
actual returned artifacts are pinned before completion callbacks and checked
afterward. Rewrites invalidate source-dependent analyses through the existing
pipeline session. Mixed graph policies, other explicit source rewrites,
vector/parallel schedules and transformation schedules currently refuse this
selection. An unselected call keeps its existing behavior.

Removing a logical tensor does not prove an allocation or total traffic saving.
Ordinary bufferization may already share its storage with a later output, and
recomputation can increase input reads. Inspect the consumed LLVM, linked
products and complete execution, reporting scalar IR payload separately from
machine traffic and timing. This route supplies no opaque target-instruction
interpretation, hardware support or formal Phase 2 admission. Frozen SDK
exports remain unchanged; Phase 2 exposure needs its own reviewed release.

For an integration that prepares source before lowering,
`SourceRecomputationPolicy` freezes the original permissions, budgets, context,
verifier and optional complete-cost resolver. Its one `bind` attempt accepts
the actual prepared source and original lowering settings, then constructs the
existing issued plan and selection. Unsupported regions retain the original;
later changes refuse rather than refreshing that plan. Costs must match the
new prepared plan's stamp and alternative identities.

For saved-model compilation, wrap the policy in one live
`SourceRecomputationBuild` and pass it through the ordinary API:

```python
from merlin.compile.baremetal_model import compile_saved_model
from merlin.llvmlower.source_recomputation_policy import SourceRecomputationPolicy
from merlin.runtime.source_recomputation_build import SourceRecomputationBuild

policy = SourceRecomputationPolicy(
    context, permissions, budgets,
    verify_current=original_lowering_owner,
    comparison_for=original_complete_cost_resolver,
)
build = SourceRecomputationBuild(policy, verify_final=original_final_image_owner)
result = compile_saved_model(
    **ordinary_model_arguments,
    source_recomputation=build,
)
```

The same live object reaches the inner builder. It binds once to the final
prepared source after device outlining, using the actual lowering settings.
An earlier source plan or cost decision cannot be relabeled for this prepared
program. No candidate, missing costs or inconclusive bounds retain the exact
prepared original. `source_recomputation=None` preserves the existing route.

The builder retains exact source-stage audits, the actual LLVM submitted to
the model-object command, that object in the final link, the completed recipe
and final executable. With full readback selected, execution retains console
and output files before calling an owner that could modify them. Original
callback, source, settings
and product checks run throughout compilation; changed products refuse.
These checks establish artifact correspondence, not semantic equivalence of
arbitrary LLVM or a compiler that ignores a selected rewrite.

`original_final_image_owner` receives keyword arguments `stage`, `context`,
`selection`, `products` and `execution`. It returns `None` or raises. The
`linked` stage checks the completed image; `executed` additionally receives the
ordinary execution result. This separate owner must qualify the actual
arithmetic, ABI, target execution and any transfer of earlier measurements to
the final image. The low-level `original_lowering_owner` retains its existing
entry/return contract. Saved build records cannot restore either live owner.

The first build integration supports the scalar serial route and refuses
explicit competing source policies, custom preparation or providers, profiling,
host bitcode transport and late LLVM transforms. Ordinary default packing and
gather rewrites remain enabled.
Select a fresh policy and build object for each compilation. This integration
changes no frozen SDK export, numerical threshold or Phase 2 authoring grant.
Pure transport and mutation tests do not qualify a target execution path.

The opt-in Gemmini calibration regression uses a public integer producer feeding
two contractions, with three carried-state invocations per arm. It compiles
separate original and checked-candidate source copies through the unchanged
generated compiler. Ordinary default optimizations remain enabled. All six
input buffers and nine outputs are checked, including signed narrowing and
state wraparound. A submitted-LLVM witness joins the producer and epilogue
loops to the actual contraction arguments; unknown effects, intervening writes,
early frees and changed input versions refuse. Quiet and traced execution use
the same final admitted ELF. Quiet runs supply complete numerical readback;
diagnostic traces supply retired-instruction evidence. The selected diagnostic
runtime can interleave debug bytes with output frames, so those trace streams
do not supply numerical admission and must not be repaired into passing output.

Configure the existing toolchain and explicit original runtime inputs described
in [portable generated-target admission](portable_generated_target_admission.md).
The test consumes the same closed runtime-selection data as the public packing
regression; that file supplies inputs, never restored admission authority:

```sh
MERLIN_TEST_GEMMINI_RECOMPUTATION_SELECTION=/absolute/path/selection.json \
MERLIN_COMPILER_DRIVER=python \
python -B -m pytest merlin/tests/gemmini/test_public_recomputation_execution.py \
  -p no:cacheprovider -o tmp_path_retention_policy=all \
  --basetemp "$TMPDIR/recomputation-calibration"
```

Missing selections skip this opt-in regression and establish no qualification.
The Python lowering transport is required for the frozen captured-quantization
policy; the scalar `stock-cli` transport rejects that policy option.
The selected path constructs a fresh live admission service. This experiment
calibrates an explicit legal alternative; it does not automatically select a
profitable rewrite through `compile_saved_model`. Costs remain unknown. Repeating
a producer with mixed-width inputs may increase requested scalar bytes even
when its intermediate disappears. The witness counts submitted LLVM work,
not final machine traffic; functional Spike counters establish no latency gain.

Run the paired native source models and mutation checks with:

```sh
python -B -m pytest merlin/tests/ir/test_recomputation.py \
  merlin/tests/ir/test_source_recomputation_selection.py \
  merlin/tests/infra/test_recomputation_llvm_payload.py \
  merlin/tests/infra/test_source_recomputation_model.py \
  -o tmp_path_retention_policy=all --basetemp "$TMPDIR/recomputation-check"
```

The neutral integer models check all outputs, preserved inputs and carried state
through ordinary native lowering. Their retained LLVM observations distinguish
removed intermediate accesses from repeated input reads and existing buffer
reuse. Warm complete-call observations use alternating arm order and the same
reset inputs. The test does not require the candidate to win: overlapping
bounds must preserve the original image. These fixtures establish neither
device instruction scheduling nor a Gemmini timing result.

### Avoid compatible dense operand packing

An optional `SourceDenseLowering.logical_abi: LogicalBorrowedABI` lets Merlin
own the ranked descriptor bridge. The provider implements the explicit
`binding.logical_kernel_symbol` taking three logical row-major pointers;
Merlin separately generates, pins, compiles and links the bridge. The provider
must not also define that bridge. No backend C is inspected to infer effects.

The original dense software declaration must include:

```yaml
logical_pointer_abi:
  schema: merlin.logical_pointer_abi.v1
  layout: dense_row_major
  alignment_bytes: [2, 2, 4]  # Example: independently required A/B/C alignment.
  access: [read_only, read_only, fully_written]
  bounds: logical_elements
  completion: synchronous
```

RTL geometry and capacities do not establish this pointer ABI. Supply the exact
parsed declaration, independently qualified `BorrowedBufferEffects`, a
`format_qualification_identity` for bounded logical reads and complete writes,
and an explicit `max_stack_bytes` fallback-array budget. The effects must match
the selection's original `borrowed_effects`. These are conditional premises
about the implementation; a declaration or numerical test alone cannot qualify
them. The current bridge supports positive static rank-two i8/i16/i32 buffers
and the selected 64-bit ranked caller ABI.

With `borrow_contiguous=True`, each actual descriptor independently bypasses
packing when its strides are exactly row-major and its pointer **including the
offset** meets the declared alignment. Other positive-stride layouts or
misaligned pointers use bounded typed scratch and byte-preserving copies.
`borrow_contiguous=False` forces the same bridge to pack, providing a paired
baseline with identical primitives, contracts and schedules. A budget that
cannot hold the complete fallback retains the original source route.

Before calling the primitive, the bridge checks extents, signed index and
address arithmetic, output injectivity and disjoint output and input bounding
ranges. Malformed descriptors and output overlap trap. Read-only inputs can
alias at this bridge boundary; source scheduling still requires its own
original ownership and disjointness premises. Accessible allocation sizes are
not encoded in ordinary ranked descriptors and remain an independent caller
premise. The scratch budget includes array storage and alignment padding;
compiler frame overhead and callee stack still require separate budgeting.

Optional `observe_transfers=True` additionally requires separately qualified
`observer_effects` and `observer_preserves_logical_state=True`. The provider
implements `binding.logical_observer_symbol(read_bytes, write_bytes, mask)`;
the bridge calls it after primitive completion and output copies. Counts cover
executed logical copy-loop bytes. They exclude guard accesses and logging and
do not measure physical bus traffic or accelerator latency. Observation is off
by default.

This route does not cache prepared weights between calls, infer invocation
invariance or support asynchronous primitive completion. A backend without the
qualified logical-pointer contract retains its existing ABI implementation.

With the ordinary toolchain configured, reproduce its neutral paired model,
fallback and mutation checks with:

```sh
python -B -m pytest merlin/tests/infra/test_logical_dense_model.py \
  merlin/tests/infra/test_source_logical_abi.py \
  merlin/tests/ir/test_logical_borrowed_abi.py -p no:cacheprovider \
  --basetemp "$TMPDIR/logical-buffer-check"
```

The model report separates build and functional-simulator wall time and sampled
process-tree RSS. Build includes native qualification and linking; simulator
time includes diagnostic instrumentation. Neither measures accelerator latency.

For a loop that replaces a carried buffer, a source-family selection can also
supply `borrowed_effects: BorrowedBufferEffects`. The original provider contract
must separately establish `no_retained_aliases`, `no_releases` and
`complete_before_return`, with a `qualification_identity` for that ABI
implementation. All three premises must be explicit. Instruction completion
flags, read/write annotations and numerical test success do not establish ABI
effects. The default is `None`, which grants no borrowed-call lifetime facts.

Merlin freezes these premises with the original selected source and contracts.
At the checked post-deallocation boundary it requires exact ranked declarations
and checks each surviving fresh-result wrapper's positional forwarding body.
Grants attach to actual calls in checked root functions; a shadowed symbol does
not inherit them. The initial route supports one fresh result and no private
workspace. Unknown calls, ABI changes and unsupported scopes refuse.
This selection requires the checked deallocation stage; it does not disable
verification or grant runtime support.

### Preserve multiple fresh results in one external call

`FreshTensorWriterContract.result_arguments` optionally names the ordered
writable argument positions returned by a tensor declaration. Its default empty
tuple retains the existing single `result_argument` behavior. For example, a
declaration taking `(input, state, output, next_state)` can return
`(next_state, output)` with `result_argument=3`,
`fully_written_arguments=(2, 3)` and `result_arguments=(3, 2)`.

This mode requires explicit original `BorrowedBufferEffects`. Every writable
tensor must appear exactly once in the result mapping, with the declared result
type and order. Each writable actual argument must remain a sole-use
`tensor.empty`, unless the caller separately selects the existing
initialized-writer permission. Merlin creates distinct fresh buffers, invokes
the borrowed void implementation once, and returns every mapped buffer through
the normal upstream bufferization and deallocation pipeline. The caller must
still establish full writes and the implementation's ownership/completion
effects; matching descriptors or successful numerical tests do not establish
those premises.

The original caller can bind `BorrowedBufferResults` through `from_writers`.
Its version 2 roster has one row per borrowed call, with all ordered result
positions. Singleton-only rosters retain version 1 unchanged. The native
checker verifies every appended output parameter, the exact forwarding body,
readonly/write argument roles and distinct fresh output allocation roots.
Unknown owners, shared output roots, extra effects and changed releases refuse.
This initial plural scope requires direct fresh allocations at actual call
sites; caller-owned output block arguments need a separate disjointness proof.

`prove_lowered_writer_results` separately checks the actual final LLVM wrapper
and every output descriptor's constant offset, shape, strides and allocation
extent. Exact contiguous readonly caller descriptors can be forwarded, but
their runtime shape, accessible extent, lifetime and readonly behavior remain
original caller obligations. Allocator freshness/alignment and implementation
effects are also required separately. The single-result
`prove_lowered_writer_result` API continues to refuse plural contracts.

This is a compiler transport mechanism, not automatic horizontal matmul fusion
or target retention qualification. Source grouping, complete reductions and
actual cross-contraction load retention need their own checks. Existing frozen
authoring releases acquire no additional exports or permissions.

### Expose checked device result identity

For the ordinary generated contraction-shim route, explicitly select:

```python
from dataclasses import replace
from merlin.runtime.backends import spike_model

# The original integration supplies this live factory for the selected kernel
# and runtime. It is not reconstructed from candidate IR or a saved report.
routing = replace(
    original_device_routing,
    descriptor_results=True,
    descriptor_effects=original_borrowed_effects_factory,
)
built = spike_model.build(
    capture_directory,
    build_directory,
    backend="scalar",
    device=routing,
    full_readback=True,
)
```

`DeviceRouting.descriptor_results` defaults to `False`. The original routing,
package, kernel ABI, numerical policy, runtime and final linked-ELF admission
must already be selected. This option changes how host bufferization sees the
returned descriptor; it supplies no target instruction or kernel implementation.
Selecting it without an original `descriptor_effects` factory refuses before
device compilation. Leave `descriptor_results=False` to retain the original route.

The factory receives `(prepared_source_path, actual_device_build)` and returns
the existing live `BorrowedBufferCalls` selection. Its original validator must
qualify no retained aliases, no releases and completion before return for the
actual kernel/runtime. The source hash, every kernel object, checked shim C and
object, positional memref types and result argument must match that build.
Merlin revalidates the original owner and its source/object pins across lowering
and final admission. A qualification name and three booleans transport an
independent contract; they do not establish its truth.

The generated checked C forwarder calls the original shim and requires its
returned allocated pointer, aligned pointer, offset, sizes and strides to match
the supplied result descriptor field by field. A mismatch traps. The original
shim's shape, stride and overlap checks still execute. Read/write
annotations or matching tensor types alone do not establish this identity.
The forwarder does not establish full writes, no retained aliases, no releases
or synchronous completion. Those remain original implementation obligations.
These obligations are mandatory for the rewrite, even when the ordinary
unoptimized ABI already assumes them. RTL geometry, read/write annotations and
successful functional execution cannot replace their independent qualification.

The first supported ordinary route has static, unencoded rank-two signless
integer tensors, two read-only inputs and a third result argument. Each selected
result initializer must be a sole-use `tensor.empty`, or a standard fill of a
sole-use empty root. Existing initialization is preserved, including initialized
values read by the original call. Shared or observed initializers and unsupported
calls retain their original form; symbol collisions, ABI mismatches and stale
selections refuse. No scalar arithmetic is reordered or removed.

Selected device kernels and the checked shim are compiled once before host
lowering; the final link reuses that same retained build. The source rewrite runs
after automatic destination and permutation preparation, before preprocessing
and CSE. It protects selected fresh allocation roots and exposes the checked
identity through a tensor wrapper and a void descriptor call. The selected
standard pipeline runs `drop-equivalent-buffer-results` after bufferization and
before `buffer-results-to-out-params`; any reuse follows from that actual wrapper
body. Unsupported initializers do not receive an alias assertion.

The current composition requires scalar serial Python lowering with an explicit
64-bit descriptor layout. Group/catalog routes, opaque post-offload ABI rewrites
and host LLVM rewrite hooks refuse this selection. The low-level
`lower_model` and `lower_model_file` APIs also accept
`descriptor_result_identities=(original_identity, ...)`; their empty tuple is
unchanged behavior. That tuple must come from an original checked bridge and its
actual C/object binding, with explicit `borrowed_effects`. Missing effects keep
the original call and source bytes; malformed or stale effects refuse.
Constructing a `DescriptorResultIdentity` or loading a
saved report supplies no such proof. Borrowed-call, shared-source-provider,
profiling, vector/parallel and static-arena compositions remain unsupported at
this low-level boundary.

Inspect `lower/descriptor-result-identity.json`,
`lower/descriptor-identity.input.mlir`,
`lower/descriptor-identity.selected.mlir` and `lower/lowering_recipe.json`.
The ordinary device build retains `device/descriptor-result-effects.json` with
the original borrowed-call record and pins. This is provenance, not an admission
grant.
`LowerResult.stats["descriptor_result_identity"]` joins the retained report to the
actual upstream source and returned LLVM. The ordinary build also checks the
original typed roster against the actually compiled shim C and object, retains
the kernel compilation receipts and pins the implementation/checker sources in
the compilation recipe. Changed source, library, callbacks, report or products
refuse; a report's rewrite count alone is not evidence of eliminated allocation
or copying. Inspect final LLVM and executed effects separately.

Neutral native tests establish this conditional host mechanism, including live
results, carried state and checked descriptor failures. They do not establish
Gemmini execution or qualification for this route. Existing V6/V7 authoring
grants remain unchanged. Phase 2 exposure needs a separately reviewed, explicit
versioned library/service selection and the original permissions; this option
and its reports create no implicit access or performance claim.

The public V26 Gemmini inputs do not currently supply the required live
kernel/runtime effects owner. Its opt-in destination qualification test therefore
reports that missing prerequisite before fresh capture/admission. This is open
integration work, not evidence that Gemmini cannot support the optimization.

### Bound convolution activation packing

An optional dense preparation route replaces a sole-use static integer
`im2col` gather and reshape with serial window panels. It preserves the source
channel/kernel reduction order, strides, dilation, tails and exact zero seed.
Existing explicit padding remains unchanged. Unknown operations between the
gather and contraction are barriers; no device gather or padding capability is
inferred from a matmul instruction.

Starting with the same qualified dense selection, choose an explicit host
activation-payload budget and independently qualified allocation effects:

```python
from dataclasses import replace
from merlin.verify.host_packing_storage import PackingAllocationEffects

allocation = PackingAllocationEffects(
    allocator_symbol=runtime_allocator_symbol,
    deallocator_symbol=runtime_deallocator_symbol,
    qualification_identity=original_allocation_qualification,
    pointer_bits=selected_host_pointer_bits,
    fresh_private_allocations=True,
    successful_allocations=True,
    releases_only_via_deallocator=True,
)
bounded = replace(
    dense_selection,
    max_convolution_packing_bytes=activation_payload_budget,
    convolution_route="auto",
    packing_allocation_effects=allocation,
)
baseline = replace(bounded, convolution_route="full")
```

The allocation premises must establish disjoint, contiguous live allocations
of the requested extent, successful representable storage, and releases only
through the selected deallocator. Bind qualification to the actual runtime
implementation and linked object. Allocator names and an identity string do not
establish these facts. The original `borrowed_effects` must separately establish
synchronous completion, no retained aliases and no releases by the dense ABI.
RTL dimensions and capacities establish neither runtime contract.

`auto` retains full packing when it fits; otherwise it searches for the largest
resource-legal panel within the budget. Each main and tail shape must satisfy
the selected dense primitive and physical-storage contracts. `panel` requests
that search even when full packing fits. `None` disables normalization; `full`
retains the original source route. Search and source growth are bounded by
`max_convolution_candidates` and `max_convolution_added_operations` (both default
to 4096). This is a deterministic storage policy with unknown timing costs.

The budget counts the **sum of distinct main and tail activation-panel logical
payloads per source island**. It excludes padding, output tiles, provider/ABI
scratch, allocator overhead and other islands. It is a candidate bound, not a
hard RAM limit: unavailable legal panels or exhausted search/growth budgets
retain the full route, which can exceed it. The preparation receipt's
`source_normalization[].outcomes` records requested budget, retained or selected
payload, route, reason and `met`, `exceeded` or `unknown` status. Unsupported
source costs remain unknown.

Every selected panel arm must acquire its checked dense binding. If any arm
declines, Merlin discards the private candidate and replans the original full
source once. Implementation preparation receives only the final bindings.
Malformed source, stale contracts and incorrect realizations diagnose rather
than qualifying an alternative.

Before the provider build callback, the selected panel route checks actual final
host LLVM and its pinned model object. It follows the original positional RHS
descriptors to dominating allocation roots outside CFG cycles, closes pointer
uses under the original borrowed effects, and checks releases. A loop-carried
source tensor or an intermediate bufferization checkpoint alone is insufficient.
The retained native report separates logical payload, actual allocator-requested
bytes and alignment/request overhead; only logical payload is budgeted.
Missing allocation qualification or a failed final check is a diagnostic.
For each actual allocation root, native logical payload is the maximum selected
descriptor payload. Requested backing is charged in full to each participating
island; a combined report counts a shared root once. The legacy
`alignment_overhead_bytes` field means requested bytes minus selected logical
payload, so it can also include slack or spare capacity.

Neutral paired recurrent-model tests cover i8/i16, rectangular primitives,
padding, stride, dilation, tails and overflow, checking every output and carried
state against an independent ordered integer reference. They demonstrate a
storage tradeoff: smaller final activation backing can accompany more dispatch,
weight packing, primitive commands, code and compilation work. They establish
neither accelerator latency nor Gemmini support. See the
[storage scope](../reference/shared_compiler_passes.md#bounded-convolution-packing)
and reproduce the source/fallback and ordinary-model checks with:

```sh
python -B -m pytest merlin/tests/infra/test_source_convolution_routes.py \
  merlin/tests/infra/test_convolution_route_outcomes.py \
  merlin/tests/infra/test_source_convolution_fallback.py \
  merlin/tests/infra/test_host_packing_storage.py \
  merlin/tests/infra/test_convolution_route_model.py -p no:cacheprovider \
  --basetemp "$TMPDIR/convolution-storage-check"
```

These trusted model-binding controls grant no new Phase 1/2 author imports.
An experiment must explicitly select an equal, versioned API grant before
authoring agents can tune them; existing frozen profiles retain their boundaries.
The target-dialect pass and consuming-emitter route remains the primary device
integration boundary.

## Build the model

After constructing selections from the backend's qualified implementations:

```python
from merlin.runtime.backends import spike_model

built = spike_model.build(
    capture_directory,
    build_directory,
    backend=selected_host_backend,
    shared_source_lowerings=(dense_selection, pointwise_selection, row_selection),
    full_readback=True,
)
```

Supply a family alone or combine supported selections. The existing
`shared_row_lowering=row_selection` option remains available for row-only builds.
Do not select the same family twice or combine these options with explicit
`prepared_model_transform` or `host_provider_builder` hooks.

Merlin discovers selected families on one original source snapshot and applies their
nonoverlapping replacements in one transaction. Registration order does not
change discovery or rewriting. The existing provider linker checks the combined
object roster, function ABIs and required symbols. An active device selection
also requires the original `DeviceRouting` and `LinkedElfAdmissionService`,
including admission of the final linked image.

The saved-model API accepts the same tuple:

```python
from merlin.compile.baremetal_model import compile_saved_model

receipt = compile_saved_model(
    capture=capture_directory,
    package=selected_host_package,
    board_catalog=board_catalog,
    board=selected_board,
    dts=host_dts,
    output=fresh_output_directory,
    target=selected_target,
    run="spike",
    arena_mb=arena_mb,
    readback="full",
    shared_source_lowerings=(dense_selection,),
)
```

The output directory must be fresh and below the configured `MERLIN_OUT_ROOT`.
To judge every result, supply the capture's complete `goldens.npz` and
`output_order.json`; a reference for result zero does not validate other results
or carried state. Existing board, host ISA, simulator and device-admission rules
still apply. Selecting source passes grants no new candidate authoring access.

## Inspect results and handle refusal

To compare storage allocation independently, keep the original hardware,
software, source inputs and other controls identical. Derive a baseline and an
optimized selection from the same qualified pointwise selection:

```python
from dataclasses import replace

fixed = replace(
    pointwise_selection.budgets,
    max_fusions=0, max_spills=0,
    eliminate_configuration=False, eliminate_copies=False,
)
baseline = replace(pointwise_selection, budgets=replace(fixed, optimize_storage=False))
optimized = replace(pointwise_selection, budgets=replace(fixed, optimize_storage=True))
```

The disabled allocator still checks the original addresses, capacities, aliases
and completion. Its baseline must already fit the same selected hardware;
changing capacities between arms is a different experiment. To compare exact
copy elimination separately, keep `optimize_storage=True` in both arms and vary
only `eliminate_copies`.

Private-copy forwarding requires a complete synchronous exact copy, physically
shared backing and an unchanged source until the destination's final read.
Different memories, conversions, asynchronous effects and unknown ownership
retain the copy. If extending the source lifetime cannot be allocated, the
compiler restores the original composition and replans within its repair budget.

Inspect final decoded addresses and executed accesses to establish storage
reuse. A smaller touched address range does not reduce a fixed memory bank's
declared capacity or prove lower host heap usage. Reservations for packing and
staging still count toward physical high-water bounds. Report requested
instruction transfers and host descriptor packing separately: deleting an
internal copy does not remove the surrounding ABI packing or other model work.
Summing those observed payloads describes a measured subset, not complete
model traffic or physical bus transactions. Numerical and latency claims still
need their own complete-output and qualified timing evidence.

Preparation writes `source_dense.json`, `source_pointwise.json`, `source_rows.json` or, for a combined
selection, `source_schedules.json`. The receipt identifies source and selected
IR, checked bindings, pass reports and declined mappings. Retain the normal
compilation recipe and provider-link receipt with the final image.
Combined providers retain each family's numerical and effect witness identity
in `host_provider/source_lowerings.json`; this records their custody and does
not create a new numerical qualification.

For a selected saved-model build, `output.shared_source_preparation` retains the
actual preparation receipt path and hash. `output.host_provider` and
`output.linked_elf_admission` retain the producer's records when present.
Before publishing execution success, the caller rechecks selected contracts,
prepared source/selection files and completed provider dependencies against
their original identities. These are custody checks, not a new numerical or
hardware qualification. `run="none"` records compilation only.

Unsupported source or unavailable capabilities keep the original computation.
If no region qualifies, the prepared source bytes remain unchanged and no
provider objects are linked. Wrong arithmetic, overlapping selections, stale
inputs or an invalid emitted program raise a diagnostic. These failures must
not be treated as successful optimization.

The host lifetime verifier distinguishes initial, previous, fresh and final
buffer generations in a closed serial `scf.for`. The loop must have a positive
static trip count, direct fresh allocations, false entry and true backedge
ownership flags, synchronous intrinsic or explicitly contracted borrowed uses,
guarded previous-owner releases
after their last uses, and one final owner release after observation. Separate
carried buffers remain distinct even when they share an ownership flag;
unchanged carried buffers preserve their original aliases. Numeric element
types and explicit view provenance prevent buffer handles from escaping through
other memory. The ordinary post-deallocation IR is checked before lowering
continues, and the lowering recipe records the checker source.
Nested serial loops may carry numeric scalars, including a reduction sum, with
dynamic or zero trip counts. They must have verified scalar-only bounds,
arguments, results and yields, with no loop attributes. This admits the loop
container only: captured buffer uses and every nested effect remain checked,
and previous-buffer reads must still precede the guarded release. Memory-valued
loop state, nested releases, unknown calls and pointer escapes refuse.
With selected borrowed effects, `borrowed-buffer-calls.json` retains the
original roster, `borrowed-buffer-lifetime.mlir` retains the complete actual
module at that boundary, and `borrowed-buffer-lifetime.json` records conditional
checking. The parent authenticates the native output and pins these files
through provider construction and final dependency checks. These observations
establish neither implementation effects nor hardware qualification themselves.

Zero-trip or dynamic buffer-replacement loops, ownership returned to another
scope, unknown calls, and runtime pointer-deduplication helpers remain unsupported
by this proof.
Read/write annotations do not establish noescape or completion semantics for a
provider call. Such a loop can still report a lifetime diagnostic even when its
upstream ownership lowering is legal; there is no checked automatic fallback
for it. Do not disable the check. This verifier extension moves no allocations,
does not reduce storage by itself, and does not enable arena reuse for loops.

Validate complete outputs and carried state against an independent reference
under the unchanged numerical policy. Neutral model tests exercise discovery,
actual linked bytes and execution counters; they do not establish accelerator
support or hardware latency. Compare commands, requested traffic, packing,
compile cost and code size separately from qualified paired model timing.

With the ordinary native toolchain and functional simulator configured, run the
isolated model comparisons and the allocator's capacity, alias and completion
checks together:

```sh
python -B -m pytest merlin/tests/infra/test_pointwise_storage_model.py \
  merlin/tests/infra/test_selected_storage_contracts.py \
  merlin/tests/infra/test_private_copy_elimination.py \
  -p no:cacheprovider --basetemp "$TMPDIR/source-storage-check"
```

The model comparisons preserve the same source, hardware/software declarations,
library, packing and caller inputs. They reconstruct linked instruction operands
and compare complete recurrent outputs and state. A corrupted actual address
must fail independent native execution qualification. The source pointwise
family still requires synchronous effects; asynchronous storage/completion
mechanisms have separate program-level tests and confer no source-model support.

## Remove overwritten private writes in a schedule

An original host integration can select private payload deletion in the ordinary
schedule compiler:

```python
from merlin.compile.scheduling.dead_write_pipeline import compile_schedule_with_dead_writes

result = compile_schedule_with_dead_writes(
    kernel, objects, banks, storage_adapter, encoding, program_limits,
    deletion_law=original_deletion_law,
    private_ownership=True,
    max_write_deletions=128,
    max_write_deletion_scan_steps=100000,
)
```

`original_deletion_law` is a separately qualified `WriteDeletionLaw` bound to
the selected hardware, numerical policy and storage adapter. It identifies exact
removable instruction/semantic forms and requires no faults, no additional
observations and preservation of all other effects when a write is removed.
RTL widths and memory capacities do not establish these software premises.
The caller must independently establish private ownership and complete effects,
including implicit reads and physical aliases.

For example, an initial private copy can be removed when later writes completely
overwrite its bytes before any read. Partial overwrites may jointly cover those
bytes. Unknown effects, asynchronous transfers, observations or incomplete
coverage retain the original writes. The pass preserves surviving instructions,
operands and their order. It does not remove arbitrary initialization, external
state updates or configuration operations.
Overlapping assigned temporary regions also retain their existing schedule.
Unknown instructions that the ordinary schedule compiler itself cannot admit
keep its precise diagnostic; this pass supplies no alternative executor.

The pass runs after allocation and before completion and emission checks. It
retains the checked physical assignment conservatively, invalidates old storage
and realization analyses, and reconstructs current lifetimes and completion.
A separate verifier reconstructs omissions from actual emitted, decoded calls
against the retained input from before the proposal. A selected plan or report
cannot replace that check. Final instruction counts, addresses, branches and
program limits are checked after the rewrite.

Without a law, the call uses the unchanged ordinary route. A zero deletion
budget provides a paired control. Stale source, resource, implementation, law or callback bindings,
changed survivors and invalid explicit pass ordering refuse. The original host
selects the live optimizer and separate verifier; `PostStorageRewrite` records
are conditional transport, not qualification or author permission.

This host entry is outside the frozen V6/V7 SDK inventories. Phase 2 exposure
needs an explicit reviewed release or original service selection, fixed before
authoring and identical across treatments. Phase 2 may tune admitted choices
within the original bounds; it cannot replace the deletion law or verifier.
Neutral native tests establish this mechanism with different memory and encoding
models. Actual accelerator effects, a consuming backend and complete-model
qualification remain separate integration requirements. Requested payload bytes
and command counts establish no physical bus traffic or latency improvement.
