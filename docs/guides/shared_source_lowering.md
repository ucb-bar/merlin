---
title: Enable shared passes in model builds
kind: guide
status: current
owner: core
last_verified: 2026-10-19
related: [shared_pass_phase2, ../reference/shared_compiler_passes]
code_refs:
  - src/merlin/compile/baremetal_model.py
  - src/merlin/llvmlower/source_model.py
  - src/merlin/llvmlower/source_dense.py
  - src/merlin/llvmlower/source_pointwise.py
  - src/merlin/llvmlower/source_rows.py
  - src/merlin/llvmlower/dealloc_lifetime.py
  - src/merlin/llvmlower/borrowed_buffer_calls.py
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

Dense checking covers source and schedule IR. Its pinned C emission does not
establish arbitrary target C or final machine-code semantics. The backend must
independently qualify instruction bodies and ABI effects and supply applicable
final-image checks. Static command sites are not final machine instructions.

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

### Expose checked device result identity

For the ordinary generated contraction-shim route, explicitly select:

```python
from dataclasses import replace
from merlin.runtime.backends import spike_model

routing = replace(original_device_routing, descriptor_results=True)
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

The generated checked C forwarder calls the original shim and requires its
returned allocated pointer, aligned pointer, offset, sizes and strides to match
the supplied result descriptor field by field. A mismatch traps. The original
shim's shape, stride and overlap checks still execute. Read/write
annotations or matching tensor types alone do not establish this identity.
The forwarder does not establish full writes, no retained aliases, no releases
or synchronous completion. Those remain original implementation obligations.

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
actual C/object binding. Constructing a `DescriptorResultIdentity` or loading a
saved report supplies no such proof. Borrowed-call, shared-source-provider,
profiling, vector/parallel and static-arena compositions remain unsupported at
this low-level boundary.

Inspect `lower/descriptor-result-identity.json`,
`lower/descriptor-identity.input.mlir`,
`lower/descriptor-identity.selected.mlir` and `lower/lowering_recipe.json`.
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
