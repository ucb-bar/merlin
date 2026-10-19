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
  - src/merlin/llvmlower/logical_borrowed_abi.py
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
With selected borrowed effects, `borrowed-buffer-calls.json` retains the
original roster, `borrowed-buffer-lifetime.mlir` retains the complete actual
module at that boundary, and `borrowed-buffer-lifetime.json` records conditional
checking. The parent authenticates the native output and pins these files
through provider construction and final dependency checks. These observations
establish neither implementation effects nor hardware qualification themselves.

Zero-trip or dynamic loops, ownership returned to another scope, unknown calls,
and runtime pointer-deduplication helpers remain unsupported by this proof.
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
