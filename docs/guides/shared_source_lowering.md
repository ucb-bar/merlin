---
title: Enable shared passes in model builds
kind: guide
status: current
owner: core
last_verified: 2026-10-18
related: [shared_pass_phase2, ../reference/shared_compiler_passes]
code_refs:
  - src/merlin/compile/baremetal_model.py
  - src/merlin/llvmlower/source_model.py
  - src/merlin/llvmlower/source_dense.py
  - src/merlin/llvmlower/source_pointwise.py
  - src/merlin/llvmlower/source_rows.py
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
| `SourcePointwiseLowering` | Static, contiguous, total integer elementwise operations whose scalar semantics appear in the selected software specification | Qualified fusion, temporary allocation, persistent configuration cleanup and exact device spills |
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

The host lifetime verifier remains conservative for loops that replace a carried
buffer with a fresh allocation on every iteration and conditionally free the
previous buffer through an ownership flag. Such builds can report a deallocation
placement diagnostic even when upstream ownership lowering is legal. There is
currently no checked automatic fallback for this form; do not disable the
lifetime check to admit it.

Validate complete outputs and carried state against an independent reference
under the unchanged numerical policy. Neutral model tests exercise discovery,
actual linked bytes and execution counters; they do not establish accelerator
support or hardware latency. Compare commands, requested traffic, packing,
compile cost and code size separately from qualified paired model timing.
