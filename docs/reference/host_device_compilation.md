---
title: Host and device compilation
kind: reference
status: current
owner: runtime
last_verified: 2026-10-20
related: [runtime, zephyr, adding_a_target, experiment_abi]
code_refs: [src/merlin/targetgen/contract/build_recipe.py, src/merlin/targetgen/contract/build_service.py, src/merlin/targetgen/contract/compile.py, src/merlin/runtime/backends/base.py, merlin/runtime/c]
---

# Host and device compilation

Merlin has two different kinds of host work. **Host computation** is a
qualified CPU implementation of a source operation. **Host control** allocates
buffers, moves data, submits accelerator work, waits, and manages model state.
An accelerator launch may need host control even in strict-native mode; it
does not make an unsupported tensor computation a supported host route.

The compiler should produce one execution plan with separate, explicit assets:

```text
captured graph and declared constants
  → checked region partition and typed edge ABI
  ├─ host compute regions → LLVM MLIR → host objects
  └─ accelerator regions  → target OOT compiler → target program assets
  → host control and target adapter → linked host image + device assets
  → selected bare-metal, Zephyr, Linux, or simulator executor
```

The partition, numerical policy, buffer representation, output order, state,
and transfer obligations are compiler decisions. The execution environment
must not quietly move a failed required-native region to host computation.
Runtime input tensors are passed at invocation; expected outputs and grading
data belong only to the evaluator.

For layout-dependent native selection, a v4 or v5 semantic kernel request records
the physical extent and element order of every input and constant, plus the
required element order of each output. Instruction descriptions record their
operand and result layouts. A no-copy reshape, slice, or concat view is
eligible only when its sources have the declared contiguous representation;
the allocation checker then verifies the shared addresses and offsets.
Older requests without physical layouts remain readable, but cannot select
these views. Layout names must be backed by the selected target's actual
packing rules before such a view is qualified on hardware.

Input ports retain their declared order through serialization and C generation.
A v5 semantic request adds `input_order` when that order differs from the
lexically sorted storage map; older requests retain their lexical ordering.
Changing the port order changes the request identity. Use the generated
application interface for the argument and result order, including state ports.

This diagram does not establish support for every model or platform.
`MerlinProgram` carries a target-independent dispatch and memory plan. The
staged semantic application compiler uses it to emit a serial C dispatcher,
ordered tensor interfaces, and selected target program assets. A generated
C/H pair can embed those programs and declared parameters; the platform
provides launch/memory callbacks, startup, cache handling and the final link.
These application routes remain diagnostic until their source coverage,
numerical policy and selected execution platform are qualified. The existing
whole-model Zephyr route compiles RVV host work; it does not yet load and
schedule Atlas device programs.

Application buffer lifetimes follow dispatch completion events. For supported
plain buffers, bounded arena planning first tries the existing deterministic
first-fit algorithm and independently checks its placement. Other cases, or a
failed first-fit attempt, use finite-domain allocation. Solver `UNKNOWN` remains
inconclusive. Artifact verification checks the saved addresses, sizes, alignment,
aliases and lifetimes against the current graph; it does not rerun allocation.
Code generation is replayed using that checked placement. This avoids making
verification depend on another optimization search or its timeout.

## Matrix staging across launches

The shared `plan_streamed_mnk` planner keeps each output's complete K reduction
in one launch and streams independent M/N panels. If that minimum panel does
not fit, `plan_streamed_k_reduction` can select bounded M/N/K chunks for a target
that provides exact accumulator save/restore. Its first chunk may be shorter;
later chunks contain complete K tiles. Only the original first-tile padding
is retained, and an epilogue runs after the final chunk.

The default carry interface reserves separate accumulator-input and output
buffers. Reusing the output buffer requires explicit in-place qualification.
The plan records the accumulator type and numerical policy; carrying it must
not introduce a conversion or rounding step. Shared C helpers pack source bytes
and scatter valid output cells. The OOT compiler owns seed loading, device
program generation and replay checks. A plan alone does not establish that a
target implements this route.

`render_streamed_k_driver_c` composes first, continuation and final callbacks.
It packs dense encoded input chunks, alternates two disjoint completed-value
buffers, and stages the complete output until all callbacks succeed. It performs
no arithmetic on the carry. A target adapter must bind those callbacks to its
checked programs and preserve all contexts and scratch through completion.

## Device I/O memory selection

`native-build`, `native-compile`, `native-verify`, `native-select` and
`native-profile` accept `--io-memory-bytes` when the selected provider supports
it. The provider converts this byte size into its external-buffer geometry;
the target snapshot records that geometry. Generation, compilation and replay
must use the same selection. An explicit profile JSON already owns its storage
geometry and cannot be combined with this flag.

This budget is separate from `--application-memory-bytes`, which bounds the
generated application's tensor arena and output staging. It does not enlarge
register files, on-chip scratchpad or instruction memory. Providers check their
alignment and address ABI, retain the selected size in program manifests, and
reject export under a different configuration. The platform must map the
selected device-visible range and supply the required memory synchronization.

## Pointwise broadcast storage

Source preparation moves exact BF16-to-F32 widening before a complete affine
broadcast, then folds a broadcast's index map into an F32 pointwise consumer.
The small cast remains an explicit computation. Input values, arithmetic order,
rounding boundaries, escaping views and output order are preserved. Only
dimension projections that cover every source element qualify; partial slices,
general strides and casts with different numerical policies stay unchanged.
The source-preparation v3 receipt retains original nodes and generated cast IDs,
and replays indexing and numerical preconditions before publication. This is
logical fusion; physical storage sharing still requires allocation checks.

`PointwiseTiling` can reuse input tiles along an axis on which the operand's
affine map is constant. The v3 tiling record gives each input's stored tile
counts and byte size; output storage still covers the full iteration space.
The target supplies row/column address steps and may keep the entire operation
in one launch. This changes packing and addressing, not arithmetic or input
ownership. The default plan retains full materialization.

Reuse requires a lane-wise selected body. Reused values can occupy unobserved
lanes of the final output tile, so this is not valid for a reduction, cross-lane
operation or observable exception state. Output assembly copies only valid
logical elements. Source buffers remain unchanged, and allocation and replay
still check each physical input/output range independently.

A pointwise region may use one SSA input more than once. Identical index maps
use one prepared input tile; different maps require separate realizations.
The tiled body retains both operand uses and their arithmetic. Sharing a value
does not permit unrelated live values to overlap or authorize overwriting inputs.

## Explicit model state

`native-compile --application` accepts an optional `--application-state state.json`.
The declaration names the exact staged program hash and identifies state
arguments and any returned updates. For example, a function with `history` as
argument zero and updated history as result one can use:

```json
{
  "schema": "merlin.application_state.v1",
  "source_program_sha256": "<SHA-256 of compile-inputs/program.mlir>",
  "ports": [
    {
      "argument_index": 0,
      "name": "history",
      "mode": "updated",
      "output_index": 1,
      "initialization": "caller"
    }
  ]
}
```

Argument indices refer to the parsed function, including lifted parameters and
buffers. Output indices refer to the complete ordered result tuple. Names must
match the staged argument binding. A state update must have the same type and
shape as its input. A `read_only` port uses `output_index: null`. Every lifted
buffer must be declared; ordinary runtime tensor arguments can also be marked
as state. Parameters cannot become state through this interface.
The current state interface transports raw f32, f64, BF16 and FP8 codes, and
signed-i64 tensors with positive static shapes. Declaring a port does not admit
new state computations.
Compute the hash with `sha256sum compile-inputs/program.mlir`, or read
`members["program.mlir"]` in the staged `compile-inputs.json`. Use the staged
file, whose weight paths may differ from the original capture.

State remains a runtime input. It is not frozen into an instruction stream.
`initialization: "caller"` emits no initial value. For a captured buffer,
`initialization: "captured"` embeds its exact initial bytes and adds
`<application>_initialize_captured_state` to the C/H pair. The initializer
validates all destinations before writing any, and does not initialize
caller-owned initial values. Call it only while the state is idle. Changed
captured initial values change compiler input identity.

The generated header lists each state port's input index, update output index,
type, shape and byte size. Inputs and update outputs must use separate storage.
The application publishes all outputs only after successful dispatch; on error,
the caller retains the previous state and drains or resets outstanding work.
After success, it can pass the returned state to the next invocation, using
separate buffers or swapping two allocations. Merlin does not advance hidden
state or reset it on every call. Supply the same declaration to
`native-verify --application --application-state state.json`.

With `--application-parameters external`, immutable parameters bind from a
separate weight blob and no longer occupy the public invocation input array.
State declarations still use the original function's argument indices. Generated
state macros and `application-state.json` give the corresponding public input
indices; `source_input_index` retains the source ordinal. Use those generated
indices rather than subtracting a presumed parameter count. Captured state
initialization remains separate from immutable parameter binding.

This is an ABI for functional tensor state, not proof that a frontend captured
every Python mutation or effect. Updates must already be explicit results in
the source program. Arbitrary mutable Python objects and implicit buffer writes
remain unsupported. Native operation coverage, numerical quality, large state
storage and platform execution still require separate qualification. See
[`running_state`](../../examples/workloads/running_state/README.md) for captured
update and read-only-buffer examples.

The compiled CPU layout route also supports a guarded runtime-indexed slice
copy: one signed-i64 position replaces one complete slice of a fixed-size,
initialized tensor. The frontend must preserve the axis, signed bounds checks
and dynamic `tensor.insert_slice`. Lowering checks those operations structurally
before emitting the byte copy; it rejects missing or changed guards. The CPU
copy preserves all source bits, rejects an invalid position before writes and
requires disjoint input/output buffers. Admission is explicit in the selected
host-operation list. See [`indexed_state`](../../examples/workloads/indexed_state/README.md)
for a cache update followed by an accelerator reduction. This does not provide
device-resident KV storage or a complete autoregressive session.

`native-probe-model-regions` also retains `host-bridge.mlir`: the outlined
host graph with each emitted native region replaced by a typed external function
declaration. Its manifest records the exact source region, symbol, argument and
result types, and native request digest. The remaining region bodies stay in the
IR for inspection; the manifest lists them as unmet required-native work. This
diagnostic artifact has no linked device callback or execution evidence and is
never a strict-native model result.

For a native compilation that selected an entire static tensor graph, use
`merlin-targetgen native-bridge-graph` with the exact source Linalg file and
compiled artifact directory. It checks the published binary, execution plan,
selected graph, precision plan, source hash, and ordered output count, then
writes `host-bridge.mlir` and a status manifest. The bridge retains the source
function's typed arguments and result order but replaces its body with one
external device call. It is still a diagnostic declaration: the target package
must provide that call, input/output conversion, launch, wait, and readback.

```sh
merlin-targetgen native-bridge-graph --linalg /absolute/source.mlir \
  --entry forward --compiled /absolute/native-program \
  --device-symbol selected_graph --out /absolute/fresh-host-bridge
```

After MLIR bufferization, the public `forward` C interface takes source input
memref descriptors followed by preallocated result destinations. The external
selected-graph callback has a different ABI: one pointer to an aggregate of
result descriptors followed by input descriptor pointers. A target callback
must use the lowered symbol and verified ABI; matching only the number of
pointers can silently swap inputs and outputs.

## Shared build boundary

[`HarnessBuildRecipe`](../../src/merlin/targetgen/contract/build_recipe.py)
already supplies the target-independent host compile/link command shape: a
selected compiler, flags and ISA, include roots, support sources, linker
script, link flags, and static stack policy. The
[`BuildOnlyService`](../../src/merlin/targetgen/contract/build_service.py)
pins a trusted renderer and its source bytes without importing an execution
backend or reference model. The generic
[`compile_lowered_to_elf`](../../src/merlin/targetgen/contract/compile.py)
path translates LLVM MLIR, builds an object with the target recipe's ISA, then
links it to a runner-owned harness. A target adapter supplies the recipe; core
code must not select compiler flags or target instructions by target name.
The recipe resolves the effective RISC-V ABI from an explicit `-mabi` flag or by
asking the selected compiler, never from a runner default, and the paired build
records an ABI receipt so the kernel object and its harness link under one ABI.
Each compiled source gets its own object (`named_object_paths`): a unique basename
keeps its historical `<stem>.o`, colliding basenames are named by their link
position, and caller-supplied objects are never overwritten. Only output names
change; the link order and the recipe do not.

The existing model C substrate in [`merlin/runtime/c/`](../../merlin/runtime/c)
builds MLIR memref descriptors and invokes a compiled whole-model function.
Its `merlin_host_main.c` is a host verification driver. The same directory also
holds optional, target-neutral header-only numerics (ordered-FMA and product-norm
certificates, interval endpoints, radix packing, prepared softmax spans) and the
`templates/` the attention lowering instantiates; lowering-generated host code
includes them, and none of them is a device runtime or enables an optimization by
itself (see its `AGENT.md`). The tracked
`merlin_hal.h` describes a broader replay seam, but its documented
`merlin_program.c` consumer does not exist, so that header is not yet a
working universal accelerator dispatcher.

Target packages own device-specific code. A RoCC target can link its custom
instructions into the host ELF: the host core issues the instructions, and
the accelerator executes the commands. A self-hosted accelerator can instead
have a separate instruction binary that its host driver loads and starts.
Gemmini's bounded RoCC path uses the former shape; Atlas's selected diagnostic
uses the latter. The shared build recipe accepts either target's support
sources without pretending their launch and memory rules are identical.

A source-bound device catalog may expose a dense source tensor ABI whose dtype
differs from the accelerator's native operand format. Its manifest must then
declare `source_conversion` with schema `target_owned_dtype_conversion_v1`,
the same three source dtypes as `abi.dtypes`, one or more target-native dtype
triples, a nonempty canonical JSON numeric policy and its SHA-256, and each
binding's selected `target_dtypes`. Merlin checks source identity, zero
initialization, operation shape, the declared native datapath, the policy
digest, object symbols, and complete call coverage. The target package owns
the actual quantization, packing, conversion accuracy and kernel implementation.
Without this explicit declaration, a catalog can route only source dtypes
that the device itself declares. The manifest is an auditable build contract,
not a numerical certificate for the target conversion.
A backend may also declare a whole-model driver
([`whole_model_driver`](../../src/merlin/runtime/backends/base.py)): the C program
a whole model runs as on that target, with its fallback calls and timing output.
The target owns it; the shared runtime only loads the modules the backend names,
under a per-target module name, and refuses when none is declared.

## Runtime environment

The runtime ABI describes discovery, buffers, command submission, completion,
handles, metrics, and traces in [Runtime](runtime.md). Platform implementations
can be bare metal, Zephyr, Linux, or a simulator harness. Zephyr is an execution
environment and driver framework; it is not required for LLVM to compile the
host object or for a bare-metal bring-up ELF to run. A model may still need a
particular selected environment for its threading, memory, or I/O contract.

There is a working, separately scoped whole-model RVV Zephyr backend in
[`zephyr_model.py`](../../src/merlin/runtime/backends/zephyr_model.py). The
general target-module generator in
[`zephyr_module.py`](../../src/merlin/targetgen/generate/zephyr_module.py)
currently emits a non-building driver scaffold. Generating that scaffold does
not establish an Atlas or Gemmini Zephyr driver. A target's Zephyr adapter
must implement the actual register or RoCC protocol, memory visibility,
completion mechanism, errors, and metrics against a selected board.

The target-independent interface should keep the same invocation identity
across environments: program and constants identity, target revision, engine,
host and device ABI, input/output/state bindings, launch policy, and explicit
host/accelerator route accounting. An environment switch may change the host
image and driver, but cannot silently change the accelerator program,
numerical policy, or required-native coverage.

## What to qualify per target and platform

| Boundary | Shared responsibility | Target/platform evidence |
| --- | --- | --- |
| Host compile | LLVM translation and `HarnessBuildRecipe` invocation | ISA flags, linker, support sources, stack and ELF ABI |
| Device compile | Semantic region and artifact identity | Instruction selection, allocation, emission and target binary checks |
| Host/device edge | Typed buffers, state and transfer accounting | Address spaces, packing, visibility and ownership |
| Launch and completion | Explicit submit/wait result | MMIO/RoCC sequence, DMA lifetime, interrupt or polling rule |
| Execution | Separate host and accelerator routes | Actual emitted instruction execution and source-to-output checks |

For a new target, first provide a bare-metal or simulator adapter that proves
its launch and memory contract on bounded programs. Add a Zephyr binding when
the selected deployment requires Zephyr services. Both use the same compiler
assets and route accounting; neither substitutes for qualification of the
other platform. Kernel execution, full-model invocation, and physical SoC
execution remain distinct results.
