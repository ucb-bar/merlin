---
title: Serial C dispatch
kind: guide
status: current
owner: compiler
last_verified: 2026-10-20
related: [model_memory_and_device_lowering, static_arena_wiring]
code_refs: [src/merlin/compile/application_c.py, src/merlin/compile/layout_c.py, src/merlin/compile/dispatch_c.py, src/merlin/compile/c_sources.py, src/merlin/runtime/program.py, src/merlin/xdsl_dialects/lowering/arena_plan.py]
---

# Serial C dispatch

`emit_program_c` turns an existing `MerlinProgram` into a C source/header pair.
The driver calls compiled CPU and device wrappers in graph order and uses the
program's temporary-buffer offsets. It has no allocator, operator interpreter,
startup code or platform driver. The application owns the final executable.

## Compile and bind operations

First outline/partition the source and compile its selected regions. Every node,
including copies, conversions and other support operations, needs a binding:

```python
from merlin.compile.dispatch_c import DispatchCall, emit_program_c

# program is a MerlinProgram built with explicit ArenaRequirements.
report, files = emit_program_c(program, "application", {
    0: DispatchCall("compiled_cpu_prepare", "cpu"),
    1: DispatchCall("compiled_device_dispatch", "accelerator"),
    2: DispatchCall("compiled_cpu_finish", "cpu"),
})
```

The names above are caller-supplied compiled symbols. Each implements:

```c
int compiled_region(void *context, const void *const *inputs,
                    void *const *outputs);
```

Operand/result pointers follow that node's source order, including repeated
operands. The wrapper owns conversion to its actual kernel ABI. For MLIR C
interfaces, this means constructing the required memref descriptors; raw pointer
arrays must not be passed directly to a memref function. The wrapper returns
zero only after its outputs are visible and all reads/writes have completed.
Device wrappers perform the required launch, wait and cache maintenance through
their selected runtime. No replacement CPU call is generated on device failure.

Bindings must preserve the source operation's shapes, layouts, arithmetic and
effects. Export checks their declared placement against emitted region choices;
it does not prove wrapper semantics or hardware behavior. A transition's output
storage does not identify the processor executing its conversion.

`plan_program(graph, adapter, emitter, memory_requirements=..., policy=...)`
connects placement/fusion selection to emission and buffer planning. The selected
provider supplies legal alternatives and concrete lowering. The memory callback
receives the rewritten graph, so its events and digest include fused regions and
inserted transfers. The returned search report retains costs and unknown bounds.

## Allocate caller storage

The report lists external buffers in ABI order: arguments, constants, then unique
results. `inputs` and `outputs` preserve the source signature and output order.
Bind each external buffer once using its listed size and representation.
Arguments, constants and results remain outside the temporary arena.

```c
#include "application.h"

application_state state = {0};
/* arena and buffers are application-owned, with sizes from the report. */
int status = application_run(&state, arena, arena_bytes,
                             buffers, APPLICATION_BUFFER_COUNT, context);
```

Provide at least `APPLICATION_ARENA_BYTES`, aligned to
`APPLICATION_ARENA_ALIGNMENT`. The exporter checks sizes, lifetimes, aliases,
reservations and offsets without rerunning the placement solver. The generated
driver checks buffer bounds, pointer arithmetic, arena alignment and external
overlap before any call. External aliasing is refused. The dispatcher state cannot overlap the arena
or any external buffer. Internal declared views
require an explicitly `alias_safe` wrapper; views do not become hidden copies.

On success, `state.active` is zero and `state.completed` counts all operations.
On a kernel error, outputs may be partial. `state.kernel_status` retains the
wrapper result and the state remains active. Drain or reset the endpoint before
clearing that state or reusing any buffers. Retrying with active state returns
`APPLICATION_BUSY`. Serialize access to each state and its arena; this is not a
thread-safe or asynchronous dispatcher.

## Package and link

Write the returned source/header to a fresh output directory and save `report`
as JSON. The report includes source hashes and ordered calls, but is not an
execution result. Link the driver with its compiled wrappers and kernel objects.
For a provider that supplies C/H too, `c_source_pair` can combine both pairs into
one application C/H pair using their logical header/source names. Device images
and constants remain provider-owned outputs; the driver does not relocate them.

The platform chooses the C compiler ABI, startup and linker script. The driver
uses only standard integer/pointer types and makes no OS calls. No generated
`main` or second runtime is required. Monolithic LLVM compilation retains its
existing arena binder and does not automatically use this path.

## Typed application buffers

`emit_application_c` binds the dispatch driver to an ordered tensor interface,
embeds declared constants, and stages results until all calls succeed. When
all input/output ports are f32, the interface uses `float` pointers. Otherwise
it uses opaque pointers and records each port's dtype and shape. The caller
supplies dense buffers in that order; repeated results remain separate outputs.

Layout compilation derives element widths from the declared dtype. Reshape,
copy, transpose, affine gather, slice insertion, concatenation and embedding
lookup preserve the element bytes and numerical policy. BF16 uses two bytes,
FP8 E4M3 one byte, and binary64 and signed i64 eight bytes. Boolean tensors use
one byte per element, with zero/one values; they are not bit-packed. Embedding
indices remain signed i64 and are checked before any output write. Output/input
overlap is rejected.

Non-f32 constants are emitted as integer bit patterns, preserving encoded signs,
NaN payloads and subnormals without a floating-point conversion. This provides
buffer transport, not arithmetic lowering or quantization. CPU layout placement
must be selected explicitly, and each arithmetic region still needs its own
compiled implementation and numerical qualification.

## Tests and limits

`merlin/tests/runtime/test_dispatch_c.py` compiles and runs generated C with
repeated inputs, real temporary reuse, reservations, guards and failing calls.
Placement/fusion tests check that emission precedes planning and reject changed
placements. The OOT integration test separately combines generated device C/H
with a CPU-before/device/CPU-after driver using fake hardware callbacks.
That test exercises host coordination, not device instruction execution.

This path requires static byte sizes and explicit compiled implementations.
It does not infer missing conversions, qualify CPU/device numerical behavior,
implement asynchronous overlap, or make an unsupported model compilable.
