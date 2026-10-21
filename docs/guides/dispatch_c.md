---
title: Serial C dispatch
kind: guide
status: current
owner: compiler
last_verified: 2026-10-20
related: [model_memory_and_device_lowering, static_arena_wiring]
code_refs: [src/merlin/semantic_compiler/constant_folding.py, src/merlin/compile/c_bucketize.py, src/merlin/compile/c_integer.py, src/merlin/compile/c_predicates.py, src/merlin/compile/application_c.py, src/merlin/compile/layout_c.py, src/merlin/compile/dispatch_c.py, src/merlin/compile/c_sources.py, src/merlin/runtime/program.py, src/merlin/xdsl_dialects/lowering/arena_plan.py]
---

# Serial C dispatch

`emit_program_c` turns an existing `MerlinProgram` into a C source/header pair.
Buffer requirements are emitted as a compact table. Bounded loops check pointer
ranges, arena/state overlap and pairwise aliasing before any compiled call.
Explicitly read-only input pairs may overlap; other overlaps remain errors.
The checks use quadratic runtime work but linear source size, avoiding a large
unrolled validation function for models with hundreds of parameters. Dispatch
calls themselves remain explicit calls to compiled implementations.
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

## Compile source predicates

`emit_predicate_c` emits checked C for f32-to-Boolean nonzero tests, signed-i64
comparisons, ordered f32/f64 comparisons, Boolean AND/XOR and widening, and typed
element selection. Floating comparisons use integer bits: either NaN makes every
ordered predicate false, and both signed zeros compare equal. Selection
preserves the chosen value's bits. Nonzero tests distinguish both zeros from
subnormals, infinities and NaNs without depending on host floating-point flags.
Unordered floating comparisons and arithmetic require other implementations.

The source dtype, numerical policy and index maps are checked before emission.
Boolean inputs must contain zero or one; invalid values fail before any output
write. Emission checks accesses against the declared tensor bounds; generated
code uses byte copies and rejects pointer overflow and overlap.

When every operand is a declared constant, evaluation within the 65,536-element
work budget emits initialized constant data and removes runtime operand ports.
The emitted record retains operand and result hashes. Runtime inputs are never
folded, and exceeding the optimization budget keeps the runtime implementation.
The placement policy must explicitly permit CPU predicate computation. A failed
device compilation does not select this route.

### Boundary counts

`emit_bucketize_c` compiles the recognized f32 lower/upper-bound count to a
checked binary search. Nondecreasing boundaries, duplicates, infinities, signed
zeros and NaN queries follow the source reduction. A NaN query returns the
boundary count. Constant boundaries are checked during compilation; all
boundaries are also validated before runtime output writes. Unsorted or NaN
boundaries return an error. Comparisons and little-endian loads/stores use
integer operations, with no dependency on host floating-point flags or libm.

This CPU operation requires explicit placement. It has no constant-folding path;
declared constants remain initialized operands. Invalid types, indexing maps,
numerical policies, overlapping buffers and pointer overflow are rejected.

## Exact integer support

`emit_integer_c` compiles one explicitly CPU-placed signed-i64 operation:
add, subtract, multiply, seeded single-axis sum, prefix sum, minimum/first-index,
or conversion to
binary32/binary64. Arithmetic wraps modulo 2^64; reductions preserve their seed
and scans reset at each lane. Boolean prefix sums require zero/one input bytes.
Invalid Boolean data is rejected before output writes.

Minimum and first-index projections reduce one axis of a nonempty signed-i64
tensor. Equal minima select the lowest index. The comparison biases the sign
bit and uses unsigned carriers, preserving `INT64_MIN` without signed overflow
or an implementation-defined cast. Both projections keep their source result
identities; separate compiled calls count as separate work.

Integer-to-float conversion implements round-to-nearest, ties-to-even with
integer bit operations, including signed endpoints and significand carry.
It does not depend on the host floating-point rounding mode. Dense buffers use
little-endian element bytes; unaligned access is supported. Code generation
checks the exact source policy and indexing maps, and runtime checks reject
overlap between outputs and inputs, and pointer overflow. These operations remain CPU work;
they do not count as accelerator execution or quantize model computation.

## Fold declared numerical setup

Staged application compilation runs `fold_source_constants` before placement
and state binding. It folds declared signed-i64 arithmetic and exact IEEE BF16
and binary64 arithmetic, plus the admitted integer/float conversions. Every
operation must match its full type, shape, identity maps and numerical policy.
Integer ratios implement rounding without consulting the host floating-point
environment. Runtime tensors and unqualified math intrinsics remain computations.

The default global budget is 65,536 element visits, counting operands and results.
Exhaustion leaves the operation in the graph. NaN arithmetic with unqualified
payload behavior is also deferred. Neither condition proves hardware limitations.
Folding preserves node IDs, sharing, runtime boundaries and ordered outputs.

The application source pass also removes unused pure computations after constant
propagation. Version 2 of its existing stage receipt records their original
positions and definitions for reconstruction and replay. Input ports, constants,
ordered outputs and state dependencies remain intact. Dead unqualified math
is not evaluated. Direct callers can retain the previous folding-only behavior.

`source-constant-folding.json` records original nodes, operand/result hashes,
work and source digests. Original nodes can be reconstructed from the transformed
graph and this receipt without copying model weights. Application verification
requires the receipt from a fresh source intake and replays it before checking
region exports. This is checked constant specialization, not a runtime evaluator,
precision change or qualification of source math intrinsics.

The exact pass also handles logical identities, reshapes and affine gathers of
declared constant tensors. Movement preserves raw bits, including NaN payloads;
gathers check every read against the source bounds. A physical view with storage
or lifetime obligations is outside this pass.

For the recognized ordered f32 prefix scan, compilation keeps an f64
accumulator and rounds each output to f32. It does not accumulate previously
rounded outputs. NaN arithmetic remains deferred. The element budget bounds
both input and output work; exhausting it leaves the operation for later
lowering. Runtime input tensors never enter constant evaluation. Versioned
receipts retain replay of the earlier primitive sets.

### Declared constant powers

Source powers remain unqualified under ordinary exact folding. An application
may separately select `source_constant_math` with policy
`positive-power-correct-rne-mpfr-v1`. This changes their numerical definition to
correctly rounded mathematical power; it does not claim bitwise agreement with
the framework's power implementation. Runtime power, sine and cosine remain
unqualified.

Build the compiler-only helper with a provisioned C compiler and MPFR/GMP
development packages:

```sh
merlin-targetgen native-math-build --out "$TMPDIR/constant-math-helper"
```

The fresh directory contains `manifest.json`, the C source, build log and a
content-addressed shared library. Copy the manifest's `policy` object into the
explicit application precision policy's `source_constant_math` field. Set
`MERLIN_CONSTANT_MATH_HELPER` to that shared library's absolute path. The selected
provider must also admit this field; older policies do not enable it.

The policy pins the helper and loaded MPFR/GMP versions and file hashes, plus
element, precision and round budgets. Only declared constant positive finite
bases and finite exponents are admitted. Directed lower and upper bounds must
round to the same f32/f64 bits. An inconclusive result remains a resource limit.
The helper performs no host floating-point arithmetic and reads no runtime
samples. MPFR/GMP are compiler dependencies, absent from generated C/H.

`source-preparation.json` records the ordered exact-folding and numerical-change
stages. Verification reconstructs the original graph and replays the stages,
dependency identities, constants and work limits. Original-model error must be
measured separately. This preparation does not establish complete model support.

## Tests and limits

Already encoded BF16 inputs can use `render_bf16_codes_c` to validate and copy
little-endian code bytes. It checks the entire finite/code-range domain before
writing output. It performs no rounding, scaling, clipping or signed-zero
canonicalization; policies requesting those transformations are rejected.
This is a representation boundary, not an arithmetic implementation.

`emit_float_arithmetic_c` implements an explicitly CPU-placed `f64_mul` with
IEEE binary64 round-to-nearest-even. Integer carriers preserve the exact
106-bit product before rounding, including subnormal results, signed zero and
overflow. The C compiler must provide an unsigned 128-bit integer type. No
floating runtime or rounding-mode change is required. Dense buffers may be
unaligned; outputs must be disjoint from inputs. Shared input operands retain
one ABI port. NaNs and zero-times-infinity remain outside the admitted domain;
the entire tensor is checked before output writes.

The same emitter supports source `bf16_add` and `bf16_sub` with IEEE
round-to-nearest-even. It adds exact signed magnitudes in units of the smallest
BF16 subnormal, then rounds once. Signed zeros, subnormals, cancellation and
overflow are retained. Buffers contain two-byte little-endian BF16 codes;
they need no floating ABI or 128-bit integer extension. NaNs and an infinity
minus the same infinity are rejected before output writes. These operations
must also be selected explicitly for CPU placement.

This is source-precision CPU computation. Its presence does not admit an Atlas
operation or authorize moving a failed device region to the CPU. Providers
must select it explicitly from the source type and numerical policy.

`emit_unary_math_c` provides explicitly selected sine/cosine calls for f32 and
f64. Its `merlin.compiled_unary_math.v1` policy selects RV64GC/lp64d, finite
inputs, an enabled FPU, RNE, and unobserved floating flags/errno. It also carries
the existing `merlin.host_linkage_contract.v1` archive hash and symbol list.
The supplier pin and numerical permission are separate requirements; neither
establishes bitwise agreement with the source framework.

Generated C refuses non-RNE execution and checks every input for finiteness
before writing output. It calls the selected symbol through a volatile function
pointer, preserving the library call. The platform enables the FPU, supplies
the matching ABI and links the exact selected archive without fast-math.
This implementation does not call a reference evaluator or compiler helper.

Applications merge actual math calls into `application-c.json`'s `host_linkage`
list. Same-archive requirements share one symbol roster; conflicting suppliers
for one symbol are rejected. The C/H package README points to that list. The
final linker still needs to establish the actual supplier and archive identity;
Merlin's existing link-supplier checker records that observation. The compiled
CPU route never substitutes for an accelerator failure.

`merlin/tests/runtime/test_dispatch_c.py` compiles and runs generated C with
repeated inputs, real temporary reuse, reservations, guards and failing calls.
Placement/fusion tests check that emission precedes planning and reject changed
placements. The OOT integration test separately combines generated device C/H
with a CPU-before/device/CPU-after driver using fake hardware callbacks.
That test exercises host coordination, not device instruction execution.

This path requires static byte sizes and explicit compiled implementations.
It does not infer missing conversions, qualify CPU/device numerical behavior,
implement asynchronous overlap, or make an unsupported model compilable.
