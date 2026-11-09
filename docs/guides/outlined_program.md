---
title: Deploy a closed outlined kernel chain
kind: guide
status: current
owner: core
last_verified: 2026-11-09
related: [staged_dispatch, provider_workflow, worker_placement]
code_refs:
  - src/merlin/compile/outlined_program.py
  - src/merlin/compile/frozen_inputs.py
  - src/merlin/llvmlower/deployable_outline.py
  - src/merlin/llvmlower/deployable_packing.py
  - src/merlin/llvmlower/deployable_ir.py
  - src/merlin/llvmlower/kernel_backend.py
  - src/merlin/runtime/outlined_chain.py
  - src/merlin/runtime/prepared_staged_kernel.py
  - merlin/runtime/c/merlin_staged_kernel.h
  - merlin/runtime/c/merlin_descriptors.c
---

# Deploy a closed outlined kernel chain

`merlin.compile.outlined_program` builds a deployable C driver and isolated
arithmetic kernel objects from a closed static tensor graph. It reuses the
ordinary outliner, typed dispatch DAG, kernel extraction, compiler and memref
descriptor ABI. It does not compile the parent `forward` or execute a NumPy
driver on the target. This is a separate opt-in workflow; ordinary model builds
remain unchanged.

## Explicit immutable inputs

The Python build API accepts `frozen_inputs=FrozenInputs(...)`. The default
`None` retains the dynamic argument ABI. Parameter names, capture metadata and
unchanged values observed in earlier calls never grant immutability.

First derive the ordinary plan with `plan_deployable_chain` using the same
source, worker assignments and data-preparation options as the build. Supply
its `source_ir_sha256`, plus each selected logical source argument's exact
ordinal, signless integer dtype, dense shape, absolute raw-file path and SHA256:

```python
from merlin.compile.frozen_inputs import FrozenInput, FrozenInputs
from merlin.common.digest import sha256_file

immutable = FrozenInputs(
    source_ir_sha256=plan.source_ir_sha256,
    inputs=(FrozenInput(0, "i8", (4, 8), weight_file, sha256_file(weight_file)),),
    offline_budget_bytes=1 << 20,
    published_budget_bytes=1 << 20,
)
# Forward all the ordinary explicit build arguments, adding:
# build_outlined_program(..., frozen_inputs=immutable)
```

This is an explicit Python capability; the outlined JSON CLI does not construct
it implicitly. Frozen arguments disappear from the generated caller input
roster. `external_input_source_indices` maps the retained caller slots back to
the original source arguments. A different frozen value requires a new build.
Original source, canonical kernels, DAG and dispatch ordinals remain recorded;
`dispatch_mapping` identifies the remaining runtime ordinals. Explicit prepared
selections use the original ordinals and cannot be silently folded.

Only closed, admitted wrapping integer kernels are evaluated offline. The
selected native host compiler's complete output bytes, input preservation and
guards are compared with an independent typed integer interpreter. Exact
projected/permuted indexing and initialized wrapping-sum reductions are required.
Floating arithmetic, unknown contracts and unsupported bodies grant no folding
permission; they remain worker work or cause explicit refusal. Runtime activation
quantization, correction and scaling are not algebraically simplified.

Admitted byte views can be packed offline. Only frozen buffers still consumed
by runtime work or returned as results are published, through the existing
binary-blob assembly transport and offset descriptors. Tensor values are never
rendered as C arrays. Compiler-created, temporary host-image names are not
dependencies: each evaluation records its actual private loaded-image digest,
a byte-identical retained shared image, typed source/entry, compiler artifacts
and tools in an explicit manifest.

The offline budget conservatively accounts for live inputs, tensor and Cartesian
scalar-domain intermediates, independent reference work and guarded native
evaluation storage. It does not bound compiler-process memory. The published
blob has its own limit; live blob plus provider/tensor arena also fit the build's
workspace budget. Caller input/output storage, generated context, stacks, code
and platform memory require separate complete linked-image admission. Build
time and live storage are recorded; neither establishes amortization, speedup,
target numerical qualification or full-model readiness.

For a graph whose original unique-storage plan exceeds the target capacity,
`analysis_workspace_budget_bytes` can declare a separate, explicit LP64 bound
for pre-specialization **metadata-only** planning. It is legal only together with
`FrozenInputs`. It does not allocate that source arena or raise the final target
budget: specialization must still fit its live readonly blob plus final
tensor/provider arena inside the original `workspace_budget_bytes`. The receipt
records both limits and the original planned byte count. This distinction does
not by itself admit a full model's caller data, code, platform memory or lifetime.

Byte permutation uses the source proof's complete permutation and dense strides,
then a transpose view and at most one contiguous payload copy. Thin dimensions
create no integer index journal. Publication copies also count against offline
payload accounting. Allocator overhead and compiler-process resident memory are
not bounded by this logical accounting. Retained host images are rechecked at
manifest and builder closure. Rank-zero offline tensors remain explicitly
unqualified even when the ordinary tensor runtime supports their ABI.

Every arithmetic dispatch runs on one explicitly assigned compute worker.
The controller initializes bindings, packs admitted views, prepares descriptors,
dispatches work and copies final results. It never calls a compiled arithmetic
entry. Independent kernels are currently serialized in source dependency order;
this is not automatic within-kernel parallelism or a throughput claim.

## Compile the graph

Supply a `merlin.outlined_program.v1` configuration. For example:

```json
{
  "schema": "merlin.outlined_program.v1",
  "roles": {
    "cpu_hart_ids": [0, 1, 2],
    "model_init_hart": 2,
    "pack_hart": 2,
    "dispatch_hart": 2,
    "finish_hart": 2,
    "compute_harts": [0, 1],
    "profile_harts": [0, 1]
  },
  "worker_slots": [0, 1, 0],
  "workspace_budget_bytes": 65536,
  "alignment_bytes": 64,
  "compute_cflags": [
    "-march=rv64gcv_zvl256b", "-mabi=lp64d",
    "-fno-vectorize", "-fno-slp-vectorize"
  ],
  "vectorize": false,
  "codegen_target": "rvv"
}
```

These hart IDs, VLEN, workspace and alignment are caller choices. Select them
from the actual board/provider and source graph. Worker slots must cover exactly
the graph's arithmetic dispatches. The example assigns three consecutive kernels
to worker slots zero, one, zero; it does not partition their reductions or alter
quantization. VLEN, SEW, ELEN, LMUL and DLEN remain distinct. Disabling explicit
vector scheduling and auto-vectorization makes a conservative functional build;
it grants no vector utilization or peak-throughput claim.

```bash
python -m merlin.compile.outlined_program --source model.mlir \
  --config outlined.json --output fresh-build
```

The corresponding Python API is
`build_outlined_program(source, output, roles=..., worker_slots=...,
workspace_budget_bytes=..., alignment_bytes=..., compute_cflags=...)`.
The directory must be fresh. A failed compile retains its source and failure
receipt rather than replacing an existing result.

`build.json` identifies each source dispatch, its original provenance, selected
worker slot, emitted entry and object bytes. The canonical code key normalizes
only function and SSA spelling. It retains every constant, type, map, numerical,
effect, provenance and unknown attribute, plus lowering identity. Reuse is only
within one build and never follows from equal signatures or shapes. The source
dispatch-to-function mapping remains explicit even when entries are reused.

This workflow uses checked generic operation assembly for its code keys, emitted
kernel source and compiler input. It reparses that text independently and refuses
if operation attributes, properties, types or complete region bodies differ.
Named/custom assembly can omit extension metadata or reconstruct a named
operation's standard body; it is not a lossless code-identity format. The shared
printer default and ordinary monolithic compiler path are unchanged.

## Explicit prepared staged kernels

The Python builder optionally accepts
`prepared_staged_kernels={dispatch_index: PreparedStagedKernel(...)}`. This is
an explicit provider selection, separate from the ordinary compiled entry. The
provider binds the complete canonical generic source hash, exact header/object
bytes, callback symbols, state type/size/alignment and active worker slots.
A mismatched or unavailable selection refuses; it never silently falls back to
an integrated-packing implementation. This option has no CLI flag yet and does
not enable an offload policy or whole-model placement.

The provider owns source equivalence, target context and instruction requirements,
preparation and synchronous compute/finish semantics. Its `prepare` and `finish`
callbacks execute on the declared controller. Only selected pinned workers invoke
`compute`. Ordinary unselected entries retain their compiled memref ABI. The
common invocation reuses checked argument tables, capacities and descriptors,
plus the immutable sequence and worker roster. Its backing arrays remain live
through finish and the external worker join.

All declared provider metadata and scratch are reserved in the same whole-chain
workspace budget as tensor intermediates, with no lifetime reuse. Generated C
checks actual state size/alignment and callback signatures. Providers must put
every mutable scratch object in that state; hidden buffers, implicit teams and
asynchronous work escaping this contract are unsupported. Platform stacks,
executable, context and library memory need separate admission. A byte-bound
object and successful source build alone establish neither numerical equivalence
nor device completion or hardware correctness.

Providers constructing positive pitched leaf views can reuse
`merlin_prepare_strided_descriptor`: it transactionally checks extents, strides,
offset, element alignment, accessed span, allocation capacity/address range and
metadata separation before using the same rank-exact descriptor packing. Bounds
do not establish injectivity, writable access or noalias/restrict permissions;
those remain explicit source/provider ownership obligations.

## Link a platform driver

The build emits `merlin_chain.c/.h`, individual kernel objects and an explicit
runtime source/include roster. Compile the generated controller code and
`merlin_staged.c`/`merlin_descriptors.c` for the controller's ordinary ISA. Link
the selected kernel objects with their actual compute ISA. The exact compiler
pointer/index layout and each lowered ranked-memref ABI are checked. The first
kernel subset refuses heap calls, LLVM allocas and OpenMP instead of borrowing
an unbounded allocator or an implicit nested team.

The platform creates persistent pinned workers once, initializes their required
architectural context, and supplies the staged runtime's hart/clock/counter
hooks. It calls the generated `merlin_chain_run` on the controller with:

- A zero-initialized context containing dense input/output pointers, declared
  byte capacities and a private aligned workspace.
- The compiled role policy and a nonzero first sequence.
- A launch hook that wakes every already-ready worker to call
  `merlin_staged_worker(team, slot)`.
- A join hook that confirms every worker has returned and stopped accessing
  the team object before its next initialization.

The driver creates a fresh staged generation for each dispatch. It requires
quiescence and an external completion join before reusing team storage. Sequences
increase across kernels and repeated runs. Stale sequences, concurrent reentry,
role mismatch and incomplete joins refuse. Errors poison the context; an
unquiesced error retains the context and every backing buffer until the external
owner resolves the outstanding worker. There is no callback cancellation.

Input tensors are immutable and outputs/workspace must be disjoint. Every
intermediate has unique storage retained for the entire chain. Reshape aliases
keep their source allocation alive; admitted static strided slices are packed
as bytes on the controller. No speculative last-use reuse, hidden input layout
conversion or dtype conversion is performed. Kernel parameters used as writable
accumulators, including through tensor aliases, refuse this initial immutable
input ABI.

`merlin_prepare_descriptors` is the shared no-invoke ABI helper. It uses the same
rank-exact packing as the monolithic runtime and validates all positive shapes,
byte extents and capacities before writing any descriptor. The target must
independently establish actual allocation ownership and coherent synchronous
completion; a pointer/capacity declaration alone cannot prove either.

## Check the target and retain profiles

The caller owns startup, ISR/vector context, physical hart mapping, source/kernel
link closure, console, bounded execution and an independent complete-output
checker. Verify all source inputs and exterior guards as well as every output.
A successful guest exit alone grants no numerical or hardware qualification.

The context retains one quiescent `merlin_staged_report` per completed dispatch.
Its kernel-call counters must increase exactly once on the assigned worker;
the controller reads no profiling counter. Optional ABI-2 clock windows use
the [staged window validator](staged_dispatch.md) and remain distinct from local
cycle/retirement counters. The interval from the first leader ARM timestamp
through the final leader acknowledgement can include inter-kernel controller
dispatch gaps only when the common clock domain is independently established.
No frequency, physical busy time or calibrated cycle accuracy is inferred.

## Initial admission limits

The path accepts ordinary dense static rank-0 through rank-8 tensors with
`i1/i8/i16/i32/i64/f32/f64` storage and exact dense shape/cast aliases or static
in-bounds slice packing. Positive extents remain required for every existing
dimension. A rank-zero tensor contains exactly one element: it is neither an
empty tensor nor a C scalar argument. Its LP64 ranked descriptor contains only
allocated pointer, aligned pointer and offset, without size/stride fields. Live scalar driver arithmetic, dynamic/control-flow
drivers, unknown views, external calls, pre-bufferized/effectful kernels and
plain scalar argument/results refuse. Numerical source semantics remain those
of the captured/authored IR; this workflow neither quantizes a model nor repairs
an incorrectly captured dtype.

Boolean `i1` values use one logical bit in a one-byte container per element.
The explicit ABI requires canonical bytes zero or one: every initial boolean
input byte is checked before dispatch, and the assigned compute worker checks
every produced boolean byte before publication. Noncanonical values refuse;
upper padding bits are never assumed zero and no implicit boolean conversion
changes input data. These checks add real validation work. Boolean transposes
stay compiled worker operations rather than unqualified controller byte
permutations; selected prepared-provider boolean kernels require a separate
canonical-byte proof and currently refuse through this interface.

Unknown parent-function or argument/result contracts also refuse: a changed
kernel arity cannot silently inherit or discard those contracts. Module and
kernel-operation attributes remain in the emitted source and cache identity.

Full models still require source-correct captures, complete supported view and
arithmetic coverage, quantization admission, target/link memory bounds and
complete numerical gates. The workspace bound covers declared chain buffers;
the generated context, platform stacks, executable and runtime-library closure
need separate memory admission. This path establishes a deployable closed-chain
mechanism, not full-model readiness, a hardware performance result or a matrix
offload policy.

## Optional constant and layout preparation

`build_outlined_program(..., data_preparation=True)` and the same optional
boolean in the JSON config enable a source-proved preparation seam. The default
is false. Immutable static constant `tensor.splat` producers become equivalent
`linalg.fill` producers using the exact original literal attribute. The existing
outliner clones these producers into their arithmetic kernels, so reduction
seeds and numeric constants require no scalar controller ABI. A closed fresh
literal fill whose tensor remains live outside those arithmetic roots is outlined
as its own zero-input tensor kernel on an assigned compute worker. The original
producer is still cloned into ordinary reduction roots, preserving their immutable
input contract. Only an exact literal/empty/identity-yield body is eligible.
This changes no
quantization, floating arithmetic order, overflow policy or literal bits.

A closed rank-zero `tensor.extract` immediately consumed by one length-one
`tensor.from_elements` may also stay together in its original tensor-input and
tensor-output compute kernel. This selection proves exact type, encoding,
complete scalar uses and equal source attributes. The original operations and
all their metadata remain intact. It neither evaluates a scalar on the
controller nor grants a source/destination storage alias. Other scalar
consumers, arithmetic, indices and unknown contracts retain normal refusal.

A standalone kernel containing only a fresh-output static `linalg.transpose`
with its exact identity-yield body may become controller byte packing. The
permutation, dimensions, dtype, full destination coverage and source/destination
storage are proved from the actual function. Nondefault encodings, unknown
contracts, dynamic shapes or nonidentity bodies grant no packing permission.
The outlined transport also supports a static dense `tensor.concat`: its axis,
all operand/result types, complete contiguous byte spans and fresh destination
are proved from source. Read-only operands may duplicate or partially overlap
each other; the destination stays disjoint. Unknown attributes, nondefault
encodings, empty or dynamic extents, rank-zero concat and unproved boolean
storage refuse. This is byte transport, without implicit numeric conversion or
in-place cache-update permission.

A same-rank static `tensor.insert_slice` may likewise become fresh-output byte
transport. It copies the complete immutable initial tensor before overwriting
only the proven source spans. Static offsets, sizes, positive strides, dtype,
encoding, complete bounds and distinct output ownership are mandatory. Inputs
may overlap each other; neither is modified. A contiguous innermost source span
may use one byte copy, while other fixed strides retain ordered element copies.
Dynamic coordinates, rank reduction, rank-zero insertion and unproved boolean
storage require separate proofs. This does not grant in-place KV-cache reuse.
Reported read/write byte counts are logical transport work, not measured memory
traffic or physical bandwidth.

All views and intermediate buffers retain unique whole-chain storage; packing
never mutates inputs or shortens an alias lifetime. Only byte/address operations
run on the controller. All reductions, quantization, integer compensation,
scaling and other numerical kernels stay on their explicitly assigned workers.

Worker assignments cover the arithmetic dispatches remaining after this explicit
preparation; source packing entries remain in the recorded DAG with their
provenance. Plan/build records bind this choice and the preparation owner source.
This seam supplies neither automatic offload policy nor source-model numerical,
allocator, target or performance qualification. Full independent target output
checks and complete memory/runtime admission remain required.

For a proved dense rank-2 permutation, the separate options
`packing_tile_shape=(rows, columns)` and `packing_tile_budget_bytes=budget`
select blocked byte transport. Both are required together with
`data_preparation=True`; JSON uses a two-element list. The budget bounds twice
the effective tile's element count times the storage width, for its logical
source and destination footprints. It allocates no scratch and declares no cache
capacity, cache-line traffic or hardware performance benefit. Partial tiles copy
every element exactly once. Higher-rank proved permutations explicitly retain
the flat path. The default `None` preserves the original loop and plan metadata.
The selected shape, budget, logical traffic and fallback reason enter the plan
and build identity. Input/output alias rejection and whole-chain lifetimes remain
unchanged; tile selection grants no in-place permission or numerical conversion.

Ordinary compiled kernel objects are checked against their recorded compilation
hashes and included in the build's `input_sha256` closure, alongside prepared
provider objects. A changed object refuses the build. The caller still binds and
checks the complete linker, runtime and resulting target ELF independently.
