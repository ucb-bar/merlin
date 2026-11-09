---
title: Deploy a closed outlined kernel chain
kind: guide
status: current
owner: core
last_verified: 2026-11-09
related: [staged_dispatch, provider_workflow, worker_placement]
code_refs:
  - src/merlin/compile/outlined_program.py
  - src/merlin/llvmlower/deployable_outline.py
  - src/merlin/llvmlower/deployable_ir.py
  - src/merlin/llvmlower/kernel_backend.py
  - src/merlin/runtime/outlined_chain.py
  - merlin/runtime/c/merlin_descriptors.c
---

# Deploy a closed outlined kernel chain

`merlin.compile.outlined_program` builds a deployable C driver and isolated
arithmetic kernel objects from a closed static tensor graph. It reuses the
ordinary outliner, typed dispatch DAG, kernel extraction, compiler and memref
descriptor ABI. It does not compile the parent `forward` or execute a NumPy
driver on the target. This is a separate opt-in workflow; ordinary model builds
remain unchanged.

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

The first path accepts positive static rank-1 through rank-8 tensors with
`i8/i16/i32/i64/f32/f64` storage and exact dense shape/cast aliases or static
in-bounds slice packing. Live scalar driver arithmetic, dynamic/control-flow
drivers, unknown views, external calls, pre-bufferized/effectful kernels and
plain scalar argument/results refuse. Numerical source semantics remain those
of the captured/authored IR; this workflow neither quantizes a model nor repairs
an incorrectly captured dtype.
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
