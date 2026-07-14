---
title: Runtime
kind: reference
status: current
owner: runtime
last_verified: 2026-07-14
related: [zephyr]
code_refs: [merlin/python/merlin/runtime, merlin/runtime]
---

# Runtime

Merlin **owns** the runtime abstraction: the runtime ABI, the command-buffer format, the
event/handle model, and the metrics/trace schemas. Targets provide **adapters**; they never
invent independent runtime models. This is what keeps metrics comparable across targets (and
therefore makes DSE/exploitability coherent).

Merlin's runtime spans two Merlin-owned planes:

- **`merlin/python/merlin/runtime/`** — the target-independent Python plane: the command-buffer model
  (`commandbuffer.py`, `program.py`), the event/cost **simulator** + reference oracle
  (`simulator.py`, `reference.py`) that drives DSE, the dispatch runtime (`dispatch_runtime.py`), and
  the `metrics`/`tensor` types — plus the concrete `backends/` registry (see [Backends](#backends)).
- **`merlin/runtime/`** — the dependency-free C substrate that runs a compiled whole model via its
  MLIR C-runtime interface, identically on host and bare metal:

  ```
  c/         Merlin C runtime — merlin_model.{h,c} (memref builder + merlin_run),
             merlin_host_main.c (host verify driver), merlin_bump_linux.c (bump allocator)
  abi/       mlir_runtime.c — Merlin's MLIR C-runtime symbols (memrefCopy, rsqrt) so the lowered
             forward() links identically on host and bare metal without libmlir_c_runner_utils
  baremetal/ one execution environment's harness (spike/: crt.S, htif, linker scripts, RVV asm)
  ```

These map onto the `runtime` dialect concepts: command buffers, dispatches, queues, persistent
handles, waits, profiling regions. See [core dialects](core_dialects.md).

## Runtime ABI (Merlin-owned spec)

The runtime ABI is the versioned operation set every backend/adapter must implement. It is a
**specification**, not a validated data artifact (there is no runtime_abi instance file to validate —
adapters declare support against a version string `RUNTIME_ABI_VERSION` in their
`runtime_adapter_plan`). The current operation set:

- device discovery: `device_get`
- buffers: `buffer_alloc`, `buffer_view_create`
- command buffers: `command_buffer_create`, `command_buffer_append`, `submit`, `wait`
- persistent handles: `handle_create`, `handle_destroy`
- observability: `metrics_read`, `trace_emit`

Backends register in `runtime/backends/base.py` and are enumerated in [Backends](#backends). Error
model: integer return codes, `0 == success`.

## Schemas (Merlin-owned, validated)

- `command_buffer` — the target-independent command-buffer format (its `abi_version` names a runtime
  ABI version, above).
- `metrics` — the common metric vocabulary all backends normalize into.
- `trace` — the trace-event stream format.

## Target adapters

A generated target repo's `runtime/` holds the **adapter**, not a runtime:

```
runtime/adapter/adapter.py            RuntimeAdapter: lower / encode / run_simulator / normalize
runtime/adapter/command_encoding.yaml abstract command buffer -> target encoding
runtime/adapter/metrics_mapping.yaml  raw counters -> common metrics
runtime/simulator/semantics.py        simulate(command_buffer) -> common metrics
runtime/command_buffer/example_*.json sample Merlin command buffer
```

`runtime_adapter_plan.yaml` declares which runtime features the target supports and how its
counters map onto the common metrics. The simulator (`merlin.runtime`, dependency-free) does
**real** integer tensor math: it executes the command buffer, recomputes an independent
reference, asserts they agree (residency must not change results), and emits real metrics +
trace + outputs. The generated adapter delegates execution to it.

## Backends

Concrete backends live under `merlin/python/merlin/runtime/backends/` and are addressed by **target
class**, not silicon instance, through the registry in `backends/base.py`. Two orthogonal enums
classify them:

- `TargetClass`: **CPU** (scalar/vector RVV + whole-model), **GPU** (SIMT), **NPU** (systolic/tensor).
- `BackendKind`: **KERNEL** (compile + run one command buffer), **WHOLE_MODEL** (run a whole captured
  model), **MATMUL_ROUTE** (route matmuls to an external/hand GEMM for attribution).

`base.py` is the single source of truth: the `_REGISTRY` (name → `BackendInfo`), the `Backend`
protocol every module implements (`available()`, `compile_command_buffer()`, `run_elf()`,
`parse_output()`, `run_command_buffer()`), the shared `OUT`/`METRIC`/`DONE` console parser, and the
lookup helpers (`list_backends()`, `get_backend()`, `backends_of_class()`). The **invariant** that
keeps metrics comparable across targets: every backend gates its run against `reference_outputs(cb)` —
the *same* oracle the Python simulator is held to — and normalizes counters into the common `metrics`
schema. Residency / parallelization must never change results. The registry's job is the
instance→class generalization (mirroring `xdsl_dialects.targets.factory`) so tooling reasons about
"the CPU/RVV backend" or "the NPU/systolic backend", not a silicon name.

| Backend | Class | Kind | What it is |
|---|---|---|---|
| `spike` | CPU | KERNEL | cmd buffer → `rvv_codegen` C driver → chipyard gcc → `spike --isa=rv64gcv_zfh_zvfh -pN` (multicore RVV CPU) |
| `saturn_vec` | CPU | KERNEL | non-matmul RVV vector family on spike rv64gcv (+ optional Saturn-OPU RTL); reuses the spike harness |
| `gemmini` | NPU | KERNEL | cmd buffer → `gemmini_codegen` → oracle: `spike --extension=gemmini` (bootstrap) or Verilator RTL (cert) |
| `muon` | GPU | KERNEL | SIMT C++ → `clang-muon` (Vortex/RV32) → oracle: `cyclotron` perf model or RadianceMuon VCS RTL (cert) |
| `spike_model` | CPU | WHOLE_MODEL | whole captured model end-to-end on spike (model.mlir → rv64gcv ELF + Merlin C runtime + weights blob) |
| `zephyr_model` | CPU | WHOLE_MODEL | whole model on Zephyr SMP (spike today, FireSim on a 2-tile board); RVV worker pinned to the vector tile |
| `xnnpack_board` | CPU | MATMUL_ROUTE | board (RVV): route f32 `linalg.matmul` dispatches through XNNPACK |
| `openblas_board` | CPU | MATMUL_ROUTE | board (RVV): route through OpenBLAS GEMM |
| `ours_board` | CPU | MATMUL_ROUTE | board (RVV): route through the "OURS" hand GEMM |
| `xnnpack_host` | CPU | MATMUL_ROUTE | host (x86): XNNPACK reference — the third e2e column beside `hand_v0` and ours |

Each backend's external toolchain / simulator resolves via a `MERLIN_*` env var and is optional: its
`available()` returns False (or it fails at use with an actionable message) when the toolchain is
absent, so the Python simulator + whole-model paths that need no external tool still work. `spike`
uses `MERLIN_CHIPYARD`; `zephyr_model` also drives FireSim (see [zephyr](../guides/zephyr.md)).
(There is no longer a standalone `vcs` backend — the prebuilt-VCS-RTL-as-certification-oracle role now
lives inside `saturn_vec` and `muon`.)

### Design references (CompGen / ModelBlaster)

Studied as runtime-design inspiration (`/path/to/CompGen`,
`/path/to/ModelBlaster`); lessons adopted or queued:

- **Adopted now**: declarative backend/toolchain resolution (env-var roots, one place —
  ModelBlaster's `pipeline/backends.py` registry pattern); static per-model buffers and
  embedded golden data instead of malloc on bare metal; one-line verify summaries over
  full dumps (FireSim-safe); `rdcycle`/`mcycle` bracketing per region.
- **Queued for the Zephyr/C runtime**: CompGen's HAL device vtable
  (`runtime/include/compgen/hal.h`) as the shape of the Merlin C ABI's backend layer;
  CompGen's O(1)-reset bump arena (`arena.h`) for activation memory; compile-time-gated
  trace ring buffer (`trace.h`) for `trace.schema.yaml` events; ModelBlaster's
  `modelblaster_pool` k_sem rendezvous (~20k-cycle parallel-for) for Zephyr SMP.
- **Avoid**: single-successor task DAGs (fan-in needs wrapper NOPs — CompGen `task.h`);
  schedule tables keyed by table position instead of dispatch id; per-backend weight
  layouts baked into kernels (keep the canonical layout in the command buffer; pack at
  dispatch).
