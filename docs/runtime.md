# Runtime

Merlin **owns** the runtime abstraction: the runtime ABI, the command-buffer format, the
event/handle model, and the metrics/trace schemas. Targets provide **adapters**; they never
invent independent runtime models. This is what keeps metrics comparable across targets (and
therefore makes DSE/exploitability coherent).

`merlin/runtime/` is the target-independent substrate:

```
common/          shared runtime types
command_buffer/  enqueue / submit / wait
simulator/       event/cost simulator (drives DSE)
baremetal/       bare-metal backend (scaffold)
zephyr/          Zephyr RTOS backend (scaffold)
```

These map onto the `runtime` dialect concepts: command buffers, dispatches,
queues, persistent handles, waits, profiling regions. See `docs/core_dialects.md`.

## Schemas (Merlin-owned)

- `runtime_abi` — the versioned operation set every backend/adapter supports.
- `command_buffer` — the target-independent command-buffer format.
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

`simulator` (fast validation / cost modeling / DSE), `host`, `baremetal`, `zephyr` (see
`docs/zephyr.md`), `firesim`/`external`. Every backend normalizes its counters back into the
common `metrics` schema.

### Real-ISA backends (implemented)

- **spike (baremetal RVV)** — `merlin/python/merlin/runtime/backends/spike.py`: the
  command buffer is compiled (`rvv_codegen.py`) into a bare-metal driver around the
  hand-written RVV kernel and the Merlin harness (`merlin/runtime/baremetal/spike/`),
  then run with `spike --isa=rv64gcv_zfh_zvfh -pN` as a multicore RVV CPU. Correctness
  gate: parsed outputs must equal `reference_outputs(cb)` — same oracle as the Python
  simulator. `cycles` is the hart-0 mcycle delta. Toolchain via `MERLIN_CHIPYARD`.
- **vcs (gated)** — `backends/vcs.py` replays the *same ELF* on a pre-built Saturn VCS
  simulator (`MERLIN_SATURN_SIMV`); it never builds RTL.

### Design references (CompGen / ModelBlaster)

Studied as runtime-design inspiration (`/scratch2/agustin/CompGen`,
`/scratch2/agustin/ModelBlaster`); lessons adopted or queued:

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
