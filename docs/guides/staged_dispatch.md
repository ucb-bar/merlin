---
title: Explicit preparation and compute dispatch
kind: guide
status: current
owner: core
last_verified: 2026-11-08
related: [worker_placement, provider_workflow, profile_evidence]
code_refs:
  - src/merlin/runtime/staged_dispatch.py
  - src/merlin/runtime/staged_windows.py
  - merlin/runtime/c/merlin_staged.h
  - merlin/runtime/c/merlin_staged.c
---

# Explicit preparation and compute dispatch

The staged runtime executes one explicitly authored preparation/compute/finish
unit. It is an opt-in C API and Zephyr harness renderer. Ordinary model builds
and their integrated packing remain unchanged. A monolithic model is not
automatically segmented by selecting this API.

The caller supplies an explicit CPU-to-physical-hart map and roles. For example,
this caller-owned three-hart policy assigns preparation to one controller and
all model arithmetic and profiling to its other two harts:

```python
from merlin.runtime.staged_dispatch import StagedRoles

roles = StagedRoles(
    cpu_hart_ids=(0, 1, 2),
    model_init_hart=2, pack_hart=2, dispatch_hart=2, finish_hart=2,
    compute_harts=(0, 1), profile_harts=(0, 1),
)
roles.validate_compute_capabilities(selected_provider_compute_harts)
```

No hart numbers are implied by the runtime. A capability roster and the actual
OS enumeration must agree with the selected target. `StagedExecutionContract`
requires explicit descriptions of shared-memory coherence, synchronous compute
completion, pinned worker context, preserved interrupt context and single-owner
storage lifetime. These are caller declarations, not independent hardware proof.
In particular, scalar atomics do not prove completion of an asynchronous vector
or device command. The compute callback must establish that completion before
returning, using its target-owned implementation and ordering contract.

## C lifecycle

Compile `merlin_staged.c` and include `merlin_staged.h`. The controller calls
`merlin_staged_team_init`, then starts every pinned compute worker. Each worker
calls `merlin_staged_worker(team, slot)` once. The controller calls
`merlin_staged_controller(team)` once. A team refuses a second dispatch.

The runtime performs this sequence:

1. Publish ARM and wait for every compute worker's readiness acknowledgement.
   Worker zero reads the shared global clock before acknowledging readiness.
2. Call `model_init` and `prepare` on the controller. These callbacks may prepare
   inputs and descriptors; all model arithmetic belongs in compute callbacks.
3. Publish the prepared context with release/acquire ordering. Each worker calls
   `compute(context, slot)` on its declared hart, recording local cycle and
   retirement endpoints around that synchronous callback.
4. Wait for every worker completion, then call `finish` on the controller.
   Finish may copy completed outputs; it must not run deferred model arithmetic.
5. Publish FINAL_ACK. Worker zero reads the global end timestamp. Every worker
   acknowledges that it no longer accesses the stage's descriptor or storage.

The caller owns all buffers and callbacks. A single nonreentrant stage may be
in flight. Initialization must precede worker entry; descriptors, immutable
inputs and private outputs must remain valid until `report.quiescent` is true.
No allocator, OpenMP dispatch, ISA instruction, timer ISR or background thread
is installed by the C runtime.

Callback errors are retained. Polling is finite and records timeout or missing
quiescence rather than silently dropping workers. Poll budgets are liveness
bounds, not elapsed-time measurements. ABORT does not cancel a running callback.
If quiescence is false, preserve storage and stop further dispatch; an external
owner must resolve the outstanding thread/process before reuse. A process
watchdog remains a separate caller responsibility.

## Zephyr entry and compilation

`render_zephyr_staged(roles, contract, bindings, compute_capable_harts=...,
sequence=..., max_wait_polls=...)` returns source plus explicit metadata.
`StagedBindings` names caller-owned context, callbacks, hart/clock/counter hooks
and a completion-report callback. The renderer pins all worker threads before
starting any, pins the controller separately, and uses the portable C lifecycle.
It does not infer a board, source graph, device extension or console protocol.

Compile controller/preparation/finish support using its actual ordinary ISA;
keep specialized compute implementations in separate translation units with
their declared ISA. Audit the emitted code and link closure. The renderer does
not establish that a compiler refrained from emitting a forbidden instruction.
The completion callback owns full numerical output checking, diagnostics and
terminal behavior; it receives raw worker observations on the controller but
must not create controller profiling tracks or counters.

## Counter meaning and remaining integration

The global interval begins on a compute hart before controller model
initialization/preparation and ends after output finish/acknowledgement. It
includes preparation, dispatch, compute and waiting in that scope. Select a
shared clock that continues during waiting; local `mcycle` may stop during WFI.
Only compute harts invoke profiling readers. Local cycle/retirement endpoints
and global ticks have separate domains and are not subtracted to invent stalls.
Sequential counter reads have bracket overhead and temporal skew.

No physical unit busy time, calibrated timing accuracy, overlapping stages,
double buffering, kernel reentrancy or full-model segmentation follows from this
API. A full-model role split needs a separately verified compiler/runtime plan
that dispatches every nonmatrix and matrix arithmetic task to compute harts,
retaining source quantization and dependency order. Calling a complete parent
model on the controller does not satisfy that contract.

## Optional per-worker shared-clock windows

The successor C ABI is version 2 (`MERLIN_STAGED_ABI_VERSION`). It adds a
default-off `enable_worker_windows` configuration field and a checked
`shared_window_ticks(context, uint64_t *out)` hook. Source/header layouts change;
rebuild every consumer against the selected header and implementation together.
Do not replace the header underneath a running or byte-bound image.

When enabled, each compute worker reads its own ARM-ready, compute-begin,
compute-end and final-ACK timestamps. The checked hook returns zero on success;
any read error preserves its exact status and clears that sample's validity.
The complete mask is 15. Status distinguishes valid, read-error, nonmonotonic
and incomplete records. Zero is a valid timestamp when its bit is present.
Clock errors do not cancel numerical execution, but they cannot pass the
separate profiling gate. The leader's compatibility interval is derived from
its own ARM/final samples. The legacy clock hook remains the default-off path.

Select these windows explicitly in `render_zephyr_staged` with
`worker_clock_windows=True`, a checked hook in `StagedBindings`, and an explicit
`shared_clock_domain`. All compute workers must actually observe one declared
monotonic epoch. The renderer records that caller declaration; it does not prove
clock coherence or assign a frequency.

`merlin.runtime.staged_windows.analyze_worker_windows` consumes a quiescent ABI-2
record and an independently selected role/sequence policy. Records contain only
the active compute-worker roster. The adapter verifies masks, raw endpoints,
statuses, leader compatibility and local counter monotonicity. Contradictory
or malformed records refuse; declared read errors, regressions and incomplete
records yield failed profiles with null durations. A nonquiescent failure may
publish only stable error metadata and an empty worker roster; reading unfinished
worker observations would itself violate the lifetime protocol.

The report separates pre-compute wait, compute callback window and post-compute
wait in shared raw ticks. These include publication, scheduling, preparation,
other-worker completion, finish and instrumentation costs in their respective
windows. They are not pure preparation cost, causal stalls, controller busy time
or physical compute-unit utilization. Local cycle/retirement counters retain
their independent semantics and are not mixed with shared-clock ticks.

The adapter writes fresh JSON, worker CSV and aligned Chrome/Perfetto tracks via
`write_worker_windows`. Tracks use one shared raw-tick origin across workers.
Without qualified Hz, one tick is one display-microsecond coordinate, explicitly
marked as a coordinate rather than elapsed microseconds; real seconds, bandwidth
and physical busy remain null. No controller profiling track is emitted.

```bash
python -m merlin.runtime.staged_windows --input observed.json --roles roles.json \
  --sequence 1 --clock-domain declared-shared-clock \
  --counter-semantics declared-worker-counter-domain --output fresh-profile
```

The input is a caller-exported C report with `abi_version: 2`; the role JSON
contains exactly the `StagedRoles` fields using JSON hart lists. Both input byte
digests are retained. A timing report establishes neither numerical equivalence
nor actual source-clock/target qualification; keep the complete output gate and
image association separate.
