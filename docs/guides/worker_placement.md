---
title: Placing compute workers and a scalar controller
kind: guide
status: current
owner: runtime
last_verified: 2026-11-08
related: [zephyr, simulator_selection, reproducibility]
code_refs: [src/merlin/runtime/worker_placement.py, src/merlin/runtime/backends/zephyr_placement.py, src/merlin/runtime/backends/zephyr_model.py, src/merlin/runtime/boards.py]
---

# Placing compute workers and a scalar controller

Physical hart count, online OS CPU count and compute-worker count are different
quantities. Select them independently when one hart controls work on a subset
of the machine. Matrix workers must belong to the board's declared matrix-hart
set; vector capability alone does not imply matrix capability.

The opt-in whole-model Zephyr API accepts `worker_harts`, `controller_hart` and
`cpu_hart_ids`. For example, a caller with independently verified three-hart
identity enumeration may select two compute workers and a separate controller:

```python
from merlin.runtime.backends.zephyr_model import build_app

image = build_app(
    model_directory, fresh_build_directory,
    board=selected_board,
    cpus=3, n_harts=2,
    worker_harts=(0, 1), controller_hart=2,
    cpu_hart_ids=(0, 1, 2),
    matrix=selected_matrix_routing,
)
```

All placement options default to `None`, retaining the existing harness and
runtime discovery. Explicit worker lists disable vector-capability discovery
for pool placement and pin the existing OpenMP team to that exact list. The
first selected worker runs serial model regions and team member zero. A
controller runs a separate pinned cooperative thread, starts that worker,
waits for its completion and leaves numerical work to the compute team.

`image['placement']` records physical worker/controller IDs, their OS CPU pin
indices, the online CPU map and their separate counts. The optional map is
ordered by OS CPU index, and must be distinct and complete. The pure
`plan_placement` API can translate a declared nonidentity map, but the current
Zephyr builder refuses it: its existing device-tree emission has only qualified
identity enumeration. Supplying a map is not proof that the target boots with
that enumeration. Independently verify actual CPU and hart IDs in the guest.

## Preserve ownership and context constraints

The existing OpenMP runtime owns one global team and one active outer region.
Nested regions execute serially on their encountering worker. The current
matrix wrappers use shared pack/scale/output scratch and are not general
reentrant services for independent concurrent matrix calls. Static loop
partitioning assigns whole output tiles to workers; shared inputs must remain
read-only until all workers finish. Pinning alone does not establish any of
these ownership properties for a different provider kernel.

Generic RVV save/restore does not save a matrix register file. Preemptible matrix
threads, migration and competing matrix callers require an independently
qualified target-specific ownership/context implementation. A placement plan
explicitly leaves tile-context safety and concurrent-call reentrancy unproven.
The current cooperative pool avoids voluntary context switches while waiting;
this is a scheduling constraint, not a substitute for arbitrary task switching.

A scalar controller role grants no asynchronous DMA API. Shared-buffer transfer
and completion protocols, interrupts, timer behavior, per-hart ISA and full
numerical output checking remain separate obligations. Neither successful pinning
nor an OS CPU count proves every requested worker actually executed.
