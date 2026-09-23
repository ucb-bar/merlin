---
title: "Macro scheduling: a stream-level plan over compute groups"
kind: design
status: draft
owner: core
last_verified: 2026-09-17
related: [static_arena_wiring, command_stream_reorder_emitter, compiler_plane]
code_refs:
  - src/merlin/xdsl_dialects/lowering/stream_plan.py
  - src/merlin/xdsl_dialects/lowering/compute_groups.py
  - src/merlin/xdsl_dialects/lowering/dispatch_program.py
  - src/merlin/xdsl_dialects/lowering/arena_plan.py
  - src/merlin/xdsl_dialects/lowering/global_plan.py
---

# Macro scheduling: a stream-level plan over compute groups

## Scope

Two levels of scheduling exist in a compiled model, and they have different owners.

*Inside a group*: how one contraction and its absorbed stages are tiled, ordered and issued to a
unit. That is the kernel body per group key, owned by the fine-grained scheduling work.

*Between groups*: which buffers exist and for how long, what configuration a unit is left in from
one group to the next, which host regions can run while a device group is in flight, and where the
program must wait. That is this document. It is the level a dataflow compiler expresses in a
stream dialect: partitioned, resource-typed, asynchronous, with explicit synchronisation.

The unit of scheduling is the **compute group** (see the compute-groups log). Before groups
existed every linalg operation was a dispatch, and a plan over four thousand single-op dispatches
describes the capture, not the program. Over groups it describes what runs: one device group per
contraction with its epilogue, one host region per fused run of host work.

## What already exists

| need | in tree | state |
|---|---|---|
| a program of dispatches over typed buffers | `lowering/dispatch_program.py` (`DispatchProgram`, `Node`, `Buffer`; `verify_program`, `slice_program`) | built from the outliner; now one node per compute group |
| whole-program memory plan | `lowering/arena_plan.py` (`plan_arena`, `pack_disjoint`) | `pack_disjoint` is on the measured path; `plan_arena` is analysis only |
| alternatives per region with transitions and cycle intervals | `lowering/global_plan.py` (`RegionAlternative`, `TransitionAlternative`, `GlobalPlan`, `verify_global_plan`) | data model and verifier; no chooser wired to groups |
| asynchronous interface ops | `interface.async_copy`, `interface.await`, `interface.event`, `interface.fifo.*` | defined, verified, unused by a model route |
| submission and events | `runtime.command_buffer.*`, `runtime.submit`, `runtime.wait`, `runtime.event` | used by the device offload rewrite as a pass-through stage |
| configuration as dataflow | xDSL `accfg` (`setup` threads a state, `launch` returns a token, `await` consumes it) | shipped in the pinned xDSL; nothing uses it |

So this is a composition job. No new dialect is proposed: the payload stays `func`/`linalg`/
`tensor`, groups are outlined functions with `merlin.group` and `merlin.placement`, the timeline
uses `accfg` for configuration and launch and the existing `runtime`/`interface` events for
movement, and the plan is data (`stream_plan_v1`) beside the dispatch program.

## The four decisions

### 1. Resource lifetimes

Every buffer of the dispatch program is one of four kinds, decided from the program and the group
plan, never declared by hand:

* **constant** — a model argument that is a weight, or a value computed only from constants.
  Packed, quantized to the accumulator's domain where a group needs it (an integer bias, a folded
  requantization multiplier) and laid out once, offline. This is where the per-layer host work that
  is not per-inference belongs, and today much of it runs per inference.
* **external** — model inputs and results. Bound by the caller; never in the arena.
* **transient** — produced by one group and consumed by later ones. Lives in the arena from its
  producer to its last consumer.
* **staging** — a transient that crosses the host/device boundary: written by one side and read
  by the other. It is a transient with a transfer attached, and it is the quantity to minimise.
  Two adjacent device groups whose intermediate stays resident on the unit have no staging buffer
  between them at all.

`arena_plan.pack_disjoint` already packs transients by live range. What changes is the liveness it
is given: last USE over the group program, which is what the group boundaries make safe to free.

### 2. Configuration state

A unit is configured (dataflow mode, activation, scale, strides, pool window) and then launched.
Emitting the full configuration before every launch is correct and wasteful; emitting only what
changed requires knowing what the unit was left in, which a command stream does not say.

`accfg` says it: `accfg.setup` takes the previous state and returns the next, so configuration is
an SSA value threaded through the group sequence. Two consecutive groups with equal configuration
fields share a state and the second setup folds away; a host region between them does not disturb
the state unless it touches the unit. Deduplication is then ordinary dataflow, and the group key
(stages, numerics contract, extents) is what decides which fields two groups share.

The register fields come from the target's derived register layouts, so which fields exist and
which a group sets are data. Nothing here knows a register's name.

### 3. The timeline

`accfg.launch` returns a token and `accfg.await` consumes it, so a device group is launched, host
work that does not depend on it runs, and the program waits only where a value is needed. A host
region is independent of an in-flight device group when it reads none of that group's results,
directly or through anything between them in the program. The overlap a program can have is a
property of its dependence graph over groups, and `stream_plan` measures it before anything is
built: how much host work (in elements) is independent of the device group issued before it.

A fence is placed only at a true dependence: a host region that reads a device result, a device
group that reads a host result, or the end of the program. The current emission fences after every
device command because nothing told it otherwise.

### 4. Placement of host groups in time

Host regions are schedulable units like any other. Two that are independent of each other and of
the device can run on separate harts; the dispatch program already expresses that as a DAG. It is
listed last because it is worth the least until the host term is small.

## Order of implementation

1. `stream_plan`: lifetimes, fences and overlap, as an analysis over the grouped dispatch program.
   Done with this document; numbers are in the devlog.
2. Offline prepack as a compiler stage: constant-kind buffers computed once, including the
   accumulator-domain bias and requantization multiplier a closed integer group needs.
3. Configuration threading with `accfg` and setup folding, behind the group emission.
4. Asynchronous launch and await with fences at true dependences only.

Steps 2 to 4 change emitted programs and each is corroborated on FireSim against the previous
checkpoint. Step 4 is gated on the measurement the plan calls for: overlap is worth building when
the host term it would hide is no longer the whole program.

## What this does not decide

The kernel body of a group, tiling and loop order inside it (the fine-grained scheduling work);
which groups exist (group formation); what a unit admits (the capability contract and the readout
facet). This level consumes all three.
