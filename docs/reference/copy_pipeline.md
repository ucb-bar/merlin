---
title: Qualified input-copy pipelines
kind: reference
status: current
owner: core
last_verified: 2026-10-20
related: [architecture, shared_pass_phase2]
code_refs: [src/merlin/compile/scheduling/copy_pipeline.py, src/merlin/compile/scheduling/schedule_pipeline.py, merlin/tests/infra/test_copy_pipeline.py, merlin/tests/targetgen/test_shared_pass_pipeline_consumption.py]
---

# Qualified input-copy pipelines

`merlin.compile.scheduling.copy_pipeline` hoists existing asynchronous input-copy
launches into explicitly assigned private buffer slots. It preserves every
instruction, launch order, completion wait, arithmetic operation, descriptor and
configuration operation. No hardware loop instruction is required.

The existing `temporary_storage.prefetch_private_copies` already supports qualified
input prefetch across synchronous memory operations. Its original admission permits
only one outstanding copy and excludes shared backing banks. The new family adds
separately qualified concurrent copies, motion across pure token waits, and
independent regions of a shared backing bank. It does not widen that existing
admission. An explicit `copy_pipeline_motion` selection connects it to the ordinary
schedule pipeline after allocation. Unselected compilation keeps its existing route.

`staged_dense` separately models descriptor issue, compute consumption and completion
fences. Its checker remains necessary for those operations. The copy pipeline treats
descriptor and configuration operations as barriers; it cannot establish a device
compute pipeline from memory capacities alone.

## Selected software and RTL contract

`resolve_copy_pipeline_admission` accepts an explicit software declaration, selected
RTL facts, the independently qualified `StorageAdapter`, and a separate original
execution-owner qualification identity. Capacity fields must be structural fact
references. Missing, unknown or nonpositive capacities refuse the family.

```python
declaration = {
    "schema": "merlin.copy_pipeline.v1",
    "crossing_pairs": [
        ["input_copy", "token_wait"],
        ["input_copy", "integer_transform"],
    ],
    "concurrent_pairs": [["input_copy", "input_copy"]],
    "max_in_flight": {"fact": ["resources", "copy_engine", "in_flight"]},
    "queue_capacity": {"fact": ["resources", "copy_engine", "queue_entries"]},
    "independence": "disjoint_memory_preserves_state_and_issue_order",
    "wait_semantics": "token_completion_only_preserves_state",
}
```

These names are illustrative software instruction names. Each selected pair needs
independent qualification for the actual instruction implementations, ports,
configuration and descriptor state. `crossing_pairs` qualifies launch motion over
pure token waits or synchronous exact copies/qualified elementwise operations.
`concurrent_pairs` qualifies simultaneously pending input copies in their preserved
issue order. Read/write independence is checked separately at actual addresses,
including aliases of one physical bank. External aliases must likewise use a
common canonical address space and byte coordinates in the adapter's effects.

A global drain that completes every currently outstanding operation is not
automatically a pure wait for its original token. An adapter must faithfully decode
the realized completion effects after scheduling. A completion declaration inferred
from instruction names, capacity, or issue order is insufficient.

The admission records digests of the exact supplied facts and software declaration,
selected hardware identity and both qualification identities. These records establish
neither a sandbox grant nor execution qualification. The original target owner must
bind the contract to the selected hardware revision and final emitted effects.

## Transformation and tuning

`pipeline_private_copies` takes the existing actual-address kernel, private objects,
byte-addressed banks, reservations and fixed external windows. Private slots must fit
physically before the transformation; virtual addresses cannot establish independence.
The existing temporary-storage checker verifies complete initialization, ownership,
typed alias views, physical capacity and reservations.

With two independent private slots, eligible source order such as:

```text
copy panel 0 into slot A
wait for panel 0
transform and store panel 0
copy panel 1 into slot B
wait for panel 1
transform and store panel 1
```

can become:

```text
copy panel 0 into slot A
copy panel 1 into slot B
wait for panel 0
transform and store panel 0
wait for panel 1
transform and store panel 1
```

`CopyPipelinePolicy` exposes finite prefetch distance, maximum moved launches and
maximum pending copies. The effective pending limit is the minimum of the policy,
fact-derived in-flight capacity and fact-derived queue capacity. The default pending
limit is one. Two or more copies require explicit selection and qualification. A
missing pair or a dependency stops motion; no instruction or wait is invented.
These bounds describe the qualified input-copy resource. Additional queue traffic,
shared execution resources and their progress rules must be covered by the selected
overlap qualification; this checker does not infer a whole-device queue model.

`copy_pipeline_motion(admission, policy=..., budgets=...)` binds the original
qualified proposer and separate checker to `compile_schedule(..., copy_pipeline=...)`.
The fixed storage stage checks actual allocated addresses before movement and
rebuilds lifetime and completion analyses afterward. If compaction reused slots,
one bounded repair may retain original slots only when those placements fit and
verified motion occurs. Otherwise the compact route remains. A simultaneously
selected deletion-only post-storage rewrite conservatively disables this motion.

Phase 2 can tune distance, moved-launch count and pending-copy count within the
original bounds. It cannot change facts, qualification, decoders or legal stage
ordering. The default pending count remains one; increasing it requires a selected
qualified policy and sufficient fact-derived capacity. This API supplies no
handwritten target implementation or modification of a frozen compiler release.

## Independent final checks and limits

`verify_copy_pipeline` checks the realized calls independently of the search. It
requires exact source-call membership and boundaries, preserved launch/nonlaunch
order, every inverted dependency's qualified pair, motion budgets, actual tokens and
waits, physical memory hazards, concurrency qualification and pending-copy occupancy. The
existing storage checker then rechecks the realized addresses. The ordinary pipeline
repeats the independent copy check against final decoded bytes. Partial token waits
retain pending work; they cannot masquerade as global drains. Source, contracts,
policy and realized code are checked for mutation during callbacks.

Tests execute the transformed payload in a separate delayed-copy byte interpreter:
bytes become visible only at actual waits. They compare complete outputs, guards,
private state and token completion order across different widths, alias views and
two/three slots. Mutations exercise missing waits, premature slot reuse, external
aliasing, changed addresses, launch reordering, capacity and qualification failures.
Its event clock is synthetic functional evidence; reports retain `timing: UNKNOWN`.

The ordinary target-IR tests also compile and link an independent C token-queue
machine through the selected-IR consumer. Its actual fetch instructions enqueue
transfers; only the matching wait publishes bytes. Two packet widths/byte orders
and shared/separate backing models preserve complete outputs, three carried-state
epochs, inputs and guards. The loaded command image must match selected operands;
ignoring selected order, deleting a wait or breaking wait visibility fails. These
tests observe concurrency occupancy, not accelerator latency or hardware support.

This bounded family accepts straight-line schedules and existing private slots. It
does not synthesize buffers, unroll loops, remove completion instructions, schedule
asynchronous arithmetic/readout, or cross unknown state/descriptor effects. Final
machine emission, decoded-effect correspondence and execution qualification remain
required from the original target owner. Consequently this family alone establishes
no measured accelerator speedup.
