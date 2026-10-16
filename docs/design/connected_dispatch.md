---
title: Connected dispatch planning
kind: design
status: current
owner: core
last_verified: 2026-10-16
code_refs: [src/merlin/xdsl_dialects/lowering/connected_dispatch.py]
---

# Connected dispatch planning

`plan_connected_dispatches` is a target-neutral first step toward avoiding
unnecessary host round trips between accelerator operations. It reads Merlin's
whole-model `DispatchProgram`, then asks a selected target for a direct-handoff
interface, storage capacity, numerical equivalence, and measured cost. It
selects only adjacent dispatches with one private intermediate value. The
target backend must report each realized handoff through `require_realized`;
otherwise compilation refuses the optimized path.

`lower_connected_dispatches` selects the plan, calls the chosen backend, and
requires the backend to account for the selected edges. Backends consume the
plan while the typed producer and consumer are still available and check their
physical buffer lifetimes. An adapter can implement this interface without
importing Merlin: its facts are plain dictionaries and its lowered artifact
reports the edges it actually realized.

The same decision interface can serve regular Gemmini's accumulator or
scratchpad handoff and MX Gemmini's matrix, VPU, and resident requantization
handoffs. Their adapters supply distinct legality and cost facts; Merlin does
not encode their commands. Unknown precision effects, insufficient capacity,
fanout, observable outputs, intervening operations, and multi-result kernels
currently leave the original materialized path in place.

The first executable adapter is MX's contiguous BF16 VPU path. It derives
arithmetic and scalar constants from outlined typed MLIR and removes the
intermediate DMA store and reload when their scratchpad spans match. Both VPU
arithmetic commands and all synchronization remain in their original order.
PyTorch captures from model2MLIR main `905d2c9` passed full-output bit checks
against PyTorch and an independent BF16 oracle on Nicolas's Spike model at two
shapes. The MX repository archives the generated IR, command streams, inputs,
goldens, and logs in `docs/evidence/connected_dispatch_20261016`.

Cost may be unknown during qualification: explicit calibration mode selects
legal candidates without inventing savings. Normal selection requires a
positive measured cost. The qualification recompiles with its measured cost
and checks that it selects the same commands. That measurement applies to the
captured graph, selected profile, and recorded Spike timing configuration.

This path does not yet optimize full TinyLlama or SmolVLA graphs: their f32
boundaries, matrix/VPU/requantization chains, and multi-result state require
additional adapters and numerical qualification. A regular Gemmini adapter
against its accumulator path also remains to be implemented. Each new path
must compare optimized and original outputs and command traffic on the same
input. A different BF16 or FP8 rounding point requires a separate numerical
policy and qualification; it cannot claim equivalence to the materialized path.
