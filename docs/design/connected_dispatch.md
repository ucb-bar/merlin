# Connected dispatch planning

`plan_connected_dispatches` is a target-neutral first step toward avoiding
unnecessary host round trips between accelerator operations. It reads Merlin's
whole-model `DispatchProgram`, then asks a selected target for a direct-handoff
interface, storage capacity, numerical equivalence, and measured cost. It
selects only adjacent dispatches with one private intermediate value. The
target backend must report each realized handoff through `require_realized`;
otherwise compilation refuses the optimized path.

The plan does not rewrite a kernel or remove a transfer on its own. Backends
must consume it while the typed producer and consumer are still available,
form a connected region, and check their physical buffer lifetimes. Until that
integration exists, the plan is an analysis result and gives no speedup.

The same decision interface can serve regular Gemmini's accumulator or
scratchpad handoff and MX Gemmini's matrix, VPU, and resident requantization
handoffs. Their adapters supply distinct legality and cost facts; Merlin does
not encode their commands. Unknown precision effects, insufficient capacity,
fanout, observable outputs, intervening operations, and multi-result kernels
currently leave the original materialized path in place.

The next executable step is an MX adapter that connects this plan to the
existing typed resident-chain lowerer, followed by a regular Gemmini adapter
against its accumulator path. Each must compare the optimized and original
outputs and command traffic on the same input before enabling a model-wide
policy. In particular, a different BF16 or FP8 rounding point requires a
separate numerical policy and qualification; it cannot claim equivalence to
the materialized path.
