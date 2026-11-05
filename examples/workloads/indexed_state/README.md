# Runtime-indexed tensor state

Inputs are `cache [2,8,4]`, `update [2,1,4]` and signed-i64 `positions [1]`.
The function replaces one slice along axis 1, returning the updated cache and
its sum along axis 1. The runtime position must be in `[0,8)`; it can change
without recompilation. The cache has fixed capacity and does not grow.

The frontend must preserve `aten.index_copy` and lower its bounds checks and
dynamic `tensor.insert_slice`. In the selected mixed CPU/accelerator route,
the copy is compiled CPU code and the reduction uses the selected device
precision. Copying state preserves its f32 bytes; the reduction may change
arithmetic under an explicitly selected policy.

Declare argument 0 as caller-initialized state with update output 0 through
`--application-state`. After a successful call, pass that returned tensor to
the next invocation. Keep the old state after a failed call. Inputs and outputs
must use disjoint buffers. See Merlin's host/device compilation reference for
the declaration format. This example does not establish persistent device
residency, dynamic cache capacity or a complete attention/session result.
