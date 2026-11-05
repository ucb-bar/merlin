# Stateful attention and MLP

One invocation projects a new token to Q/K/V, replaces one K/V cache slice,
attends over the masked cache, applies the output projection and a residual
ReLU MLP, and returns four results in order:

1. Token output `[1,32]`.
2. Updated key cache `[1,35,32]`.
3. Updated value cache `[1,35,32]`.
4. Attention probabilities `[1,1,35]`.

Runtime inputs are `values`, `key_cache`, `value_cache`, `positions`, and
`mask`. `positions` contains one signed-i64 index in `[0,35)`. A mask entry is
exactly positive zero or negative infinity, with at least one unmasked slot.
The caller masks uninitialized slots and advances cache state only after a
successful call. The cache has fixed capacity; this interface does not track
a growing length or perform eviction automatically.

Declare `key_cache` and `value_cache` as caller-initialized updated state,
bound to outputs 1 and 2, using the staged argument indices and program hash.
Inputs and outputs use disjoint buffers. No mutation of a Python object is
part of the compiled interface.
The [host/device compilation guide](../../../docs/reference/host_device_compilation.md#explicit-model-state)
describes the state declaration and C interface.

This is an independently authored integration workload, not a checkpoint or a
replacement for the required iteration models. It exercises multiple results,
shared updated caches, dynamic indexing, whole-row masked softmax across a
column-tile boundary, and repeated host/device invocations. Precision changes
must be selected and evaluated separately. Passing it does not establish
device-resident KV storage or a complete pretrained decoder session.
