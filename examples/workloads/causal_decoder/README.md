# Iteration decoder

`loader.py` supplies the existing eight-token causal prefill model: embedding,
RMSNorm, four-head attention, residual projection, gated MLP and vocabulary
head. It uses deterministic small weights and is not a TinyLlama checkpoint.

`step.py` reuses those modules and weights for one token. Its inputs are:

| Input | Shape | Meaning |
|---|---|---|
| `tokens` | `[1,1]`, i64 | Vocabulary index in `[0,64)`. |
| `key_cache`, `value_cache` | `[1,4,35,8]`, f32 | Caller-owned cache state. |
| `positions` | `[1]`, i64 | Cache index in `[0,35)`. |
| `mask` | `[1,1,1,35]`, f32 | Positive zero for visible entries, negative infinity otherwise. |

Results are logits `[1,1,64]`, updated keys, updated values, then probabilities
`[1,4,1,35]`. The caller masks uninitialized entries and commits returned state
only after a successful invocation. Inputs and outputs use separate buffers.
The interface has fixed capacity; it does not perform cache eviction.

The wrapped module also retains its original `decoder.mask` buffer. Although
the step uses the runtime mask, the capture still includes that buffer. Declare
it as read-only captured state, and declare the two caches as caller-initialized
state updated by outputs 1 and 2. Use the generated captured-state initializer
and input descriptors: the compiled interface includes this state input in
addition to the five forward arguments.

Capture either loader with the existing capture worker. Select target precision
and host operations separately before compilation. An FP8/BF16 target changes
the source arithmetic and needs separate original-model quality measurements.
The step interface tests stateful composition; it does not replace prefill,
establish checkpoint accuracy or require device-resident caches.
