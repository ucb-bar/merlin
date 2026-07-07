# Measured dispatch coupling (host reference executor)

> The dispatch *count* per forward is measured by running the real captured model through the dispatch runtime (cos=1.0 vs the torch golden). It grounds the `dispatches per replan` input to `command_batching` / `autonomous_K_loop`, which was previously estimated from the matmul count. No speedup is claimed — the per-dispatch host cost on the deployable runtime is still required to quantify a benefit.

| model | matmul estimate | measured dispatches | undercount | unique | cos |
|-------|-----------------|---------------------|-----------|--------|-----|
| small | 15 | 183 | 12x | 159 | 1.0000 |
| small_llama_int8 | 15 | 198 | 13x | 163 | 1.0000 |
| small_llama_fp8 | 15 | 213 | 14x | 177 | 1.0000 |

**Finding:** the matmul-count proxy under-counts real dispatch granularity by ~13x (real dispatches include every elementwise/norm/view/glue kernel). So the command-batching / autonomous-loop opportunity is *larger* than the matmul-only estimate implied — now grounded in a measured dispatch count.

## Per-dispatch host cost (host reference executor)

| model | dispatches | compute-call ms | dispatch/alloc overhead ms | overhead frac |
|-------|-----------|-----------------|----------------------------|---------------|
| small | 183 | 243.7 | 410.7 | 0.63 |
| small_llama_int8 | 198 | 242.2 | 436.3 | 0.64 |
| small_llama_fp8 | 213 | 259.9 | 445.7 | 0.63 |

**Finding:** ~63% of host time per forward is dispatch/allocation overhead, NOT compute-kernel calls — the forward is **host-dispatch-bound** on this executor, which is exactly the regime `command_batching` / `autonomous_K_loop` target. Absolute ms are host-interpreter (Python reference executor), machine-dependent — the stable, deployable-relevant signals are the dispatch *count* and the overhead *fraction*, not the absolute latency.
