# Measured dispatch coupling (host reference executor)

> The dispatch *count* per forward is measured by running the real captured model through the dispatch runtime (cos=1.0 vs the torch golden). It grounds the `dispatches per replan` input to `command_batching` / `autonomous_K_loop`, which was previously estimated from the matmul count. No speedup is claimed — the per-dispatch host cost on the deployable runtime is still required to quantify a benefit.

| model | matmul estimate | measured dispatches | undercount | unique | cos |
|-------|-----------------|---------------------|-----------|--------|-----|
| small | 15 | 183 | 12x | 159 | 1.0000 |
| small_llama_int8 | 15 | 198 | 13x | 163 | 1.0000 |
| small_llama_fp8 | 15 | 213 | 14x | 177 | 1.0000 |

**Finding:** the matmul-count proxy under-counts real dispatch granularity by ~13x (real dispatches include every elementwise/norm/view/glue kernel). So the command-batching / autonomous-loop opportunity is *larger* than the matmul-only estimate implied — now grounded in a measured dispatch count.
