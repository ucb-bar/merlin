# AGENT.md — merlin/python/merlin/runtime

## Purpose

The Python reference runtimes that execute Merlin command buffers / dispatch programs, plus
the per-backend adapters under `backends/`.

## What belongs here

- `simulator.py` / `reference.py` / `tensor.py` / `metrics.py` / `commandbuffer.py` — the
  synthetic-workload command-buffer engine (integer tensor math + metrics) and an
  independent reference recomputation it is gated against.
- `dispatch_runtime.py` — the **whole-model** host executor: outlines a captured model,
  compiles each kernel in isolation (`llvmlower.kernel_backend`, deduplicated by kernel
  body), evaluates the driver's view ops (`expand/collapse_shape`, `extract_slice`,
  `concat`, `splat`, `constant`) in numpy, invokes the compiled kernel symbols in order,
  and gates the output against the torch golden. Forward args (inputs + safetensors weights
  + extra buffers) are bound exactly as the C runtime binds them. Verified: whole
  small_llama (cos 0.9999999) and TinyLlama-1.1B (cos 1.0, next-token argmax exact on all
  tokens) == torch through the dispatch table. Kernels with a by-value scalar arg (e.g. a
  `cumsum` accumulator-init `i64`) are passed by value via `abi.ScalarArg`, NOT as a memref
  descriptor — `emit_c_interface` only wraps memrefs.
- `backends/` — host / spike adapters.

## What does not belong here

- The deployable C runtime (that is `merlin/runtime/`, outside the Python tree).
- Generated artifacts (write those to `output/`).

## Invariants

- Real implementations only; every result is gated (simulator == reference; dispatch
  runtime == torch golden). `dispatch_runtime` raises on any view op it cannot evaluate —
  no silent skips.
- Every subdirectory must also contain an AGENT.md.
