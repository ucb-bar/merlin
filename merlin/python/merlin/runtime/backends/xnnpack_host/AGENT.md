# AGENT.md — merlin/python/merlin/runtime/backends/xnnpack_host

## Purpose

HOST XNNPACK kernel backend for the dispatch runtime — the "third e2e column" scaffolding
next to baseline (hand_v0) and ours-optimized. Routes the dispatches XNNPACK covers (today:
plain f32 `linalg.matmul` GEMMs) through XNNPACK's own microkernel instead of the
Merlin-compiled `.so`; every other dispatch (attention as a batched generic, rmsnorm,
elementwise) falls through to the existing compiled path UNCHANGED. A hybrid kernel-backend
swap that isolates how much of the e2e gap is kernel-level vs runtime/glue-level.

## What belongs here

- `xnn_gemm_shim.c` — a clean `merlin_xnn_gemm_f32(M,N,K,A,B,C)` entry that packs operands
  (goi layout, zero bias, identity clamp) and drives the XNNPACK scalar f32 GEMM ukernel
  (`xnn_f32_gemm_minmax_ukernel_4x4__scalar`), vendored verbatim from `tmp/kernels/XNNPACK`.
- `shim/src/xnnpack/{common,math,microparams,gemm}.h` — minimal self-contained shims so the
  ukernel compiles standalone (NOT the real XNNPACK headers; mirrors the ceiling-driver shim).
- `__init__.py` — builds/caches the host `.so` (`output/.xnnpack_host/`), the ctypes
  `gemm_f32`, and `classify_matmul_kernel` (per-kernel routing classifier).

## How it wires in

`dispatch_runtime.execute(..., kernel_backend="xnnpack")` (or `run_model(..., kernel_backend=
"xnnpack")`, or env `MERLIN_XNNPACK_HOST=1`). Default-off: with no flag the runtime is
byte-for-byte the default compiled path.

## Invariants

- Default-off and additive. Never changes existing runtime behavior.
- Portable scalar microkernel on host: math is bit-comparable to the compiled kernel, no SIMD
  / runtime detection. HOST correctness only — board (RVV) cross-compile + timing is a
  separate later step (reuses the SAME XNNPACK ukernel family via the K1 ceiling drivers).
- Only routes dispatches it can prove faithful (2-D f32 matmul whose A/B are kernel args).
  Anything else falls through to the compiled `.so`. No silent wrong-math.
