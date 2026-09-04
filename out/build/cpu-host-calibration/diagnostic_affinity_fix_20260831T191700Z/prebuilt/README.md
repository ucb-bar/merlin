# rvvhost compiler

This package is a raw C++ out-of-tree compiler for the v1 Merlin capsule descriptor. It parses only the
generic capsule attributes and emits a fresh C ABI kernel, post-policy MLIR, and metadata for each
invocation. Unsupported modes, attributes, types, operations, layouts, hart counts, and VLEN requests are
errors; there is no fallback path.

The lowering has three legal forms:

- `scalar` uses the source-order reference loops.
- `rvv` uses RVV 1.0 intrinsics with `vsetvl` on every chunk. No fixed VLEN is assumed and every load and
  store is bounded by the returned VL. Scalar source-order loops remain in the generated source as the
  non-RVV reference semantics, but an RVV request is never reported as scalar.
- `rvv_multicore` divides the operation's independent outer iteration space with
  `[work*h/harts, work*(h+1)/harts)`. Threads are pinned to distinct CPUs from the harness-provided
  affinity set and the caller's original affinity is restored through the wrapped pthread ABI after
  joining. A multi-hart restore earns no worker attribution. Ordered reductions remain a
  single exact source-order partition; all requested workers are still created and affinity-checked.

Vector contraction is legal for contiguous row-row columns and within one 8-column packed panel. The
lowering stops at row or panel boundaries, so a dynamic final VL cannot cross layout regions. FP32 uses
vector fused multiply-add. Signed int8 products are widened first to i16 and then i32 before accumulation.
Transposed-RHS contraction uses bounded strided vector loads across output columns. Operations with
nonlinear scalar library semantics retain source order and receive a bounded vector
load/compute/store materialization step, preserving the same numerical result.

The policy uses scalable VL only. Consequently the runtime-verified fixed-VLEN specialization described by
the target contract is deliberately not enabled.
