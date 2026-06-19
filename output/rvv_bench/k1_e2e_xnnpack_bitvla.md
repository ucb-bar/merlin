# K1 whole-model e2e: XNNPACK kernels vs baseline vs ours — bitvla_fp32_consistent

Board: SpacemiT K1 (real RVV silicon, VLEN=256). N=3 runs/config, min CLOCK_MONOTONIC wall. Timer: CLOCK_MONOTONIC wall_ns (rdtime ticks alongside); cycle_accurate=false.
XNNPACK kernel: `xnn_f32_gemm_ukernel_1x4v__rvv`. cos gated vs host golden before any wall.

| config | min wall (ns) | fp32 cos | #dispatch via XNNPACK | ok | blocker |
|---|---|---|---|---|---|
| baseline (hand_v0) | 2,516,602,405 | 0.9999946 | 0 | yes | — |
| ours-optimized (fused_vfmacc_tiled) | 274,157,629 | 0.9999946 | 0 | yes | — |
| xnnpack-kernels (RVV ukernel) | 184,397,050 | 0.9999927 | 15 | yes | — |

## Speedups (min wall)
- baseline / ours-optimized = 9.1794x
- baseline / xnnpack-kernels = 13.6477x
- ours-optimized / xnnpack-kernels = 1.4868x (>1 ⇒ XNNPACK faster than our vfmacc; <1 ⇒ our vfmacc faster)

## Takeaway
Third e2e column. xnnpack-kernels routes the f32 linalg.matmul dispatches to XNNPACK's xnn_f32_gemm_ukernel_1x4v__rvv; rest on the Merlin runtime. cos gated vs the same host golden before any wall is reported.
