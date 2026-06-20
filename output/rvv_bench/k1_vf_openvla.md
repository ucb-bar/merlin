# K1 whole-model SAME-PASS head-to-head — openvla_fp32_consistent

Board: SpacemiT K1 (real RVV silicon, VLEN=256). N=5 runs/config, ONE pass, min CLOCK_MONOTONIC wall + spread. Timer: CLOCK_MONOTONIC wall_ns; cycle_accurate=false.
XNNPACK kernel: `xnn_f32_gemm_ukernel_1x4v__rvv` (resident-weight pack, excluded from timed path). cos gated vs host golden before any wall.

| config | min wall (ns) | range % (N) | fp32 cos | speedup | #xnn | ok | blocker |
|---|---|---|---|---|---|---|---|
| baseline (hand_v0) | 5,854,968,575 | 1.64% (5) | 0.9999999 | —x | 0 | yes | — |
| ours-wholemodel (accum-resident, tail-safe) | 1,185,722,339 | 3.81% (5) | 1.0000000 | 4.9379x | 0 | yes | — |
| ours-wholemodel-vf (.vf, no broadcast ladder) | 1,088,892,682 | 2.3% (5) | 1.0000000 | 5.3770x | 0 | yes | — |
| xnnpack-kernels (RVV ukernel, resident pack) | 656,471,054 | 2.67% (5) | 0.9999999 | 8.9189x | 26 | yes | — |
| openblas-kernels (sgemm 8x8, resident pack) | 686,446,892 | 1.78% (5) | 0.9999999 | 8.5294x | 0 | yes | — |

## Headline (same-pass, fair resident-weight pack)
- **best-ours (ours_wholemodel_vf) / xnnpack = 0.6029x** — XNNPACK faster than ours (>1 ⇒ our compiler kernel faster).
- speedups vs baseline: tiled —x · v3 —x · wholemodel 4.9379x · xnnpack 8.9189x.

## Takeaway
Same-pass head-to-head vs the SAME baseline in ONE pass. xnnpack-kernels routes f32 linalg.matmul to xnn_f32_gemm_ukernel_1x4v__rvv with RESIDENT-WEIGHT pack (excluded from the timed path, matching ours' pack-free scope). cos gated before any wall.
