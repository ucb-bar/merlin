# K1 whole-model SAME-PASS head-to-head — rdt2_fp32_consistent

Board: SpacemiT K1 (real RVV silicon, VLEN=256). N=3 runs/config, ONE pass, min CLOCK_MONOTONIC wall + spread. Timer: CLOCK_MONOTONIC wall_ns; cycle_accurate=false.
XNNPACK kernel: `xnn_f32_gemm_ukernel_1x4v__rvv` (resident-weight pack, excluded from timed path). cos gated vs host golden before any wall.

| config | min wall (ns) | range % (N) | fp32 cos | speedup | #xnn | ok | blocker |
|---|---|---|---|---|---|---|---|
| baseline (hand_v0) | 74,041,641,830 | 0.05% (3) | 1.0000001 | —x | 0 | yes | — |
| ours-wholemodel (accum-resident, tail-safe) | 31,398,216,725 | 0.68% (3) | 1.0000000 | 2.3581x | 0 | yes | — |
| ours-wholemodel-vf (.vf, no broadcast ladder) | 30,273,769,192 | 0.11% (3) | 1.0000000 | 2.4457x | 0 | yes | — |
| xnnpack-kernels (RVV ukernel, resident pack) | 18,971,288,080 | 0.19% (3) | 1.0000001 | 3.9028x | 23 | yes | — |
| openblas-kernels (sgemm 8x8, resident pack) | 20,315,939,413 | 0.37% (3) | 1.0000001 | 3.6445x | 0 | yes | — |

## Headline (same-pass, fair resident-weight pack)
- **best-ours (ours_wholemodel_vf) / xnnpack = 0.6267x** — XNNPACK faster than ours (>1 ⇒ our compiler kernel faster).
- speedups vs baseline: tiled —x · v3 —x · wholemodel 2.3581x · xnnpack 3.9028x.

## Takeaway
Same-pass head-to-head vs the SAME baseline in ONE pass. xnnpack-kernels routes f32 linalg.matmul to xnn_f32_gemm_ukernel_1x4v__rvv with RESIDENT-WEIGHT pack (excluded from the timed path, matching ours' pack-free scope). cos gated before any wall.
