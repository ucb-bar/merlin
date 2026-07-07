# Cross-workload primitive coverage & regret

> Which candidate primitive shapes a future DSE search space should include because they cover the real operator geometry broadly, and which overfit a single workload. **Structural geometry only — no speedup.**

## Coverage regret across workloads (10% waste band)

| primitive | avg cov | worst cov | best cov | max regret | poorly-served clusters |
|---|---|---|---|---|---|
| tile_8x16 | 68% | 0% | 100% | 100% | gemv_like |
| tile_8x8 | 68% | 0% | 100% | 100% | gemv_like |
| tile_16x16 | 48% | 0% | 99% | 99% | gemv_like |
| tile_16x32 | 48% | 0% | 99% | 99% | gemv_like |
| tile_32x32 | 48% | 0% | 99% | 99% | gemv_like |
| gemv_lane_64 | 80% | 13% | 100% | 87% | projection_like; squareish_gemm; unknown |
| gemv_lane_128 | 80% | 13% | 100% | 87% | projection_like; squareish_gemm; unknown |
| gemv_lane_256 | 75% | 13% | 100% | 87% | projection_like; squareish_gemm; unknown |

## Findings

- **Widest average structural coverage:** `gemv_lane_64` at 80% average per-workload coverage under 10% waste — **suggests this primitive should be included in the future DSE search space.**
- **Worst cross-workload regret:** `tile_8x16` (max_regret 100%: best 100% vs worst 0%) — **suggests this primitive is workload-specific**, not a general choice.
- **Overfit primitives:** `tile_8x16` covers `bitvla; pi05; rdt; smolvla; xr0` well but poorly covers the worst workload (0%).

**Caveat:** these are structural tile/lane coverage metrics — padding waste and utilisation are pure geometry. **No speedup**, latency, or performance is implied, and no hardware is assumed.
