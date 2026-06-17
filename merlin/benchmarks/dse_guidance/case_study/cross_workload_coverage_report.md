# Cross-workload primitive coverage & regret

> Which candidate primitive shapes a future DSE search space should include because they cover the real operator geometry broadly, and which overfit a single workload. **Structural geometry only — no speedup.**

## Coverage regret across workloads (10% waste band)

| primitive | avg cov | worst cov | best cov | max regret | poorly-served clusters |
|---|---|---|---|---|---|
| tile_8x16 | 58% | 0% | 100% | 100% | gemv_like |
| tile_8x8 | 58% | 0% | 100% | 100% | gemv_like |
| tile_16x16 | 30% | 0% | 88% | 88% | gemv_like; wide_skinny |
| tile_16x32 | 30% | 0% | 88% | 88% | gemv_like; wide_skinny |
| tile_32x32 | 22% | 0% | 88% | 88% | gemv_like; wide_skinny |
| gemv_lane_64 | 68% | 13% | 100% | 87% | squareish_gemm |
| gemv_lane_128 | 66% | 13% | 100% | 87% | squareish_gemm |
| gemv_lane_256 | 44% | 8% | 100% | 92% | squareish_gemm |

## Findings

- **Widest average structural coverage:** `gemv_lane_64` at 68% average per-workload coverage under 10% waste — **suggests this primitive should be included in the future DSE search space.**
- **Worst cross-workload regret:** `tile_8x16` (max_regret 100%: best 100% vs worst 0%) — **suggests this primitive is workload-specific**, not a general choice.
- **Overfit primitives:** `tile_8x16` covers `rdt; small_llama` well but poorly covers the worst workload (0%).

**Caveat:** these are structural tile/lane coverage metrics — padding waste and utilisation are pure geometry. **No speedup**, latency, or performance is implied, and no hardware is assumed.
