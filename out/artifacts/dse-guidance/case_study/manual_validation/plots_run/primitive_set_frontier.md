# Primitive-set frontier (all)

> A primitive SET covers an op if ANY member tiles it under 10% pad waste. The headline search-space result: one primitive is not enough — the best single primitive leaves a workload badly covered (low worst-workload), while a {tile + GEMV-lane} pair covers the corpus. Structural coverage only, no performance.

| set size | best primitive set | worst-workload | macro (mean) | micro (MAC-wt) | max regret |
|---|---|---|---|---|---|
| 1 | gemv_lane_64 | 0.13 | 0.80 | 0.64 | 0.87 |
| 2 | gemv_lane_64 + tile_8x16 | 1.00 | 1.00 | 1.00 | 0.00 |
| 3 | gemv_lane_128 + gemv_lane_64 + tile_8x16 | 1.00 | 1.00 | 1.00 | 0.00 |
