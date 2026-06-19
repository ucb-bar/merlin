# Primitive-set frontier robustness (all)

> The best primitive SET swept over set sizes 1–4 and pad-waste thresholds 5/10/20%, with extra candidate tiles (tile_4x16, tile_8x32, tile_16x64, tile_64x16, tile_4x32). Coverage is recomputed from operator geometry (the 10% recompute is regression-checked against the committed tile_waste). Structural coverage only — no primitive is called faster.

**Best 2-set across thresholds:** {'5pct': 'gemv_lane_64+tile_4x16', '10pct': 'gemv_lane_128+tile_4x16', '20pct': 'gemv_lane_128+tile_8x16'}. **LOO flips (10%):** none. **Threshold-robust:** False.

| threshold | set size | best primitive set | worst | macro | micro | max regret |
|---|---|---|---|---|---|---|
| 5% | 1 | tile_4x16 | 0.158 | 0.862 | 0.992 | 0.842 |
| 5% | 2 | gemv_lane_64+tile_4x16 | 0.998 | 1.000 | 1.000 | 0.002 |
| 5% | 3 | gemv_lane_128+gemv_lane_64+tile_4x16 | 0.998 | 1.000 | 1.000 | 0.002 |
| 5% | 4 | gemv_lane_128+gemv_lane_256+gemv_lane_64+tile_4x16 | 0.998 | 1.000 | 1.000 | 0.002 |
| 10% | 1 | tile_4x16 | 0.763 | 0.949 | 0.999 | 0.237 |
| 10% | 2 | gemv_lane_128+tile_4x16 | 0.998 | 1.000 | 1.000 | 0.002 |
| 10% | 3 | gemv_lane_128+gemv_lane_256+tile_4x16 | 0.998 | 1.000 | 1.000 | 0.002 |
| 10% | 4 | gemv_lane_128+gemv_lane_256+gemv_lane_64+tile_4x16 | 0.998 | 1.000 | 1.000 | 0.002 |
| 20% | 1 | tile_8x16 | 0.763 | 0.949 | 0.999 | 0.237 |
| 20% | 2 | gemv_lane_128+tile_8x16 | 0.998 | 1.000 | 1.000 | 0.002 |
| 20% | 3 | gemv_lane_128+gemv_lane_256+tile_8x16 | 0.998 | 1.000 | 1.000 | 0.002 |
| 20% | 4 | gemv_lane_128+gemv_lane_256+gemv_lane_64+tile_8x16 | 0.998 | 1.000 | 1.000 | 0.002 |

Uncovered ops as the set grows (10% threshold): `uncovered_ops_by_primitive_set.csv` (143 rows).

