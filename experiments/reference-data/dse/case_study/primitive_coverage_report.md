# Candidate-primitive coverage report

> For each candidate compute-primitive shape: how much of the real operator MAC mass it covers under a padding-waste band, and its MAC-weighted tile utilisation. **Structural geometry coverage only — no speedup, no cycle-count, no performance ranking.** Tile primitives keep K exact and model M/N tile tails; GEMV lanes apply only to vector-like shapes (see `primitive_coverage.py`).

## Primitive coverage (MAC-weighted, all workloads)

| primitive | kind | MAC util | covered ≤5% | covered ≤10% | covered ≤25% |
|---|---|---|---|---|---|
| tile_8x16 | tile | 99.3% | 97.3% | 98.3% | 99.9% |
| tile_8x8 | tile | 99.3% | 97.3% | 98.3% | 99.9% |
| tile_16x16 | tile | 97.6% | 97.0% | 97.0% | 98.7% |
| tile_16x32 | tile | 97.6% | 97.0% | 97.0% | 98.7% |
| tile_32x32 | tile | 94.5% | 97.0% | 97.0% | 97.8% |
| gemv_lane_64 | gemv_lane | 99.9% | 64.2% | 64.3% | 64.3% |
| gemv_lane_128 | gemv_lane | 99.9% | 63.8% | 64.3% | 64.3% |
| gemv_lane_256 | gemv_lane | 98.3% | 54.7% | 55.1% | 64.3% |

## Per-workload coverage under 10% padding waste

| primitive | bitvla | groot_n1d7 | molmoact | openvla | pi05 | rdt | rdt2 | smolvla | tiny_llama | xr0 |
|---|---|---|---|---|---|---|---|---|---|---|
| tile_8x16 | 96% | 16% | 89% | 0% | 99% | 100% | 5% | 96% | 76% | 99% |
| tile_8x8 | 96% | 16% | 89% | 0% | 99% | 100% | 5% | 96% | 76% | 99% |
| tile_16x16 | 96% | 16% | 0% | 0% | 99% | 88% | 0% | 79% | 0% | 99% |
| tile_16x32 | 96% | 16% | 0% | 0% | 99% | 88% | 0% | 79% | 0% | 99% |
| tile_32x32 | 96% | 16% | 0% | 0% | 99% | 88% | 0% | 79% | 0% | 99% |
| gemv_lane_64 | 100% | 100% | 100% | 100% | 67% | 13% | 100% | 20% | 100% | 100% |
| gemv_lane_128 | 100% | 100% | 100% | 100% | 67% | 13% | 100% | 20% | 100% | 100% |
| gemv_lane_256 | 89% | 100% | 100% | 73% | 57% | 13% | 100% | 20% | 100% | 100% |

Covers X% of MACs ≈ a primitive of that shape would process X% of the real MAC mass with ≤ the stated padding waste. A low cell means that primitive **poorly covers** that workload's shapes and is workload-specific. See `primitive_coverage_matrix.csv` and `primitive_regret_table.csv`.
