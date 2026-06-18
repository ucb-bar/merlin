# Candidate-primitive coverage report

> For each candidate compute-primitive shape: how much of the real operator MAC mass it covers under a padding-waste band, and its MAC-weighted tile utilisation. **Structural geometry coverage only — no speedup, no cycle-count, no performance ranking.** Tile primitives keep K exact and model M/N tile tails; GEMV lanes apply only to vector-like shapes (see `primitive_coverage.py`).

## Primitive coverage (MAC-weighted, all workloads)

| primitive | kind | MAC util | covered ≤5% | covered ≤10% | covered ≤25% |
|---|---|---|---|---|---|
| tile_8x16 | tile | 96.3% | 83.2% | 92.5% | 98.6% |
| tile_8x8 | tile | 96.3% | 83.2% | 92.5% | 98.6% |
| tile_16x16 | tile | 81.1% | 68.4% | 68.4% | 83.8% |
| tile_16x32 | tile | 81.1% | 68.4% | 68.4% | 83.8% |
| tile_32x32 | tile | 61.1% | 68.3% | 68.3% | 70.1% |
| gemv_lane_64 | gemv_lane | 100.0% | 32.9% | 32.9% | 32.9% |
| gemv_lane_128 | gemv_lane | 100.0% | 32.9% | 32.9% | 32.9% |
| gemv_lane_256 | gemv_lane | 99.9% | 32.9% | 32.9% | 32.9% |

## Per-workload coverage under 10% padding waste

| primitive | groot_n1d7 | molmoact | openvla | rdt | rdt2 | small_llama | tiny_llama |
|---|---|---|---|---|---|---|---|
| tile_8x16 | 15% | 100% | 33% | 100% | 0% | 100% | 0% |
| tile_8x8 | 15% | 100% | 33% | 100% | 0% | 100% | 0% |
| tile_16x16 | 15% | 0% | 33% | 88% | 0% | 0% | 0% |
| tile_16x32 | 15% | 0% | 33% | 88% | 0% | 0% | 0% |
| tile_32x32 | 15% | 0% | 0% | 88% | 0% | 0% | 0% |
| gemv_lane_64 | 100% | 100% | 100% | 13% | 100% | 59% | 100% |
| gemv_lane_128 | 100% | 100% | 94% | 13% | 100% | 59% | 100% |
| gemv_lane_256 | 100% | 100% | 57% | 13% | 100% | 8% | 100% |

Covers X% of MACs ≈ a primitive of that shape would process X% of the real MAC mass with ≤ the stated padding waste. A low cell means that primitive **poorly covers** that workload's shapes and is workload-specific. See `primitive_coverage_matrix.csv` and `primitive_regret_table.csv`.
