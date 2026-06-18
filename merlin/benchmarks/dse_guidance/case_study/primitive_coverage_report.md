# Candidate-primitive coverage report

> For each candidate compute-primitive shape: how much of the real operator MAC mass it covers under a padding-waste band, and its MAC-weighted tile utilisation. **Structural geometry coverage only — no speedup, no cycle-count, no performance ranking.** Tile primitives keep K exact and model M/N tile tails; GEMV lanes apply only to vector-like shapes (see `primitive_coverage.py`).

## Primitive coverage (MAC-weighted, all workloads)

| primitive | kind | MAC util | covered ≤5% | covered ≤10% | covered ≤25% |
|---|---|---|---|---|---|
| tile_8x16 | tile | 99.8% | 98.8% | 99.1% | 100.0% |
| tile_8x8 | tile | 99.8% | 98.8% | 99.1% | 100.0% |
| tile_16x16 | tile | 98.5% | 98.5% | 98.5% | 98.9% |
| tile_16x32 | tile | 98.5% | 98.5% | 98.5% | 98.9% |
| tile_32x32 | tile | 96.3% | 98.5% | 98.5% | 98.6% |
| gemv_lane_64 | gemv_lane | 99.9% | 63.7% | 63.7% | 63.7% |
| gemv_lane_128 | gemv_lane | 99.9% | 63.6% | 63.7% | 63.7% |
| gemv_lane_256 | gemv_lane | 98.3% | 54.3% | 54.4% | 63.7% |

## Per-workload coverage under 10% padding waste

| primitive | bitvla | groot_n1d7 | molmoact | openvla | pi05 | rdt | rdt2 | small_llama | smolvla | tiny_llama | xr0 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| tile_8x16 | 100% | 15% | 100% | 33% | 99% | 100% | 0% | 100% | 99% | 0% | 99% |
| tile_8x8 | 100% | 15% | 100% | 33% | 99% | 100% | 0% | 100% | 99% | 0% | 99% |
| tile_16x16 | 100% | 15% | 0% | 33% | 99% | 88% | 0% | 0% | 97% | 0% | 99% |
| tile_16x32 | 100% | 15% | 0% | 33% | 99% | 88% | 0% | 0% | 97% | 0% | 99% |
| tile_32x32 | 100% | 15% | 0% | 0% | 99% | 88% | 0% | 0% | 97% | 0% | 99% |
| gemv_lane_64 | 100% | 100% | 100% | 100% | 67% | 13% | 100% | 59% | 4% | 100% | 100% |
| gemv_lane_128 | 100% | 100% | 100% | 94% | 67% | 13% | 100% | 59% | 4% | 100% | 100% |
| gemv_lane_256 | 91% | 100% | 100% | 57% | 57% | 13% | 100% | 8% | 4% | 100% | 100% |

Covers X% of MACs ≈ a primitive of that shape would process X% of the real MAC mass with ≤ the stated padding waste. A low cell means that primitive **poorly covers** that workload's shapes and is workload-specific. See `primitive_coverage_matrix.csv` and `primitive_regret_table.csv`.
