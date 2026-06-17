# Candidate-primitive coverage report

> For each candidate compute-primitive shape: how much of the real operator MAC mass it covers under a padding-waste band, and its MAC-weighted tile utilisation. **Structural geometry coverage only — no speedup, no cycle-count, no performance ranking.** Tile primitives keep K exact and model M/N tile tails; GEMV lanes apply only to vector-like shapes (see `primitive_coverage.py`).

## Primitive coverage (MAC-weighted, all workloads)

| primitive | kind | MAC util | covered ≤5% | covered ≤10% | covered ≤25% |
|---|---|---|---|---|---|
| tile_8x16 | tile | 97.4% | 86.4% | 98.3% | 98.4% |
| tile_8x8 | tile | 97.4% | 86.4% | 98.3% | 98.4% |
| tile_16x16 | tile | 93.1% | 86.4% | 86.4% | 98.3% |
| tile_16x32 | tile | 93.1% | 86.4% | 86.4% | 98.3% |
| tile_32x32 | tile | 85.6% | 86.3% | 86.3% | 86.3% |
| gemv_lane_64 | gemv_lane | 100.0% | 14.3% | 14.3% | 14.3% |
| gemv_lane_128 | gemv_lane | 100.0% | 14.3% | 14.3% | 14.3% |
| gemv_lane_256 | gemv_lane | 99.7% | 14.2% | 14.2% | 14.3% |

## Per-workload coverage under 10% padding waste

| primitive | openvla | rdt | small_llama | tiny_llama |
|---|---|---|---|---|
| tile_8x16 | 33% | 100% | 100% | 0% |
| tile_8x8 | 33% | 100% | 100% | 0% |
| tile_16x16 | 33% | 88% | 0% | 0% |
| tile_16x32 | 33% | 88% | 0% | 0% |
| tile_32x32 | 0% | 88% | 0% | 0% |
| gemv_lane_64 | 100% | 13% | 59% | 100% |
| gemv_lane_128 | 94% | 13% | 59% | 100% |
| gemv_lane_256 | 57% | 13% | 8% | 100% |

Covers X% of MACs ≈ a primitive of that shape would process X% of the real MAC mass with ≤ the stated padding waste. A low cell means that primitive **poorly covers** that workload's shapes and is workload-specific. See `primitive_coverage_matrix.csv` and `primitive_regret_table.csv`.
