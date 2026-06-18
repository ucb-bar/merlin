# Operator-geometry report

> The first search-space-formation layer: what matmul-like operator shapes actually appear across the recaptured workloads, classified by deterministic geometry rules and (orthogonally) by semantic role from `prov.fqn`. **Structural geometry only — no speedup, no cycle-count, no performance claim.**

Operators extracted: **1051** across **11** workloads.

## Per-workload operator counts and dominant geometry

| workload | operators | dominant shape_class (by MACs) | tail-heavy | small fragments |
|---|---|---|---|---|
| rdt | 20 | squareish_gemm | 18 | 0 |
| openvla | 26 | wide_skinny | 26 | 0 |
| small_llama | 15 | wide_skinny | 15 | 0 |
| tiny_llama | 15 | gemv_like | 15 | 0 |
| rdt2 | 23 | wide_skinny | 23 | 0 |
| groot_n1d7 | 18 | wide_skinny | 16 | 0 |
| molmoact | 17 | wide_skinny | 17 | 0 |
| smolvla | 106 | squareish_gemm | 33 | 1 |
| pi05 | 777 | wide_skinny | 167 | 0 |
| xr0 | 19 | wide_skinny | 5 | 2 |
| bitvla | 15 | wide_skinny | 0 | 0 |

## Top shape classes by MAC count (geometry)

| shape_class | ops | MACs | MAC share |
|---|---|---|---|
| wide_skinny | 805 | 1,458,297,522,688 | 63.7% |
| projection_like | 96 | 708,650,358,784 | 31.0% |
| squareish_gemm | 64 | 92,362,150,912 | 4.0% |
| unknown | 12 | 28,991,029,248 | 1.3% |
| gemv_like | 74 | 811,268,096 | 0.0% |

## Top shape classes by op count (geometry)

| shape_class | ops |
|---|---|
| wide_skinny | 805 |
| projection_like | 96 |
| gemv_like | 74 |
| squareish_gemm | 64 |
| unknown | 12 |

## Semantic roles (from prov.fqn)

| semantic_class | ops |
|---|---|
| attention_qkv_projection | 432 |
| mlp_projection | 382 |
| attention_output_projection | 154 |
| unknown | 76 |
| lm_head_projection | 5 |
| embedding_projection | 2 |

## Shape-irregularity findings

- **Tail-heavy operators:** 335 op(s) waste >10% against a 32×32 tile (e.g. `rdt` op 0 1×2048, 3100% waste).
- **Small dispatch fragments:** 3 op(s) below 65,536 MACs (e.g. `smolvla` op 73, 30,720 MACs) — dispatch-bound, not compute-bound.

## Not recovered (honest)

- **Attention structure** (heads / head_dim / kv_len / mask): `unavailable` — attention is lowered into the matmul projections; only Q/K/V/O projection matmuls are visible.
- **Conv structure**: no `linalg.conv*` ops present in the current captures.
- **Batch dims**: `linalg.matmul` is 2-D here; `batch_product = 1`.

See `operator_shape_table.csv` for every operator and `operator_cluster_table.csv` for the cross-workload clusters. Thresholds are documented in `shape_taxonomy.py`.
