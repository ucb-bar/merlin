# Operator-geometry report

> The first search-space-formation layer: what matmul-like operator shapes actually appear across the recaptured workloads, classified by deterministic geometry rules and (orthogonally) by semantic role from `prov.fqn`. **Structural geometry only — no speedup, no cycle-count, no performance claim.**

Operators extracted: **1385** across **10** workloads.

## Per-workload operator counts and dominant geometry

| workload | operators | dominant shape_class (by MACs) | tail-heavy | small fragments |
|---|---|---|---|---|
| rdt | 21 | squareish_gemm | 18 | 0 |
| openvla | 30 | wide_skinny | 30 | 6 |
| tiny_llama | 30 | wide_skinny | 30 | 0 |
| rdt2 | 26 | wide_skinny | 26 | 0 |
| groot_n1d7 | 116 | wide_skinny | 100 | 0 |
| molmoact | 34 | wide_skinny | 34 | 0 |
| smolvla | 302 | squareish_gemm | 229 | 1 |
| pi05 | 777 | wide_skinny | 167 | 0 |
| xr0 | 19 | wide_skinny | 5 | 2 |
| bitvla | 30 | wide_skinny | 16 | 4 |

## Top shape classes by MAC count (geometry)

| shape_class | ops | MACs | MAC share |
|---|---|---|---|
| wide_skinny | 1032 | 1,495,412,578,816 | 64.2% |
| projection_like | 138 | 709,784,336,384 | 30.5% |
| squareish_gemm | 64 | 92,362,150,912 | 4.0% |
| unknown | 12 | 28,991,029,248 | 1.2% |
| gemv_like | 139 | 1,446,115,328 | 0.1% |

## Top shape classes by op count (geometry)

| shape_class | ops |
|---|---|
| wide_skinny | 1032 |
| gemv_like | 139 |
| projection_like | 138 |
| squareish_gemm | 64 |
| unknown | 12 |

## Semantic roles (from prov.fqn)

| semantic_class | ops |
|---|---|
| attention_qkv_projection | 446 |
| unknown | 410 |
| mlp_projection | 374 |
| attention_output_projection | 149 |
| lm_head_projection | 4 |
| embedding_projection | 2 |

## Shape-irregularity findings

- **Tail-heavy operators:** 655 op(s) waste >10% against a 32×32 tile (e.g. `rdt` op 1 1×2048, 3100% waste).
- **Small dispatch fragments:** 13 op(s) below 65,536 MACs (e.g. `openvla` op 19, 32,768 MACs) — dispatch-bound, not compute-bound.

## Not recovered (honest)

- **Attention structure** (heads / head_dim / kv_len / mask): `unavailable` — attention is lowered into the matmul projections; only Q/K/V/O projection matmuls are visible.
- **Conv structure**: no `linalg.conv*` ops present in the current captures.
- **Batch dims**: `linalg.matmul` is 2-D here; `batch_product = 1`.

See `operator_shape_table.csv` for every operator and `operator_cluster_table.csv` for the cross-workload clusters. Thresholds are documented in `shape_taxonomy.py`.
