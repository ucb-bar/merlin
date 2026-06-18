# Operator-geometry report

> The first search-space-formation layer: what matmul-like operator shapes actually appear across the recaptured workloads, classified by deterministic geometry rules and (orthogonally) by semantic role from `prov.fqn`. **Structural geometry only — no speedup, no cycle-count, no performance claim.**

Operators extracted: **134** across **7** workloads.

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

## Top shape classes by MAC count (geometry)

| shape_class | ops | MACs | MAC share |
|---|---|---|---|
| squareish_gemm | 2 | 34,377,302,016 | 67.1% |
| wide_skinny | 102 | 16,194,338,816 | 31.6% |
| gemv_like | 29 | 684,064,768 | 1.3% |
| projection_like | 1 | 2,293,760 | 0.0% |

## Top shape classes by op count (geometry)

| shape_class | ops |
|---|---|
| wide_skinny | 102 |
| gemv_like | 29 |
| squareish_gemm | 2 |
| projection_like | 1 |

## Semantic roles (from prov.fqn)

| semantic_class | ops |
|---|---|
| mlp_projection | 53 |
| unknown | 33 |
| attention_qkv_projection | 26 |
| attention_output_projection | 16 |
| lm_head_projection | 4 |
| embedding_projection | 2 |

## Shape-irregularity findings

- **Tail-heavy operators:** 130 op(s) waste >10% against a 32×32 tile (e.g. `rdt` op 0 1×2048, 3100% waste).

## Not recovered (honest)

- **Attention structure** (heads / head_dim / kv_len / mask): `unavailable` — attention is lowered into the matmul projections; only Q/K/V/O projection matmuls are visible.
- **Conv structure**: no `linalg.conv*` ops present in the current captures.
- **Batch dims**: `linalg.matmul` is 2-D here; `batch_product = 1`.

See `operator_shape_table.csv` for every operator and `operator_cluster_table.csv` for the cross-workload clusters. Thresholds are documented in `shape_taxonomy.py`.
