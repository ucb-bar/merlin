# Operator-geometry report

> The first search-space-formation layer: what matmul-like operator shapes actually appear across the recaptured workloads, classified by deterministic geometry rules and (orthogonally) by semantic role from `prov.fqn`. **Structural geometry only — no speedup, no cycle-count, no performance claim.**

Operators extracted: **76** across **4** workloads.

## Per-workload operator counts and dominant geometry

| workload | operators | dominant shape_class (by MACs) | tail-heavy | small fragments |
|---|---|---|---|---|
| rdt | 20 | squareish_gemm | 18 | 0 |
| openvla | 26 | wide_skinny | 26 | 0 |
| small_llama | 15 | wide_skinny | 15 | 0 |
| tiny_llama | 15 | gemv_like | 15 | 0 |

## Top shape classes by MAC count (geometry)

| shape_class | ops | MACs | MAC share |
|---|---|---|---|
| squareish_gemm | 2 | 34,377,302,016 | 85.7% |
| wide_skinny | 55 | 5,128,716,288 | 12.8% |
| gemv_like | 19 | 623,902,720 | 1.6% |

## Top shape classes by op count (geometry)

| shape_class | ops |
|---|---|
| wide_skinny | 55 |
| gemv_like | 19 |
| squareish_gemm | 2 |

## Semantic roles (from prov.fqn)

| semantic_class | ops |
|---|---|
| mlp_projection | 35 |
| attention_qkv_projection | 26 |
| attention_output_projection | 12 |
| lm_head_projection | 3 |

## Shape-irregularity findings

- **Tail-heavy operators:** 74 op(s) waste >10% against a 32×32 tile (e.g. `rdt` op 0 1×2048, 3100% waste).

## Not recovered (honest)

- **Attention structure** (heads / head_dim / kv_len / mask): `unavailable` — attention is lowered into the matmul projections; only Q/K/V/O projection matmuls are visible.
- **Conv structure**: no `linalg.conv*` ops present in the current captures.
- **Batch dims**: `linalg.matmul` is 2-D here; `batch_product = 1`.

See `operator_shape_table.csv` for every operator and `operator_cluster_table.csv` for the cross-workload clusters. Thresholds are documented in `shape_taxonomy.py`.
