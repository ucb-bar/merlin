# Intra-op sharding report

> How each matmul can be split across 2/4/8 candidate units along M (rows), N (columns), or K (reduction). M/N sharding is reduction-free (broadcast + concat); K sharding needs partial sums + an accumulator merge. **Structural geometry only — no speedup, no cycle claim.**

## Clean 8-way shardability (no tail) by axis

| workload | M (rows) | N (cols) | K (reduction) |
|---|---|---|---|
| rdt | 3 | 21 | 21 |
| openvla | 0 | 30 | 30 |
| tiny_llama | 14 | 30 | 30 |
| rdt2 | 3 | 25 | 25 |
| groot_n1d7 | 16 | 116 | 116 |
| molmoact | 16 | 34 | 34 |
| smolvla | 73 | 302 | 302 |
| pi05 | 610 | 777 | 777 |
| xr0 | 10 | 19 | 19 |
| bitvla | 14 | 30 | 30 |

## Findings

- **Reduction-free sharding dominates:** 2631 (op,axis) M/N opportunities split without any cross-shard reduction — only `weight_broadcast`/`activation_multicast` + `output_partition_commit`.
- **K-sharding is the high-communication mode:** 1385 (op,axis) opportunities would need a `partial_sum_object` + `accumulator_merge`; the partial-sum bytes are in `sharding_table.csv`.
- **Attention / conv sharding:** `unavailable` — that structure is lowered into the matmul projections and is not invented.

**Caveat (structural, not realized):** these are sharding *geometries* and their byte costs. They are **not a speedup**, latency, or throughput claim, and assume no hardware.
