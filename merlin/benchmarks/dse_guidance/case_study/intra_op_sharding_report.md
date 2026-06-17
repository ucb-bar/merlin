# Intra-op sharding report

> How each matmul can be split across 2/4/8 candidate units along M (rows), N (columns), or K (reduction). M/N sharding is reduction-free (broadcast + concat); K sharding needs partial sums + an accumulator merge. **Structural geometry only — no speedup, no cycle claim.**

## Clean 8-way shardability (no tail) by axis

| workload | M (rows) | N (cols) | K (reduction) |
|---|---|---|---|
| rdt | 2 | 20 | 20 |
| openvla | 3 | 26 | 26 |
| small_llama | 15 | 15 | 15 |
| tiny_llama | 0 | 15 | 15 |

## Findings

- **Reduction-free sharding dominates:** 133 (op,axis) M/N opportunities split without any cross-shard reduction — only `weight_broadcast`/`activation_multicast` + `output_partition_commit`.
- **K-sharding is the high-communication mode:** 76 (op,axis) opportunities would need a `partial_sum_object` + `accumulator_merge`; the partial-sum bytes are in `sharding_table.csv`.
- **Attention / conv sharding:** `unavailable` — that structure is lowered into the matmul projections and is not invented.

**Caveat (structural, not realized):** these are sharding *geometries* and their byte costs. They are **not a speedup**, latency, or throughput claim, and assume no hardware.
