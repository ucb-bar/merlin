# Inter-op DAG parallelism report

> Structural concurrency of the operator dependency DAG (edges recovered from the SSA use-def graph). `available_parallelism = total_work / critical_path_work` is the work/span ratio (average parallelism) — a structural property, **not a speedup**, no hardware assumed.

| workload | ops | total MACs | critical-path MACs | available parallelism | max ready width | independent components | structure |
|---|---|---|---|---|---|---|---|
| rdt | 21 | 39,466,041,344 | 35,501,375,488 | 1.1117× | 5 | 1 | mostly_sequential |
| openvla | 30 | 15,269,888 | 7,929,856 | 1.9256× | 6 | 2 | some_parallelism |
| tiny_llama | 30 | 923,795,456 | 568,852,480 | 1.624× | 6 | 2 | some_parallelism |
| rdt2 | 26 | 991,854,592 | 728,399,872 | 1.3617× | 4 | 1 | mostly_sequential |
| groot_n1d7 | 116 | 20,393,361,408 | 15,646,064,640 | 1.3034× | 17 | 1 | mostly_sequential |
| molmoact | 34 | 8,419,016,704 | 7,472,152,576 | 1.1267× | 2 | 2 | mostly_sequential |
| smolvla | 302 | 110,595,843,584 | 84,340,899,840 | 1.3113× | 13 | 1 | mostly_sequential |
| pi05 | 777 | 2,146,035,695,616 | 1,330,911,969,280 | 1.6125× | 40 | 1 | some_parallelism |
| xr0 | 19 | 1,115,879,424 | 838,760,448 | 1.3304× | 3 | 1 | mostly_sequential |
| bitvla | 30 | 39,452,672 | 25,427,968 | 1.5515× | 10 | 2 | some_parallelism |

## Findings

- **Low inter-op parallelism (rdt, rdt2, groot_n1d7, molmoact, smolvla, xr0):** the dependency DAG is a deep near-sequential chain (available parallelism < 1.5×). A future DSE tool should look to **intra-op sharding** of the large GEMMs (see `sharding_table.csv`), not inter-op concurrency.
- **Some inter-op parallelism (openvla, tiny_llama, pi05, bitvla):** independent operators (e.g. Q/K/V projections) become ready together — modest concurrency a multi-engine cluster could use.
- **Ready-set width** peaks at a handful of operators (see `concurrency_windows.csv`) — the workloads do not expose wide inter-op concurrency.

**Caveat (structural, not realized):** available parallelism is a work/span ratio of the dependency DAG. It is **not a speedup**, **not a cycle count**, and assumes no hardware, no scheduling, and no communication cost.
