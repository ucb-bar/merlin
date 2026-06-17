# Inter-op DAG parallelism report

> Structural concurrency of the operator dependency DAG (edges recovered from the SSA use-def graph). `available_parallelism = total_work / critical_path_work` is the work/span ratio (average parallelism) — a structural property, **not a speedup**, no hardware assumed.

| workload | ops | total MACs | critical-path MACs | available parallelism | max ready width | independent components | structure |
|---|---|---|---|---|---|---|---|
| rdt | 20 | 39,432,486,912 | 35,501,375,488 | 1.1107× | 4 | 1 | mostly_sequential |
| openvla | 26 | 79,544,320 | 65,470,464 | 1.215× | 3 | 1 | mostly_sequential |
| small_llama | 15 | 3,424,256 | 2,195,456 | 1.5597× | 3 | 1 | some_parallelism |
| tiny_llama | 15 | 614,465,536 | 513,802,240 | 1.1959× | 3 | 1 | mostly_sequential |

## Findings

- **Low inter-op parallelism (rdt, openvla, tiny_llama):** the dependency DAG is a deep near-sequential chain (available parallelism < 1.5×). A future DSE tool should look to **intra-op sharding** of the large GEMMs (see `sharding_table.csv`), not inter-op concurrency.
- **Some inter-op parallelism (small_llama):** independent operators (e.g. Q/K/V projections) become ready together — modest concurrency a multi-engine cluster could use.
- **Ready-set width** peaks at a handful of operators (see `concurrency_windows.csv`) — the workloads do not expose wide inter-op concurrency.

**Caveat (structural, not realized):** available parallelism is a work/span ratio of the dependency DAG. It is **not a speedup**, **not a cycle count**, and assumes no hardware, no scheduling, and no communication cost.
