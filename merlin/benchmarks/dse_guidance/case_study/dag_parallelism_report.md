# Inter-op DAG parallelism report

> Structural concurrency of the operator dependency DAG (edges recovered from the SSA use-def graph). `available_parallelism = total_work / critical_path_work` is the work/span ratio (average parallelism) — a structural property, **not a speedup**, no hardware assumed.

| workload | ops | total MACs | critical-path MACs | available parallelism | max ready width | independent components | structure |
|---|---|---|---|---|---|---|---|
| rdt | 20 | 39,432,486,912 | 35,501,375,488 | 1.1107× | 4 | 1 | mostly_sequential |
| openvla | 26 | 79,544,320 | 65,470,464 | 1.215× | 3 | 1 | mostly_sequential |
| small_llama | 15 | 3,424,256 | 2,195,456 | 1.5597× | 3 | 1 | some_parallelism |
| tiny_llama | 15 | 614,465,536 | 513,802,240 | 1.1959× | 3 | 1 | mostly_sequential |
| rdt2 | 23 | 941,031,424 | 697,761,792 | 1.3486× | 3 | 1 | mostly_sequential |
| groot_n1d7 | 18 | 2,612,133,888 | 2,103,705,600 | 1.2417× | 3 | 1 | mostly_sequential |
| molmoact | 17 | 7,574,913,024 | 7,574,913,024 | 1.0× | 1 | 1 | mostly_sequential |
| smolvla | 106 | 90,656,617,984 | 74,759,946,240 | 1.2126× | 5 | 1 | mostly_sequential |
| pi05 | 777 | 2,146,035,695,616 | 1,330,911,969,280 | 1.6125× | 40 | 1 | some_parallelism |

## Findings

- **Low inter-op parallelism (rdt, openvla, tiny_llama, rdt2, groot_n1d7, molmoact, smolvla):** the dependency DAG is a deep near-sequential chain (available parallelism < 1.5×). A future DSE tool should look to **intra-op sharding** of the large GEMMs (see `sharding_table.csv`), not inter-op concurrency.
- **Some inter-op parallelism (small_llama, pi05):** independent operators (e.g. Q/K/V projections) become ready together — modest concurrency a multi-engine cluster could use.
- **Ready-set width** peaks at a handful of operators (see `concurrency_windows.csv`) — the workloads do not expose wide inter-op concurrency.

**Caveat (structural, not realized):** available parallelism is a work/span ratio of the dependency DAG. It is **not a speedup**, **not a cycle count**, and assumes no hardware, no scheduling, and no communication cost.
