# Workload contract graph — summary

> The multi-rate workload contract graph: phases (cadence), regions (real IR facts), operators (P5 geometry), and state objects (lifetimes), with typed edges. Operator data dependencies are recovered from the SSA use-def graph (`data_dependency`, `recovered_from_ir`). **No speedup / cycle / area claim.**

## Graph size per workload

| workload | class | nodes | edges | phase | region | operator | loop_body | state | data-dep edges | edges w/ recovered evidence |
|---|---|---|---|---|---|---|---|---|---|---|
| rdt | flow_matching_action_head | 25 | 49 | 2 | 1 | 20 | 1 | 1 | 46 | 49 |
| openvla | autoregressive_decode | 32 | 54 | 2 | 2 | 26 | 1 | 1 | 51 | 54 |
| small_llama | autoregressive_decode | 20 | 31 | 2 | 1 | 15 | 1 | 1 | 28 | 31 |
| tiny_llama | autoregressive_decode | 20 | 31 | 2 | 1 | 15 | 1 | 1 | 28 | 31 |
| rdt2 | flow_matching_action_head | 28 | 67 | 2 | 1 | 23 | 1 | 1 | 64 | 67 |
| groot_n1d7 | flow_matching_action_head | 24 | 34 | 2 | 2 | 18 | 1 | 1 | 31 | 34 |
| molmoact | autoregressive_decode | 22 | 47 | 2 | 1 | 17 | 1 | 1 | 44 | 47 |
| smolvla | flow_matching_action_head | 112 | 698 | 2 | 2 | 106 | 1 | 1 | 695 | 698 |
| pi05 | flow_matching_action_head | 783 | 14352 | 2 | 2 | 777 | 1 | 1 | 14349 | 14352 |
| xr0 | flow_matching_action_head | 26 | 49 | 2 | 3 | 19 | 1 | 1 | 46 | 49 |
| bitvla | autoregressive_decode | 20 | 31 | 2 | 1 | 15 | 1 | 1 | 28 | 31 |

## Repeated structure (which workloads have a K-loop / token-loop)

| workload | repeated head | cadence | trip count (K) | loop-invariant state |
|---|---|---|---|---|
| rdt | yes | K_times_per_replan | 5 | weights |
| openvla | yes | token_loop | 7 | weights |
| small_llama | yes | token_loop | 32 | weights |
| tiny_llama | yes | token_loop | 32 | weights |
| rdt2 | yes | K_times_per_replan | 5 | weights |
| groot_n1d7 | yes | K_times_per_replan | 4 | weights |
| molmoact | yes | token_loop | 8 | weights |
| smolvla | yes | K_times_per_replan | 10 | weights |
| pi05 | yes | K_times_per_replan | 10 | weights |
| xr0 | yes | K_times_per_replan | 10 | weights |
| bitvla | yes | token_loop | 7 | weights |

## What downstream DSE phases can now consume

- **Phase/rate scheduling:** per-phase cadence + the rate model (K/H/control rate, derived replan deadline) — enough to reason about the once-vs-K-vs-control rate split.
- **Residency:** loop-invariant state edges (weights) carry byte size + reuse count.
- **Partition:** region nodes carry per-region MACs/bytes and the recovered backbone/head split.
- **Primitive sizing:** operator nodes link to the P5 shape classes / coverage table.
- **Scheduling/overlap:** real `data_dependency` edges (from the SSA use-def graph) give the true intra-phase ordering, and the cross-replan `pipeline_candidate` edge marks the backbone/head overlap the rate split permits.

Every node and edge carries a provenance label: structure and data dependencies are `recovered_from_ir`, roles/cadence `recovered_from_prov_fqn`, the rate constants `recovered_from_model_config`, the replan deadline `derived_requirement`.
