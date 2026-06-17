# Workload contract graph — summary

> The multi-rate workload contract graph: phases (cadence), regions (real IR facts), operators (P5 geometry), and state objects (lifetimes), with typed edges. Operator-level data dependencies are conservative (`unknown_dependency`) — a flat capture does not carry the true dependency graph. **No speedup / cycle / area claim.**

## Graph size per workload

| workload | class | nodes | edges | phase | region | operator | loop_body | state | known edges | unknown edges |
|---|---|---|---|---|---|---|---|---|---|---|
| rdt | flow_matching_action_head | 25 | 22 | 2 | 1 | 20 | 1 | 1 | 2 | 20 |
| openvla | autoregressive_decode | 33 | 26 | 2 | 3 | 26 | 1 | 1 | 2 | 24 |
| small_llama | autoregressive_decode | 21 | 16 | 2 | 2 | 15 | 1 | 1 | 2 | 14 |
| tiny_llama | autoregressive_decode | 20 | 17 | 2 | 1 | 15 | 1 | 1 | 2 | 15 |

## Repeated structure (which workloads have a K-loop / token-loop)

| workload | repeated head | cadence | trip count (K) | loop-invariant state |
|---|---|---|---|---|
| rdt | yes | K_times_per_replan | 5 | weights |
| openvla | yes | token_loop | 7 | weights |
| small_llama | yes | token_loop | 32 | weights |
| tiny_llama | yes | token_loop | 32 | weights |

## What downstream DSE phases can now consume

- **Phase/rate scheduling:** per-phase cadence + the rate model (K/H/control rate, derived replan deadline) — enough to reason about the once-vs-K-vs-control rate split.
- **Residency:** loop-invariant state edges (weights) carry byte size + reuse count.
- **Partition:** region nodes carry per-region MACs/bytes and the recovered backbone/head split.
- **Primitive sizing:** operator nodes link to the P5 shape classes / coverage table.

**Missing before this graph drives scheduling/DSE:** true operator data dependencies (conservative-sequential today), per-K-step timing, host command/sync latency, and the specific boundary-crossing tensors where the backbone produces none in the flat capture. These are `unavailable`/`unknown` in the graph, not invented.
