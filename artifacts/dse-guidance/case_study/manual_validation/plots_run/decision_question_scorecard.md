# DSE decision-question scorecard (all)

> The few decisions a future DSE tool must make, each answered from the workload analysis with its caveat. A metric earns its place only by answering one of these.

| # | decision question | answer (from analysis) | caveat |
|---|---|---|---|
| Q1 | Q1 best single primitive (worst-workload coverage)? | gemv_lane_64 -> worst 0.13, macro 0.80 | no single primitive covers every workload |
| Q2 | Q2 best 2-primitive set? | gemv_lane_64+tile_8x16 -> worst 1.00 (vs 0.13 single) | search primitive SETS, not one tile |
| Q3 | Q3 capacity x dtype residency thresholds? | see decision_capacity_dtype plot (int4<int8<bf16 budget to fit) | repeated-head weights only; K is configured/reference |
| Q4 | Q4 sharding axis for top-MAC ops? | M/N reduction-free vs K partial-sum (see decision_sharding_per_top_op) | communication bytes, not latency |
| Q5 | Q5 which abstractions are NECESSARY (not just possible)? | 4 necessary, 5 useful, 11 possible, 7 blocked, 0 N/A | strict predicate; low-bit abstractions blocked by capture |
| Q6 | Q6 which conclusions are driven by one workload (RDT)? | dense-MAC dominance macro 0.1395 vs micro 0.0397; collapses if removed: none | micro view is biased by RDT's 87%-of-workload op |
| Q7 | Q7 which claims depend on configured K (capture fidelity)? | all residency / loop / command claims (K is config/reference) | needs a loop-preserving capture; see capture_fidelity_matrix |
