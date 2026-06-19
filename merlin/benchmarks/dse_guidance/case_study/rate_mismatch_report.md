# Rate-mismatch report

> The multi-rate contract makes the rate mismatches explicit: a backbone that runs once per replan, an action head that runs K times, and a control loop that consumes H actions at a fixed frequency are three different rates the flat capture collapsed into one. **Structural only — no speedup, no cycle budget claimed.**

## Per-workload rate structure

| workload | backbone | repeated head | K | H | control_rate_hz | replan deadline (s) |
|---|---|---|---|---|---|---|
| rdt | once_per_replan | K_times_per_replan | 5 | 64 | 30.0 | 2.133333 |
| openvla | once_per_replan | token_loop | 7 | 7 | 5.0 | 1.4 |
| small_llama | once_per_replan | token_loop | 32 | 32 | 30.0 | 1.066667 |
| tiny_llama | once_per_replan | token_loop | 32 | 32 | 30.0 | 1.066667 |
| rdt2 | once_per_replan | K_times_per_replan | 5 | 64 | 30.0 | 2.133333 |
| groot_n1d7 | once_per_replan | K_times_per_replan | 4 | 16 | 30.0 | 0.533333 |
| molmoact | once_per_replan | token_loop | 8 | 8 | 5.0 | 1.6 |
| smolvla | once_per_replan | K_times_per_replan | 10 | 50 | 30.0 | 1.666667 |
| pi05 | once_per_replan | K_times_per_replan | 10 | 50 | 50.0 | 1.0 |
| xr0 | once_per_replan | K_times_per_replan | 5 | 5 | 30.0 | 0.166667 |
| bitvla | once_per_replan | token_loop | 7 | 7 | 5.0 | 1.4 |

## Provenance of every field

- **`recovered_from_ir`:** region roles' MACs/bytes, operator shapes, loop-invariant weight bytes, and the **operator data-dependency edges** (from the SSA use-def graph).
- **`recovered_from_prov_fqn`:** the region roles and the once-vs-repeated cadence split (backbone once, head repeated).
- **`recovered_from_model_config`:** K, H, control_rate_hz (the model's published architecture constants, from the model registry).
- **`derived_requirement`:** the replan deadline (= H / control_rate) and the cross-replan pipeline-candidate overlap.

## Dependency knowledge (fully recovered)

- **rdt:** 46 `data_dependency` edges recovered from the SSA use-def graph, plus the backbone→head `control_dependency`, the loop-invariant weight edge, and the cross-replan `pipeline_candidate`. Every edge carries a recovered/derived evidence label.
- **openvla:** 51 `data_dependency` edges recovered from the SSA use-def graph, plus the backbone→head `control_dependency`, the loop-invariant weight edge, and the cross-replan `pipeline_candidate`. Every edge carries a recovered/derived evidence label.
- **small_llama:** 28 `data_dependency` edges recovered from the SSA use-def graph, plus the backbone→head `control_dependency`, the loop-invariant weight edge, and the cross-replan `pipeline_candidate`. Every edge carries a recovered/derived evidence label.
- **tiny_llama:** 28 `data_dependency` edges recovered from the SSA use-def graph, plus the backbone→head `control_dependency`, the loop-invariant weight edge, and the cross-replan `pipeline_candidate`. Every edge carries a recovered/derived evidence label.
- **rdt2:** 64 `data_dependency` edges recovered from the SSA use-def graph, plus the backbone→head `control_dependency`, the loop-invariant weight edge, and the cross-replan `pipeline_candidate`. Every edge carries a recovered/derived evidence label.
- **groot_n1d7:** 31 `data_dependency` edges recovered from the SSA use-def graph, plus the backbone→head `control_dependency`, the loop-invariant weight edge, and the cross-replan `pipeline_candidate`. Every edge carries a recovered/derived evidence label.
- **molmoact:** 44 `data_dependency` edges recovered from the SSA use-def graph, plus the backbone→head `control_dependency`, the loop-invariant weight edge, and the cross-replan `pipeline_candidate`. Every edge carries a recovered/derived evidence label.
- **smolvla:** 695 `data_dependency` edges recovered from the SSA use-def graph, plus the backbone→head `control_dependency`, the loop-invariant weight edge, and the cross-replan `pipeline_candidate`. Every edge carries a recovered/derived evidence label.
- **pi05:** 14349 `data_dependency` edges recovered from the SSA use-def graph, plus the backbone→head `control_dependency`, the loop-invariant weight edge, and the cross-replan `pipeline_candidate`. Every edge carries a recovered/derived evidence label.
- **xr0:** 46 `data_dependency` edges recovered from the SSA use-def graph, plus the backbone→head `control_dependency`, the loop-invariant weight edge, and the cross-replan `pipeline_candidate`. Every edge carries a recovered/derived evidence label.
- **bitvla:** 28 `data_dependency` edges recovered from the SSA use-def graph, plus the backbone→head `control_dependency`, the loop-invariant weight edge, and the cross-replan `pipeline_candidate`. Every edge carries a recovered/derived evidence label.

The graph is a complete **structural** multi-rate contract: every node and edge is recovered from the capture, the model config, or derived from them — what runs at which rate, which state persists, and which operator feeds which. Per-phase wall-clock timing is a runtime *measurement* (orthogonal to this static contract), not a missing structural fact.
