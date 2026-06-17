# Rate-mismatch report

> The multi-rate contract makes the rate mismatches explicit: a backbone that runs once per replan, an action head that runs K times, and a control loop that consumes H actions at a fixed frequency are three different rates the flat capture collapsed into one. **Structural only — no speedup, no cycle budget claimed.**

## Per-workload rate structure

| workload | backbone | repeated head | K | H | control_rate_hz | replan deadline (s) |
|---|---|---|---|---|---|---|
| rdt | once_per_replan | K_times_per_replan | 5 | 64 | 30.0 | 2.133333 |
| openvla | once_per_replan | token_loop | 7 | 7 | 5.0 | 1.4 |
| small_llama | once_per_replan | token_loop | 32 | 32 | 30.0 | 1.066667 |
| tiny_llama | once_per_replan | token_loop | 32 | 32 | 30.0 | 1.066667 |

## Recovered vs assumed vs unavailable

- **Recovered (`recovered_from_ir` / `recovered_from_prov_fqn`):** region roles, per-region MACs/bytes, operator shapes, loop-invariant weight bytes, the once-vs-repeated cadence split.
- **Assumed (`assumed_reference`):** K, H, control_rate_hz (architecture reference values, not measured); the replan deadline is **derived** from H / control_rate.
- **Unavailable (`unavailable`):** per-K-step wall time, true operator data dependencies, host command/sync latency, and (where the backbone produces nothing in the flat capture) the exact boundary-crossing tensors.

## Dependency knowledge

- **rdt:** 2 edges with recovered evidence (control / loop-invariant / state-lifetime), 20 conservative `unknown_dependency` operator-order edges (true data deps not recovered).
- **openvla:** 2 edges with recovered evidence (control / loop-invariant / state-lifetime), 24 conservative `unknown_dependency` operator-order edges (true data deps not recovered).
- **small_llama:** 2 edges with recovered evidence (control / loop-invariant / state-lifetime), 14 conservative `unknown_dependency` operator-order edges (true data deps not recovered).
- **tiny_llama:** 2 edges with recovered evidence (control / loop-invariant / state-lifetime), 15 conservative `unknown_dependency` operator-order edges (true data deps not recovered).

## What is missing before scheduling/DSE

- a loop-preserving capture (Level-2) to recover the true data-dependency graph and the boundary-crossing tensors;
- measured per-phase timing (per-K-step, backbone, control tick) to turn the cadence model into a schedule;
- measured host command/sync latency to size the runtime-command layer.

Until then the graph is a **structural** multi-rate contract: correct about *what runs at which rate and which state persists*, explicit about *what timing is not yet known*.
