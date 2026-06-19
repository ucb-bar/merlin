# Memory-hierarchy envelope report

> Per-region byte envelope: weights, activations, outputs, and the dtype-scaled resident set. **No bandwidth/speedup is claimed** (no bandwidth feasibility) — that needs a design YAML. Bytes a flat dequantized capture cannot expose (intermediate, scale, KV) are `unavailable`.

## Top memory-pressure regions (by weight bytes)

| workload | region | weight B | act-in B/inv | output B/inv | reuse | avoidable reload B | dominant class |
|---|---|---|---|---|---|---|---|
| pi05 | backbone_once | 12,452,184,064 | 2,741,452,800 | 3,438,067,712 | 1× | 0 | weight-dominated |
| molmoact | backbone_once | 3,787,456,512 | 3,815,424 | 6,373,376 | 1× | 0 | weight-dominated |
| molmoact | repeated_head | 3,787,456,512 | 489,472 | 811,008 | 8× | 26,512,195,584 | weight-dominated |
| groot_n1d7 | repeated_head | 2,200,436,736 | 40,995,840 | 38,924,288 | 4× | 6,601,310,208 | weight-dominated |
| pi05 | repeated_head | 1,719,926,784 | 40,921,344 | 46,754,048 | 10× | 15,479,341,056 | weight-dominated |
| smolvla | backbone_once | 701,620,224 | 372,969,600 | 371,230,464 | 1× | 0 | weight-dominated |
| tiny_llama | backbone_once | 614,465,536 | 1,155,072 | 1,274,880 | 1× | 0 | weight-dominated |
| tiny_llama | repeated_head | 614,465,536 | 151,552 | 271,360 | 7× | 3,686,793,216 | weight-dominated |

## Findings

- **Memory pressure is weight-dominated** across the recaptured workloads (weights 26,978,331,648 B, activations 4,056,612,992 B, outputs 5,010,013,632 B).
- **Top avoidable-reload candidate:** `pi05/backbone_once` — 0 B avoidable if weights are made resident (= weight_bytes × (reuse − 1)).
- **Repeatedly implied abstractions:** `resident_weight_object` (weights reused across the loop) and `resident_activation_object` (activations recomputed per step) — see `memory_abstraction_candidates.yaml`.

## Missing for real bandwidth feasibility

- intermediate-materialization, scale/zero-point, and KV/prefix bytes (`unavailable` in a flat dequantized capture);
- a target memory hierarchy (capacities, bandwidths) — supplied via a design YAML, absent here. **No bandwidth or deadline feasibility is claimed.**
