# Memory-hierarchy envelope report

> Per-region byte envelope: weights, activations, outputs, and the dtype-scaled resident set. **No bandwidth/speedup is claimed** (no bandwidth feasibility) — that needs a design YAML. Bytes a flat dequantized capture cannot expose (intermediate, scale, KV) are `unavailable`.

## Top memory-pressure regions (by weight bytes)

| workload | region | weight B | act-in B/inv | output B/inv | reuse | avoidable reload B | dominant class |
|---|---|---|---|---|---|---|---|
| pi05 | repeated_head | 9,210,953,728 | 1,944,086,784 | 2,643,781,888 | 10× | 82,898,583,552 | output-dominated |
| pi05 | backbone_once | 4,961,157,120 | 838,287,360 | 841,039,872 | 1× | 0 | weight-dominated |
| molmoact | repeated_head | 3,787,456,512 | 3,915,776 | 6,488,064 | 8× | 26,512,195,584 | weight-dominated |
| tiny_llama | repeated_head | 614,465,536 | 606,208 | 1,085,440 | 32× | 19,048,431,616 | weight-dominated |
| smolvla | backbone_once | 426,246,144 | 346,644,992 | 343,889,664 | 1× | 0 | weight-dominated |
| rdt | repeated_head | 391,118,848 | 41,519,104 | 77,030,912 | 5× | 1,564,475,392 | weight-dominated |
| rdt2 | repeated_head | 301,268,992 | 2,839,552 | 3,188,928 | 5× | 1,205,075,968 | weight-dominated |
| groot_n1d7 | repeated_head | 295,698,432 | 5,349,376 | 5,021,696 | 4× | 887,095,296 | weight-dominated |

## Findings

- **Memory pressure is output-dominated** across the recaptured workloads (weights 20,230,306,816 B, activations 20,969,884,416 B, outputs 28,192,209,088 B).
- **Top avoidable-reload candidate:** `pi05/repeated_head` — 82,898,583,552 B avoidable if weights are made resident (= weight_bytes × (reuse − 1)).
- **Repeatedly implied abstractions:** `resident_weight_object` (weights reused across the loop) and `resident_activation_object` (activations recomputed per step) — see `memory_abstraction_candidates.yaml`.

## Missing for real bandwidth feasibility

- intermediate-materialization, scale/zero-point, and KV/prefix bytes (`unavailable` in a flat dequantized capture);
- a target memory hierarchy (capacities, bandwidths) — supplied via a design YAML, absent here. **No bandwidth or deadline feasibility is claimed.**
