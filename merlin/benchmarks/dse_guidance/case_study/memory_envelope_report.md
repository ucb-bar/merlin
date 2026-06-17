# Memory-hierarchy envelope report

> Per-region byte envelope: weights, activations, outputs, and the dtype-scaled resident set. **No bandwidth/speedup is claimed** (no bandwidth feasibility) — that needs a design YAML. Bytes a flat dequantized capture cannot expose (intermediate, scale, KV) are `unavailable`.

## Top memory-pressure regions (by weight bytes)

| workload | region | weight B | act-in B/inv | output B/inv | reuse | avoidable reload B | dominant class |
|---|---|---|---|---|---|---|---|
| tiny_llama | repeated_head | 614,465,536 | 606,208 | 1,085,440 | 32× | 19,048,431,616 | weight-dominated |
| rdt | repeated_head | 391,118,848 | 41,519,104 | 77,030,912 | 5× | 1,564,475,392 | weight-dominated |
| openvla | backbone_once | 15,400,960 | 466,688 | 516,352 | 1× | 0 | weight-dominated |
| openvla | repeated_head | 3,145,728 | 235,520 | 409,600 | 7× | 18,874,368 | weight-dominated |
| small_llama | repeated_head | 1,712,128 | 75,264 | 93,184 | 32× | 53,075,968 | output-dominated |

## Findings

- **Memory pressure is weight-dominated** across the recaptured workloads (weights 1,025,843,200 B, activations 231,517,952 B, outputs 426,254,080 B).
- **Top avoidable-reload candidate:** `tiny_llama/repeated_head` — 19,048,431,616 B avoidable if weights are made resident (= weight_bytes × (reuse − 1)).
- **Repeatedly implied abstractions:** `resident_weight_object` (weights reused across the loop) and `resident_activation_object` (activations recomputed per step) — see `memory_abstraction_candidates.yaml`.

## Missing for real bandwidth feasibility

- intermediate-materialization, scale/zero-point, and KV/prefix bytes (`unavailable` in a flat dequantized capture);
- a target memory hierarchy (capacities, bandwidths) — supplied via a design YAML, absent here. **No bandwidth or deadline feasibility is claimed.**
