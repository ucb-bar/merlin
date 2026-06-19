# Presentation plots index (all)

> Each rendered plot with its one-sentence DSE-search-space implication. Structural axes only (counts / bytes / fractions) — none is a performance metric.

### Evidence type by workload

![evidence_type_by_workload](generated_plots/evidence_type_by_workload.png)

**DSE implication:** How much of each workload's evidence is IR-recovered vs assumed — where the search space rests on recovered structure vs reference values.

### Evidence type by analysis phase

![evidence_type_by_phase](generated_plots/evidence_type_by_phase.png)

**DSE implication:** Evidence provenance per analysis phase (traceability).

### Shape-class MAC share by workload

![shape_class_mac_share](generated_plots/shape_class_mac_share.png)

**DSE implication:** MAC mass per shape class — which matmul shapes a DSE primitive set must cover to capture most compute.

### Primitive x workload structural coverage

![primitive_coverage_heatmap](generated_plots/primitive_coverage_heatmap.png)

**DSE implication:** Which candidate primitives tile each workload's shapes under 10% pad waste — the primitive search space, not a performance ranking.

### Primitive coverage + max regret

![primitive_regret_bar](generated_plots/primitive_regret_bar.png)

**DSE implication:** Coverage vs worst-case cross-workload regret per primitive — primitives with high regret are corpus-overfit candidates DSE should treat cautiously.

### Boundary placement: abstraction x level

![boundary_placement_heatmap](generated_plots/boundary_placement_heatmap.png)

**DSE implication:** At which HW/SW levels each abstraction could sit — the boundary search space DSE must explore (Merlin enumerates, does not choose).

### Resident capacity by dtype (per region)

![resident_capacity_by_dtype](generated_plots/resident_capacity_by_dtype.png)

**DSE implication:** Resident weight bytes per region by dtype — the on-chip capacity the residency search space is sized against.

### Avoidable weight reload by region

![avoidable_reload_by_region](generated_plots/avoidable_reload_by_region.png)

**DSE implication:** Weight bytes re-read across the K-loop that residency could avoid — where a residency/packed-store axis has the most to act on.

### Candidates unblocked per measurement

![measurement_priority_bar](generated_plots/measurement_priority_bar.png)

**DSE implication:** How many blocked candidates each missing input would unblock — what to capture/measure next, not a result.

### Available parallelism by workload

![critical_path_parallelism](generated_plots/critical_path_parallelism.png)

**DSE implication:** Inter-op work/span per workload — the unit-multiplicity the heterogeneity search space could exploit.

### Decision: single primitive choice -> coverage

![decision_primitive_choice](generated_plots/decision_primitive_choice.png)

**DSE implication:** If DSE builds only ONE compute primitive, how much of each workload's MACs it can tile under 10% waste — the worst-case bar shows no single primitive covers every workload, so the search space needs both a tile and a GEMV lane.

### Decision: weight residency -> bytes moved vs loop count

![decision_weight_residency](generated_plots/decision_weight_residency.png)

**DSE implication:** Weight bytes moved as the head loop count grows: reload-every-step (linear) vs keep-resident (flat). The vertical gap at each workload's real K is the avoidable reload a residency knob removes (bytes, not bandwidth).

### Decision: on-chip capacity + dtype -> weights resident

![decision_capacity_dtype](generated_plots/decision_capacity_dtype.png)

**DSE implication:** How many workloads become fully weight-resident as the on-chip capacity budget grows, per storage dtype — low-bit dtypes reach full residency at a smaller budget, quantifying the capacity-vs-dtype trade in the search space.

### Decision: shard axis + count -> extra data-movement bytes

![decision_sharding_cost](generated_plots/decision_sharding_cost.png)

**DSE implication:** Extra data-movement bytes added by sharding 2/4/8 ways along M, N, or K: M/N shards are reduction-free (broadcast only) while K shards add partial-sum traffic — the cost side of the parallelization decision.

### Primitive-set frontier (worst vs mean coverage)

![primitive_set_frontier](generated_plots/primitive_set_frontier.png)

**DSE implication:** Each point is a primitive (or best set): x=mean coverage, y=worst-workload coverage. Upper-right = broadly useful; high-x/low-y = corpus-overfit. The best single primitive sits low on y; a {tile + GEMV-lane} set reaches the top-right — DSE should search primitive SETS.

### Operator cumulative MAC share (few giant vs many even ops)

![operator_cumulative_mac](generated_plots/operator_cumulative_mac.png)

**DSE implication:** Cumulative MAC share vs top-k operators per workload: a steep curve (rdt: 1 op = 87%) means DSE sizes for a few giant ops; a gradual curve means many even ops.

### Abstraction necessity (necessary/useful/possible/blocked)

![boundary_necessity_matrix](generated_plots/boundary_necessity_matrix.png)

**DSE implication:** Strict necessity per abstraction × workload (necessary/useful/possible/blocked/N-A) — what DSE should commit to, not merely what is possible; low-bit abstractions are blocked by the dequantized capture.

### Decision: shard top-MAC ops -> extra bytes / output bytes

![decision_sharding_per_top_op](generated_plots/decision_sharding_per_top_op.png)

**DSE implication:** For the top-MAC ops, extra sharding bytes normalized by the op's output bytes, per M/N/K axis — which axis partitions a hot op cheaply (the per-operator view, not a corpus aggregate).

### Frontier robustness: worst coverage vs set size by threshold

![primitive_frontier_by_threshold](generated_plots/primitive_frontier_by_threshold.png)

**DSE implication:** Worst-workload coverage vs primitive-set size at 5/10/20% pad waste — whether the 'a 2-set suffices' claim survives threshold perturbation (the specific pair may shift; structural coverage only).

### Macro vs micro vs worst primitive coverage

![macro_vs_micro_primitive_coverage](generated_plots/macro_vs_micro_primitive_coverage.png)

**DSE implication:** Macro (equal-weight) vs micro (MAC-weighted) vs worst coverage as the primitive set grows — the second primitive is where worst-workload coverage jumps; structural, no performance.

### Required compute envelope (requirement, not measured)

![required_compute_envelope](generated_plots/required_compute_envelope.png)

**DSE implication:** Required compute rate (= configured-K replan MACs / deadline) vs replan deadline — a REQUIREMENT a future accelerator must exceed, not a measured rate.

### Required memory-movement envelope (residency removes Kx)

![required_memory_movement_envelope](generated_plots/required_memory_movement_envelope.png)

**DSE implication:** Required weight bandwidth at a 100 ms deadline, weights reloaded every step vs kept resident — residency removes a K× bandwidth requirement (the residency search axis); a requirement, not a measured rate.

### Required command-rate envelope (proxy; not measured)

![required_command_rate_envelope](generated_plots/required_command_rate_envelope.png)

**DSE implication:** Required dispatch rate vs deadline — a PROXY (matmul-count, ~12× undercount), measured only for small_llama; not a hardware command rate.

### Workload influence: leave-one-out micro delta

![workload_influence_loo_delta](generated_plots/workload_influence_loo_delta.png)

**DSE implication:** Largest leave-one-out micro swing per cross-workload metric — red bars are metrics whose winner is stable but whose magnitude is not (drop one workload and the number moves sharply).

### Recovered work: linear-GEMM vs attention MAC mass

![work_coverage_by_workload](generated_plots/work_coverage_by_workload.png)

**DSE implication:** Recovered MAC mass split into linear-GEMM vs attention (both from IR shapes, no config) — attention is NOT erased, just lowered to generic and re-parsed.

### Visible linear fraction (linear / (linear+attention))

![visible_linear_fraction](generated_plots/visible_linear_fraction.png)

**DSE implication:** Fraction of recovered MAC work that is the linear-GEMM geometry this study analyzes (rest = attention) — answers 'are we analyzing most of the compute?'.

