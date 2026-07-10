# DSE-guidance result plots (PNG) — loop-preserving corpus, muted-pastel style

Full insight-mining run on the loop-preserving corpus (10 workloads). PNGs render the structural
results (counts / bytes / fractions / MACs — no perf/latency/speedup). Open the `.png` files directly.

## Start here (headline)
- **`headline_role_split.png`** — the P23 headline: per-workload structural role split from the
  `scf.for` boundary (prefix `backbone_once` ×1 vs decode/denoise `repeated_head` ×K) + IR-recovered K
  + loop-carried state. This is the prefill-vs-decode separation the flat capture could not make.
- **`work_coverage_by_workload.png`** — linear-GEMM vs attention MAC mass per workload (recovered from IR).
- **`visible_linear_fraction.png`** — how much of the MAC work the linear datapath serves.

## Requirements envelope & residency
- `required_compute_envelope.png`, `required_memory_movement_envelope.png`, `required_command_rate_envelope.png`
- `decision_weight_residency.png` — reload-vs-resident bytes vs K (dot = IR-recovered K)
- `resident_capacity_by_dtype.png`, `decision_capacity_dtype.png`, `avoidable_reload_by_region.png`

## Primitive / operator search space
- `primitive_set_frontier.png`, `primitive_frontier_by_threshold.png`, `macro_vs_micro_primitive_coverage.png`
- `primitive_coverage_heatmap.png`, `primitive_regret_bar.png`
- `operator_cumulative_mac.png`, `shape_class_mac_share.png`

## Parallelism / sharding / boundary
- `critical_path_parallelism.png`, `decision_sharding_cost.png`, `decision_sharding_per_top_op.png`
- `boundary_placement_heatmap.png`, `boundary_necessity_matrix.png`

## Evidence / robustness / measurement
- `evidence_type_by_workload.png`, `evidence_type_by_phase.png`
- `workload_influence_loo_delta.png`, `measurement_priority_bar.png`, `decision_primitive_choice.png`

## Text digest (the full insight-mining output)
The `../plots_run/` folder holds the textual results from the same run: `DSE_FINDINGS.md` (the digest),
`capture_fidelity_matrix.md` (the caveat-flip, corpus-wide), `signal_findings_report.md`,
`canonical_signal_table.csv`, `abstraction_necessity_table.csv`, etc.
