# Plot manifest (all)

> Candidate plots derived from the data. Structural axes only (counts/bytes/fractions). Rendered PNGs: 26 under generated_plots/.

| plot_id | title | type | source | rec | rendered |
|---|---|---|---|---|---|
| evidence_type_by_workload | Evidence type by workload | stacked_bar | unified_fact_table.csv | backup | yes |
| evidence_type_by_phase | Evidence type by analysis phase | stacked_bar | unified_fact_table.csv | backup | yes |
| shape_class_mac_share | Shape-class MAC share by workload | stacked_bar | shape_summary_by_workload.csv | main | yes |
| shape_class_opcount_share | Shape-class op-count share by workload | stacked_bar | shape_summary_by_workload.csv | backup | no |
| primitive_coverage_heatmap | Primitive x workload structural coverage | heatmap | primitive_coverage_matrix.csv | main | yes |
| primitive_regret_bar | Primitive coverage + max regret | grouped_bar | primitive_regret_table.csv | backup | yes |
| abstraction_pressure_bar | Abstraction pressure (workloads supporting) | bar | abstraction_pressure_ranking.csv | backup | no |
| boundary_placement_heatmap | Boundary placement: abstraction x level | heatmap | hw_sw_boundary_matrix.csv | backup | yes |
| resident_capacity_by_dtype | Resident capacity by dtype (per region) | grouped_bar | data_movement_table.csv | backup | yes |
| avoidable_reload_by_region | Avoidable weight reload by region | bar | data_movement_table.csv | main | yes |
| measurement_priority_bar | Candidates unblocked per measurement | bar | measurement_priority_table.csv | main | yes |
| critical_path_parallelism | Available parallelism by workload | bar | critical_path_table.csv | main | yes |
| epilogue_pattern_counts | Epilogue patterns by workload | stacked_bar | epilogue_pattern_table.csv | backup | no |
| decision_primitive_choice | Decision: single primitive choice -> coverage | decision_bar | primitive_coverage_matrix.csv | main | yes |
| decision_weight_residency | Decision: weight residency -> bytes moved vs loop count | decision_curve | data_movement_table.csv | main | yes |
| decision_capacity_dtype | Decision: on-chip capacity + dtype -> weights resident | decision_curve | dtype_capacity_table.csv | main | yes |
| decision_sharding_cost | Decision: shard axis + count -> extra data-movement bytes | decision_bar | sharding_table.csv | main | yes |
| primitive_set_frontier | Primitive-set frontier (worst vs mean coverage) | scatter | tile_waste_table.csv | main | yes |
| operator_cumulative_mac | Operator cumulative MAC share (few giant vs many even ops) | line | operator_shape_table.csv | main | yes |
| boundary_necessity_matrix | Abstraction necessity (necessary/useful/possible/blocked) | categorical | hw_sw_boundary_matrix.csv | main | yes |
| decision_sharding_per_top_op | Decision: shard top-MAC ops -> extra bytes / output bytes | decision_bar | sharding_table.csv | main | yes |
| primitive_frontier_by_threshold | Frontier robustness: worst coverage vs set size by threshold | line | tile_waste_table.csv | main | yes |
| macro_vs_micro_primitive_coverage | Macro vs micro vs worst primitive coverage | line | tile_waste_table.csv | main | yes |
| required_compute_envelope | Required compute envelope (requirement, not measured) | line | requirements_table.csv | main | yes |
| required_memory_movement_envelope | Required memory-movement envelope (residency removes Kx) | grouped_bar | requirements_table.csv | main | yes |
| required_command_rate_envelope | Required command-rate envelope (proxy; not measured) | line | requirements_table.csv | backup | yes |
| workload_influence_loo_delta | Workload influence: leave-one-out micro delta | bar | shape_summary_by_workload.csv | main | yes |
| work_coverage_by_workload | Recovered work: linear-GEMM vs attention MAC mass | stacked_bar | work_coverage_table.csv | main | yes |
| visible_linear_fraction | Visible linear fraction (linear / (linear+attention)) | bar | work_coverage_table.csv | main | yes |
