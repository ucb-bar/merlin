# DSE-usefulness scorecard (all)

> Can the package answer DSE-relevant questions? Each query is answered from existing artifacts with a status + recommended presentation use. Structural only — no performance claim.

| # | query | status | use | supporting artifacts |
|---|---|---|---|---|
| 1 | What compute primitive shapes should a DSE search space include? | **strong** | main | primitive_coverage_matrix.csv, primitive_regret_table.csv |
| 2 | Which primitive shapes are broadly useful across workloads? | **strong** | main | primitive_regret_table.csv |
| 3 | Which primitive shapes are workload-specific / high-regret? | **partial** | backup | primitive_regret_table.csv |
| 4 | Which workloads suggest heterogeneous processing units? | **partial** | backup | processing_unit_guidance.yaml, resource_pressure_table.csv |
| 5 | Which workloads suggest bounded-loop commands? | **partial** | backup | workload_contract_graph.yaml, boundary_candidate_contracts.yaml |
| 6 | Which abstractions are most strongly supported across workloads? | **strong** | main | abstraction_pressure_ranking.csv, hw_sw_boundary_matrix.csv |
| 7 | Which abstractions are family-specific? | **partial** | backup | abstraction_pressure_table.csv, workload_family_table.csv |
| 8 | Which boundary placements are plausible for resident weights? | **strong** | main | boundary_candidate_contracts.yaml, hw_sw_boundary_matrix.csv |
| 9 | Which boundary placements are plausible for K-loop execution? | **partial** | backup | boundary_candidate_contracts.yaml |
| 10 | Which boundary placements are plausible for packed low-bit tensors? | **weak** | backup | boundary_candidate_contracts.yaml |
| 11 | Which objects should potentially cross the HAL boundary? | **partial** | backup | runtime_object_candidates.yaml |
| 12 | Which command-ISA abstractions are structurally suggested? | **partial** | backup | command_isa_candidates.yaml |
| 13 | Which accelerator-ISA primitives are structurally suggested? | **partial** | backup | isa_candidate_primitives.yaml |
| 14 | Which candidates are blocked by missing compiler proof? | **partial** | backup | compiler_proof_matrix.csv |
| 15 | Which candidates are blocked by missing measurements? | **partial** | backup | dse_readiness_summary.csv, measurement_priority_table.csv |
| 16 | Which measurements unblock the most candidates? | **partial** | backup | measurement_priority_table.csv |
| 17 | Which findings rely heavily on assumptions? | **partial** | backup | dse_contract.json |
| 18 | Which analyses are currently shallow or incomplete? | **partial** | backup | lost_numerical_contracts.csv, boundary_candidate_contracts.yaml |
| 19 | Which plots should be generated for presentation? | **strong** | main | operator_shape_table.csv, critical_path_table.csv, hw_sw_boundary_matrix.csv, data_movement_table.csv |
| 20 | Which claims are safe to present without quantitative performance measurements? | **partial** | backup | verification_report.md |

5/20 queries answerable **strong**; 0 unavailable.
