# Merlin workload-contract analysis — case study

**Consume this package as one object:** `dse_contract.json` is the machine-readable manifest (per-workload readiness + facts, the search-space knob groups, the boundary-placement top, the measurements still needed, and a topic→artifact index). Query it with `merlin-dse-guidance --query {summary,knobs,boundary[:abstraction],missing,index}`.

**Meta-analysis (P13):** `merlin-dse-guidance --insight-mining` mines this package (per network + combined), scores DSE-usefulness, and extracts evidence-tiered presentation findings + plots into a regeneratable, **non-committed** timestamped run under `results/` (`<scope>_<UTCstamp>_dse_analysis/`).

Merlin is a compiler-based **workload-contract analysis** tool for accelerator DSE. It does not pick a design and does not calibrate against existing hardware. It recovers the temporal + numerical workload contract a flat capture erases and emits a DSE-ready package: region facts, hardware-independent requirements, HW/SW abstraction candidates, a measurement plan, and a readiness report.

Workloads (real `prov.fqn` recaptures): **rdt, openvla, tiny_llama, rdt2, groot_n1d7, molmoact, smolvla, pi05, xr0, bitvla**.

## Read this folder
- `current_state_audit.md` — V0 freeze audit (standalone); `claim_evidence_matrix.csv`, `known_limitations.md`, `reproducibility_check.log` are its companions.
- `case_study_summary.md` — start here (the central table).
- `<workload>/workload_contract_report.md` — full per-workload package.
- `requirements_table.csv`, `dtype_capacity_table.csv` — design requirements (hw-independent).
- `abstraction_pressure_table.csv` — implied HW/SW abstractions + DSE knobs (per workload).
- `abstraction_pressure_ranking.csv` — across-workload pressure ranking (a count, not a speedup).
- `resident_state_table.csv` — state lifetimes (loop-invariant / carried / boundary-crossing) + the abstraction each implies.
- `compiler_proof_matrix.csv` — the compiler proof each abstraction needs + its status (proven_for_workload / assumed / unknown).
- `workload_family_table.csv` — workloads clustered into families (iterative_denoise / token_decode / single_shot).
- `<workload>/dse_search_space_template.yaml`, `dse_search_space_template_<family>.yaml` — the **DSE search-space template** (per-workload abstraction axes + knobs).
- `hw_sw_boundary_matrix.csv`, `boundary_candidate_contracts.yaml`, `boundary_placement_report.md`, `responsibility_split_matrix.csv`, `interface_contract_sketches.md`, `isa_candidate_primitives.yaml`, `runtime_object_candidates.yaml`, `command_isa_candidates.yaml`, `boundary_dse_knobs.yaml` — the **HW/SW boundary search space**: where each abstraction could live (compiler/runtime/command/ISA/microcode/datapath), what each side manages, and the compiler proof + DSE knobs each placement creates (Merlin generates options, does not choose; no speedup).
- `dse_search_space_knobs.yaml`, `dse_search_space_knobs.md` — the **consolidated structural search-space knobs** discovered across P5-P12 (primitive shapes, sharding, processing units, pipeline overlap, memory/DMA, epilogue fusion, boundary placement) — the capstone bridge a DSE engine consumes.
- `measurement_priority_table.csv` — what to measure next, ranked by candidates unblocked.
- `operator_shape_table.csv`, `operator_geometry.yaml` — per-operator geometry (M/N/K, MACs, aspect, shape_class + semantic role). `shape_summary_by_workload.csv`, `shape_summary_by_region.csv`, `operator_cluster_table.csv`, `operator_geometry_report.md` summarise it (structural geometry only).
- `tile_waste_table.csv`, `primitive_coverage_matrix.csv`, `primitive_regret_table.csv` — candidate compute-primitive (tile / GEMV-lane) structural coverage + cross-workload regret; `primitive_coverage_report.md`, `cross_workload_coverage_report.md` read them (no speedup).
- `workload_contract_graph.yaml`, `workload_contract_graph_summary.md` — the **multi-rate workload contract graph** (the central IR later phases consume: phase/region/operator/state nodes + typed edges). `phase_rate_table.csv`, `multi_rate_contract.yaml`, `rate_mismatch_report.md` expose the per-phase cadence + rate model (structural only).
- `dag_parallelism_report.md`, `critical_path_table.csv`, `concurrency_windows.csv`, `parallel_region_candidates.yaml` — inter-op DAG concurrency (work/span, not a speedup).
- `sharding_table.csv`, `sharding_opportunities.yaml`, `intra_op_sharding_report.md` — per-matmul M/N/K sharding geometry + required reduction/broadcast abstractions.
- `operator_cluster_to_hierarchy.csv`, `parallel_hierarchy_hints.yaml`, `resource_pressure_table.csv`, `processing_unit_candidates.yaml`, `processing_unit_parallelism_report.md` — hierarchical resource analysis: which processing-unit shapes the workloads imply (one bigger / many identical / specialized).
- `pipeline_envelope.yaml`, `pipeline_stage_table.csv` — multi-rate phase model (cadence per phase). `pipeline_candidates.yaml`, `buffering_requirement_table.csv`, `overlap_opportunities.md` — candidate phase overlaps + the buffer/event/queue abstractions each requires (structural, not scheduled).
- `processing_unit_guidance.yaml`, `heterogeneity_report.md` — monolithic vs. replicated vs. heterogeneous evidence + the search-space implication (evidence only, no selection).
- `memory_hierarchy_envelope.yaml`, `data_movement_table.csv`, `reuse_lifetime_table.csv`, `memory_abstraction_candidates.yaml`, `memory_envelope_report.md` — per-region byte envelope (weight/activation/output + dtype-scaled resident set), reuse/residency, and the memory abstractions implied (no bandwidth claim).
- `dma_stream_table.csv`, `buffer_requirement_table.csv`, `dma_pressure_report.md` — structural data-movement streams + minimum buffering per region (no bandwidth/deadline claim).
- `epilogue_pattern_table.csv`, `accumulator_contract_table.csv`, `numerical_epilogue_candidates.yaml`, `lost_numerical_contracts.csv`, `fusion_opportunity_report.md` — fusion/epilogue/accumulator placement: detected matmul-epilogue patterns, accumulator + dequant/requant contract, fused abstraction certificates, and the numerical contracts the flat capture erased (no low-bit perf claim).
- `traffic_table.csv` — per-region byte traffic + avoidable reload (memory/reuse envelope).
- `dispatch_granularity_table.csv` — command-graph view (honest: loop unrolled, syncs unavailable).
- `accuracy_gated_dtype_candidates.csv` — which low-bit formats are accuracy-legal vs blocked (int8 measured; fp8/int4 unavailable).
- `torchao_integration_plan.md` — plan (not a sweep) for wiring low-bit formats to the numerical candidates.
- `dse_readiness_summary.csv` — what a DSE engine can consume today + what's missing.
- `accuracy_gate_report.md` — measured int8 accuracy (the measurable-now leg).
- `numerical_contract_fidelity_report.md`, `dispatch_coupling_report.md`, `cost_calibration.md` — supporting evidence (calibration is a demoted existing-target anchor).

## Regenerate
```
merlin-dse-guidance --case-study \
  --out merlin/benchmarks/dse_guidance/case_study
```

Every number carries an evidence label (`recovered_from_ir` / `recovered_from_prov_fqn` / `assumed_reference` / `derived_requirement` / `design_assumption` / `measured` / `proxy_measured` / `unavailable`). No file claims a speedup for unbuilt hardware.
