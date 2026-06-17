# Merlin workload-contract analysis — case study

Merlin is a compiler-based **workload-contract analysis** tool for accelerator DSE. It does not pick a design and does not calibrate against existing hardware. It recovers the temporal + numerical workload contract a flat capture erases and emits a DSE-ready package: region facts, hardware-independent requirements, HW/SW abstraction candidates, a measurement plan, and a readiness report.

Workloads (real `prov.fqn` recaptures): **rdt, openvla, small_llama, tiny_llama**.

## Read this folder
- `case_study_summary.md` — start here (the central table).
- `<workload>/workload_contract_report.md` — full per-workload package.
- `requirements_table.csv`, `dtype_capacity_table.csv` — design requirements (hw-independent).
- `abstraction_pressure_table.csv` — implied HW/SW abstractions + DSE knobs (per workload).
- `abstraction_pressure_ranking.csv` — across-workload pressure ranking (a count, not a speedup).
- `resident_state_table.csv` — state lifetimes (loop-invariant / carried / boundary-crossing) + the abstraction each implies.
- `compiler_proof_matrix.csv` — the compiler proof each abstraction needs + its status (proven_for_workload / assumed / unknown).
- `workload_family_table.csv` — workloads clustered into families (iterative_denoise / token_decode / single_shot).
- `<workload>/dse_search_space_template.yaml`, `dse_search_space_template_<family>.yaml` — the **DSE search-space template** (the bridge a DSE engine consumes: enabled axes + knobs).
- `measurement_priority_table.csv` — what to measure next, ranked by candidates unblocked.
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
