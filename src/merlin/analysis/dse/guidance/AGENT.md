# AGENT.md — merlin/python/merlin/dse_guidance

## Purpose

DSE guidance: turn a flat workload capture into grounded DSE-axis guidance.

## Modules

- `accuracy_gate.py` — Quantization accuracy gate — the measurable-now leg of the numerical contract.
- `aet_ingest.py` — Instrumentation adapter: ingest *measured* CPU/runtime coupling from the ``aet`` harness.
- `arithmetic_intensity.py` — Hardware-INDEPENDENT roofline: arithmetic intensity + ridge-point regime (P24).
- `attribution.py` — Level-1 topology recovery: attribute real IR facts to VLA topology phases.
- `axes.py` — The DSE axis catalog: what each axis can reduce, and by how much.
- `baseline_cost.py` — Parse a baseline cost breakdown and keep it honest.
- `boundary_placement.py` — HW/SW boundary-placement analysis — the boundary search space, not the choice.
- `calibration.py` — One calibration anchor: prediction vs measurement.
- `candidates.py` — Structural DSE-candidate discovery for VLAs.
- `case_study.py` — Cross-workload case study over real `prov.fqn` captures.
- `cli.py` — ``merlin-dse-guidance`` CLI.
- `command_graph.py` — Command-graph / control-interface analysis — honest about what a flat capture can show.
- `compiler_proof.py` — Compiler-proof / transformability matrix — what must be proven to exploit each abstraction.
- `contract.py` — Workload-contract analysis package — what Merlin hands a future DSE engine.
- `contract_graph.py` — Multi-rate workload contract graph — the central IR later phases consume.
- `cost_calibration.py` — Calibration against an EXISTING target — a demoted sanity-check / anchor, not the DSE result.
- `design_envelope.py` — Design-envelope derivation — requirements and theoretical bounds, NOT calibration.
- `dispatch_measure.py` — Measured dispatch coupling — the first measured runtime leg.
- `dma_buffer_analysis.py` — DMA / stream / buffer analysis — the data-movement search-space inputs.
- `dtype_certificates.py` — Accuracy-gated dtype candidate certificates — which low-bit formats are DSE-legal, and why.
- `evidence.py` — Evidence types, confidence *weights*, and cost-tier *proxies* — the single source of truth.
- `fidelity.py` — Capture fidelity report — does a flat capture preserve the VLA DSE unit?
- `fusion_epilogue.py` — Fusion / epilogue / accumulator contract — numerical structure beyond dtype capacity.
- `insight_mining.py` — Evidence mining + insight extraction over the committed dse_guidance case-study package.
- `loader.py` — Load a ``workload_region`` for DSE guidance.
- `loop_recovery.py` — Loop-preserving capture recovery (P21-S1).
- `mapspace.py` — Tool A (P20): Timeloop-native mapspace seeds from the recovered operator graph.
- `memory_envelope.py` — Memory-traffic / reuse envelope — the bytes each region moves, and what residency avoids.
- `models.py` — Registry of the real supported workloads (the model2MLIR VLA / LM zoo).
- `numerical_contract.py` — Numerical-contract fidelity audit — the precision counterpart to capture fidelity.
- `operand_locality.py` — Tool B (P20): CADOSys-style operand-locality targets from the recovered data-movement facts.
- `operator_geometry.py` — Operator-geometry extraction — the first search-space-formation layer.
- `parallelism.py` — Inter-operator DAG concurrency — how much parallelism the workload structure exposes.
- `phase_rate.py` — Phase / rate model — the cadence at which each region runs, and the workload's rate constants.
- `pipeline.py` — Orchestration core: one workload -> the full set of DSE-guidance artifacts.
- `pipeline_envelope.py` — Multi-rate phase model + pipeline-overlap analysis.
- `plots.py` — Optional matplotlib plots for DSE guidance. No-op (returns False) if matplotlib is absent.
- `presentation_final.py` — Final presentation-pass renderers (P26): clean, conference-ready restyle of the curated plot set.
- `presentation_plots.py` — Neutral PNG renderers for the insight-mining plot manifest.
- `primitive_coverage.py` — Candidate compute-primitive coverage — structural geometry only (NOT a performance model).
- `processing_unit_guidance.py` — Processing-unit multiplicity guidance — monolithic vs. replicated vs. heterogeneous.
- `quant_metadata.py` — Tool E (P20): low-bit quant-metadata visibility from the qdq recaptures.
- `real_config.py` — Real-config (deployment) magnitudes + KV sizing  (P21 S2 + S3).
- `realtime_requirement.py` — Real-time deployment requirements (HW-INDEPENDENT) — P25.
- `report.py` — CSV + Markdown emitters for the DSE guidance artifacts.
- `representation.py` — Flat vs multi-rate representation of one workload.
- `resource_hierarchy.py` — Hierarchical resource analysis — which processing-unit shapes the workloads imply.
- `search_space.py` — DSE search-space template — the bridge from Merlin to a future DSE engine.
- `shape_taxonomy.py` — Deterministic shape taxonomy for matmul-like operators (no ML clustering).
- `sharding.py` — Intra-operator sharding analysis — how each matmul can be split across N candidate units.
- `state_lifetime.py` — State-lifetime / residency analysis — which tensors persist, and the abstraction they imply.
- `study.py` — Exhaustive cross-workload study: run DSE guidance over every supported workload.
- `synth.py` — Synthesize an *analytical* baseline cost and temporal metadata from a workload region.
- `temporal.py` — Parse and validate temporal / multi-rate workload metadata.
- `topology.py` — VLA runtime topology — the workload contract a flat capture erases.
- `triage.py` — DSE axis triage: rank axes by measured/trace-derived target-gap closure.
- `workload_family.py` — Workload-family clustering — group VLAs by the system abstractions they imply.

## Subpackages

- `agent/`

<!-- Purpose/Modules derived from docstrings via build_tools/scripts/gen_package_docs.py.
     Add hand-written notes (invariants, gotchas) below. -->
