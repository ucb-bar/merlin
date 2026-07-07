# AGENT.md — merlin/benchmarks

## Purpose
Curated **benchmark/workload INPUTS** the library reads at runtime — things that cannot be
regenerated cheaply and are the source-of-record for DSE/kernel analysis. **Not results.**

## What lives here (curated inputs — keep)
- `dse_guidance/` — the VLA/LLM workload corpus + measured data `merlin.dse_guidance` consumes:
  - `recaptures_loop/` (primary) and `recaptures/` (flat, via `MERLIN_DSE_CORPUS=flat`) — per-model
    `model.mlir` captures. **Oversized models (pi05/smolvla/groot) live out-of-git under
    `artifacts/recaptures/dse_guidance/` (regenerable via m2m); `case_study._recap_dir` falls back
    there.** Small reduced-config captures stay committed.
  - `region_maps/`, `measured_cycles.yaml`, `measured_dispatch.yaml`, `accuracy_gate.yaml` —
    hand-recorded / role-map inputs (cannot be regenerated without hardware).
  - regen tooling (`variant_capture.py`, `reproduce_case_study.sh`, `REGEN.md`) documents provenance.
- `semantic_memory/` — curated matmul-reuse workload specs (pinned by `check_structure`
  `REQUIRED_BENCHMARKS`).

## What does NOT belong here
- Generated results, reports, and plots → `artifacts/dse-guidance/`. Runs → `runs/`; compiled → `build/`.

## Used by
`merlin.dse_guidance` (case_study, loader, quant_metadata, cost_calibration, accuracy_gate),
`merlin.kernels.validate`.

## Invariants
Curated INPUTS only; tool products belong under `artifacts/`. Every subdirectory has an AGENT.md.
