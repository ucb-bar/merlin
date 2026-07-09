# AGENT.md — merlin/experiments/kernel_policy/stageF

## Purpose

Stage-F measurement **harness** (`run_l2.py`) for the profiling slate: compiles + runs the paired
ablations under Spike+libgemmini (events) and Verilator (cycles). One runner; the ablation kernels
themselves are a library-consumed benchmark input and live in `merlin/benchmarks/cost_calib/`.

## What belongs here

- `run_l2.py` — the harness. The ablation `.c` kernels do NOT live here (they moved to
  `merlin/benchmarks/cost_calib/`, read by both this harness and `merlin.cost_model.calibrate`).

## What does not belong here

- Unrelated code or artifacts.
- Generated outputs (write to `runs/`/`artifacts/`; compiled trees to `build/`).

## Invariants

- Keep this directory focused on its stated purpose.
- Every subdirectory must also contain an AGENT.md.
