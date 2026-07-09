# AGENT.md — merlin/benchmarks/cost_calib

## Purpose
Curated **cost-model calibration kernels** — small paired-ablation C microbenchmarks (resident-RHS,
accumulator-commit, dispatch-batching, vl-tail) the LIBRARY compiles + runs to calibrate the Gemmini
cost model. Library-consumed INPUTS (not results), so they live here under `benchmarks/`, not in an
experiment.

## What lives here
- `*_ablation.c` — the paired on/off ablation harnesses (hand-authored; cannot be generated).

## Used by
- `merlin.cost_model.calibrate` (compiles + runs them to derive per-event cycle costs).
- `merlin/experiments/kernel_policy/stageF/run_l2.py` (the Stage-F harness — an experiment that reads
  this benchmark input; direction is fine).

## Invariants
Curated INPUTS only. Generated results → `artifacts/`; runs → `runs/`.
