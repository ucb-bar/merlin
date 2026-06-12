# AGENT.md — merlin/cost_model

## Purpose

Static instruction cost models (the shared currency for Stage-F/G profiling and Autocomp).
Predict region cycles from command counts; coefficients calibrated against the cycle-exact
Verilator sim. Per-target module (`gemmini.py`) + calibration driver (`calibrate.py`).

## What belongs here

- Files appropriate to the purpose above.

## What does not belong here

- Unrelated code or artifacts.
- Generated outputs (use gitignored `build/`/`output/`).

## Invariants

- Keep this directory focused on its stated purpose.
- Every subdirectory must also contain an AGENT.md.
