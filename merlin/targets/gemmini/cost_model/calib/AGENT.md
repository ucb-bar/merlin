# AGENT.md — merlin/targets/gemmini/cost_model/calib

## Purpose

Baremetal calibration microbenchmarks for this target's linear cost model. Each isolates one
command class; rdcycle brackets the region so per-command costs are recoverable by regression
against the Verilator sim (`../calibrate.py`).

## What belongs here

- Calibration microbenchmark sources consumed by `../calibrate.py`.

## What does not belong here

- Unrelated code or artifacts.
- Generated outputs (binaries and reports go under `out/artifacts/cache/cost_model/`).

## Invariants

- Keep this directory focused on its stated purpose.
- Every subdirectory must also contain an AGENT.md.
