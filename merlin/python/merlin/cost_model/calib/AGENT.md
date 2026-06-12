# AGENT.md — merlin/python/merlin/cost_model/calib

## Purpose

Baremetal calibration microbenchmarks for the instruction cost model. Each isolates one
Gemmini command class; rdcycle brackets the region so per-command costs are recoverable by
regression against the Verilator sim.

## What belongs here

- Files appropriate to the purpose above.

## What does not belong here

- Unrelated code or artifacts.
- Generated outputs (use gitignored `build/`/`output/`).

## Invariants

- Keep this directory focused on its stated purpose.
- Every subdirectory must also contain an AGENT.md.
