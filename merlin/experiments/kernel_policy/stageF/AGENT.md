# AGENT.md — merlin/experiments/kernel_policy/stageF

## Purpose

Stage-F measurement harnesses for the profiling slate: paired ablations run under
Spike+libgemmini (events) and Verilator (cycles). One .c per insight, one runner.

## What belongs here

- Files appropriate to the purpose above.

## What does not belong here

- Unrelated code or artifacts.
- Generated outputs (use gitignored `build/`/`output/`).

## Invariants

- Keep this directory focused on its stated purpose.
- Every subdirectory must also contain an AGENT.md.
