# AGENT.md — artifacts/dse-guidance

## Purpose

merlin-dse-guidance outputs (topology, triage, candidate axes, case_study, study, design_envelope) + timestamped <scope>_<ts>_dse_analysis runs.

## Invariants

- Contents are gitignored; only AGENT.md / README.md / .gitkeep are tracked.
- Created via `merlin.common.artifacts` (start_run / new_product / cache_dir), never hand-built paths.
- Axis: **workload/model (+ analysis mode); NOT hardware target**.
