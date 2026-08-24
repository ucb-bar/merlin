# AGENT.md — artifacts/dse-guidance

## Purpose

merlin-dse-guidance outputs (topology, triage, candidate axes, case_study, study, design_envelope) + timestamped <scope>_<ts>_dse_analysis runs.

## Invariants

- Contents are gitignored EXCEPT the skeletons (AGENT.md / README.md / .gitkeep) and the
  explicit per-file negations in `.gitignore`. This concern has curated negations -- see the
  rationale beside them; anything not named there is regenerable and is never committed.
- Created via `merlin.common.artifacts` (start_run / new_product / cache_dir), never hand-built paths.
- Axis: **workload/model (+ analysis mode); NOT hardware target**.
