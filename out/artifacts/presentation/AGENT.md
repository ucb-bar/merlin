# AGENT.md — artifacts/presentation

## Purpose

Cross-tool presentation decks, paper figures, exported media not owned by a single tool.

## Invariants

- Contents are gitignored EXCEPT the skeletons (AGENT.md / README.md / .gitkeep) and the
  explicit per-file negations in `.gitignore`. This concern has curated negations -- see the
  rationale beside them; anything not named there is regenerable and is never committed.
- Created via `merlin.common.artifacts` (start_run / new_product / cache_dir), never hand-built paths.
- Axis: **topic/campaign**.
