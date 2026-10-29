# AGENT.md — packages/merlin-experiments/src/merlin_experiments/phase1/seed_prep

## Purpose

(no module docstring — add one)

## Modules

- `__main__.py` — Pre-launch preparation of a seed compiler for a seeded open-compiler run.
- `baseline.py` — Per-case Spike cycle-model timing of a frozen compiler through an export's own grader.
- `records.py` — Seed preparation from the seed run's own records (the default, launch-path mode).
- `report.py` — Write the seed preparation products: SEED_BASELINE.{json,md} and SEED_STRUCTURE.md.
- `structure.py` — Static library-route analysis of a frozen compiler tree, per operation family.
- `timing_child.py` — Time one public-scalability case with a frozen compiler, using an export's own grader.
- `variants.py` — Infer a variant tree's own change against a seed default and apply it.

<!-- Purpose/Modules derived from docstrings via build_tools/scripts/gen_package_docs.py.
     Add hand-written notes (invariants, gotchas) below. -->
