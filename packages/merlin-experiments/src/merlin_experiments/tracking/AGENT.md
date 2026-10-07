# AGENT.md — packages/merlin-experiments/src/merlin_experiments/tracking

## Purpose

Experiment tracking views: a static HTML dashboard and a live terminal view, from records only.

## Modules

- `html.py` — Render a tracking summary as ONE self-contained HTML page: inline CSS, inline SVG, a few lines of JS.
- `records.py` — Read an experiment's own records into one summary: what the dashboard and ``watch`` show.
- `text.py` — The terminal view of a tracking summary: plain text, ANSI colour optional, no curses.

<!-- Purpose/Modules derived from docstrings via build_tools/scripts/gen_package_docs.py.
     Add hand-written notes (invariants, gotchas) below. -->

## Invariants

- Read records, never produce evidence: no module here measures, grades, builds, or writes into a run
  directory or a measurement store. The only write is the dashboard page itself.
- A missing, unreadable or foreign-schema record is `None` plus an inventory row, rendered "not
  recorded" -- never a crash, a zero or a guessed default.
- Classify phase-2 outcomes from the fields their owners write (`isa_prohibited`, `coverage_regression`,
  `screen_failed`, `infra_*`, `timing_status`) using the owners' constants and `gates.is_infra_refusal`;
  never by matching text patterns of your own.
- The page stays self-contained: inline CSS/SVG/JS only, no network, no external file.
