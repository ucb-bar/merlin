# AGENT.md — artifacts/delivery

## Purpose

Delivery packages: the zipped image sets handed to a board owner for bring-up. Each package pairs a
plain image with a `_debug_` twin of the same model, so one hand-off answers both "what number does it
get" and "what did it do", and the recipient never has to come back for the other half.

Built by `build_tools/scripts/make_delivery.py`. What goes in a package (models, hart counts, console
and clock settings, the linker's DRAM base) is read from the board descriptor — see
`merlin.runtime.boards` — never hardcoded per recipient.

## Invariants

- Contents are gitignored; only AGENT.md / README.md / .gitkeep are tracked.
- Created via `merlin.common.artifacts` (start_run / new_product / cache_dir), never hand-built paths.
- Axis: **board** (one package per bring-up target).
- A round's recipient list, contact names and per-chip bring-up notes belong in the hand-off itself,
  not in a tracked file: this repo is public, and the boards are often unreleased silicon.
