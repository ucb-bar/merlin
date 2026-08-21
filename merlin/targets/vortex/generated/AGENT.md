# AGENT.md — merlin/targets/vortex/generated

## Purpose

Per-target scratch output dir for `vortex` (gitignored except `.gitkeep`/`AGENT.md`).

## What belongs here

- Throwaway scratch a local tool writes while working on this target.

## What does not belong here

- Generated codegen packages (dialects/schedules/OOT builds) → `out/artifacts/targets/vortex/`.
- Run output, results, figures → `out/runs/` / `out/artifacts/` (see `.claude/skills/artifact-layout`).
- RTL scratch (`*.hw.mlir`, `*.ll`, arcilator bins) → the purgeable
  `out/artifacts/cache/rtl_introspect/vortex/`. Only a promoted `contracts/rtl_facts/facts.json` is kept.

## Invariants

- Nothing here is tracked, so nothing here may be load-bearing for a fresh clone.
