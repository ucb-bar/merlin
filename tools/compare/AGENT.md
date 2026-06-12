# AGENT.md — tools/compare

## Purpose

Entrypoint: run the (workload x strategy) matrix and emit a scoreboard + decision report.

## What belongs here

- README documenting intent; later, a thin CLI delegating to `merlin.dse.harness`.

## What does not belong here

- CLI logic now (scaffold).
- Search loops (that is `search` / `merlin.search`).

## Invariants

- Stay thin; delegate to the backing module. Artifacts go under `output/`.
