# AGENT.md — tools/merlin-search

## Purpose

Entrypoint: run grid / evolutionary / MAP-Elites search over candidate artifacts.

## What belongs here

- README documenting intent; later, a thin CLI delegating to `merlin.search`.

## What does not belong here

- CLI logic now (scaffold).
- New search methods beyond the three (out of scope).

## Invariants

- Stay thin; delegate to `merlin.search`. Artifacts go under `output/`.
