# AGENT.md — merlin/experiments

## Purpose

Throwaway/bookkeeping experiments for the three workstreams. Each experiment has a `README.md` and `AGENT.md` describing its question and how to reproduce it.

## What belongs here

- Experiment configs, notes, and small reproducible drivers.
- `targetgen_toy/`, `kernel_policy/`, `semantic_memory/`, `interface_dse/`.

## What does not belong here

- **Reusable library code** — that belongs in `merlin/python/merlin/`.
- Large generated results — write those to gitignored `output/`.

## Interfaces

Consume schemas and modules; emit artifacts to `output/`.

## Invariants

- Experiments must not become library code.
- Reusable logic must be lifted into `merlin/python/merlin/`.

## Testing expectations

Each experiment documents how to reproduce its result; no heavy CI.

## Notes for future agents

Keep experiments small and self-describing so other sessions can rerun them.
