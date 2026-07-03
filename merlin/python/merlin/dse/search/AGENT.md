# AGENT.md — merlin/python/merlin/search

## Purpose

The search layer. Explores candidate compiler artifacts to find what is worth keeping. Exactly three methods: grid, evolutionary, MAP-Elites.

## What belongs here

- candidate/evaluator/archive abstractions and the three search loops (grid, evolutionary, map_elites), mutation operators, and report emission.
- Search spaces described by `search_space.schema.yaml`.

## What does not belong here

- A `merlin.search` dialect — search is orchestration/experiment logic, not IR.
- Beam search, MCTS, Bayesian opt, or a generic AutoML framework (deliberately out of scope).
- Generated artifacts (write to `runs/` or `artifacts/`).

## Interfaces

Scores candidates via `evaluator.py`, which delegates compilation+measurement to `merlin.dse.harness`. Consumes `search_space` + candidate-type schemas (`compilation_strategy`, `policy_rule`, `dialect_plan`, `interface_candidate`). Emits `dse_result` / `exploitability_report` + scoreboards.

## Invariants

- Three methods only: grid (explicit sweeps), evolutionary (improve a candidate), MAP-Elites (preserve many good families). Do not add other search methods without justification.
- Scoring prioritizes correctness > compile_success > coverage > exploitability > speedup.
- LLMs may be mutation/repair operators, never 'the search method'.

## Testing expectations

Search-loop unit tests should use a fake evaluator (deterministic scores) so they do not depend on a working compiler/simulator.

## Notes for future agents

Implement in phases: (1) grid, (2) evolutionary over YAML artifacts, (3) MAP-Elites archive. Keep candidate lineage/logs. See docs/search.md.
