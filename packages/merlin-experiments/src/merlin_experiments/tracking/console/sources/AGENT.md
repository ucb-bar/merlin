# AGENT.md — packages/merlin-experiments/src/merlin_experiments/tracking/console/sources

## Purpose

Console sources: one bounded, read-only reader per concern, each ``read(ctx) -> section``.

## Modules

- `agents.py` — The author agent tree: who exists, what each is doing, its tokens and its activity spans.
- `capsules.py` — The capsule catalog: what each capsule is (op, shapes, origin, cohort) and its latest recorded result.
- `events.py` — One time-ordered feed of what happened: agent messages and failures, checks, simulations, construction actions, round ends, stops and the launch log's summary lines.
- `golden.py` — Operator-only comparison of whole-model results against the reference ledger, refusing what is not comparable.
- `jobs.py` — Jobs and queues of one experiment: the Gantt spans, stage-duration statistics and simulation queue.
- `machine.py` — Every CPU of the machine: its utilization now and who it is allocated to.
- `passes.py` — The pass matrix: what the library offers, what the agent's compiler declares and runs, and what it costs.
- `perf.py` — Per-capsule performance: Spike-model estimate, gSIM cycles, structural floor, intensity and modeled activity.
- `progress.py` — How far the experiment has come: grade progress, cohort coverage, the passing count over time.
- `resources.py` — Host resources of one experiment: per-core activity while it runs, process-tree history always.
- `stages.py` — Compiler-construction stages of one experiment, from a stage ledger or derived from existing records.
- `timeline.py` — Where the wall time went: critical-path attribution, parallelism and phase markers.
- `tokens.py` — Cumulative token use across every author agent: totals, a time series, rates, budget and limits.

<!-- Purpose/Modules derived from docstrings via build_tools/scripts/gen_package_docs.py.
     Add hand-written notes (invariants, gotchas) below. -->
