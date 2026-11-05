# AGENT.md — packages/merlin-experiments/src/merlin_experiments/tracking/console/sources

## Purpose

Console sources: one bounded, read-only reader per concern, each ``read(ctx) -> section``.

## Modules

- `agents.py` — The author agent tree: who exists, what each is doing, its tokens and its activity spans.
- `attempts.py` — Every attempt of the run's campaign: when it started and ended, why it stopped, and the time it cost.
- `case_history.py` — Per-test-case performance history: every passing observation of each case, by evidence kind, in time order.
- `cases.py` — Per-test-case breakdown: status, cycles by evidence kind, roofline gap, utilization and gain since first pass.
- `capsules.py` — The capsule catalog: what each capsule is (op, shapes, origin, cohort) and its latest recorded result.
- `certification.py` — Certification coverage: for every cohort, how many test cases hold each tier and why the rest do not.
- `cost.py` — What the run costs in tokens, per hour and per newly certified test case.
- `events.py` — One time-ordered feed of what happened: agent messages and failures, checks, simulations, construction actions, round ends, stops and the launch log's summary lines.
- `focus.py` — What the author agent is working on now: the files it edits, the operation families they touch, what is open.
- `golden.py` — Operator-only comparison of whole-model results against the reference ledger, refusing what is not comparable.
- `grader.py` — Grader pipeline health: how fast verdicts arrive, how many are delivered, what holds grades back.
- `insight_common.py` — Shared pieces of the insight panels: the panel shape, the campaign chain, the agent activity and log tails.
- `integrity.py` — Run integrity and infrastructure: errors the harness logged, broker and grade health, store, disk, neighbours.
- `jobs.py` — Jobs and queues of one experiment: the Gantt spans, stage-duration statistics and simulation queue.
- `library.py` — Which library passes the author's compiler uses, how often the author touched each, and what A/Bs measured.
- `machine.py` — Every CPU of the machine: its utilization now and who it is allocated to.
- `passes.py` — The pass matrix: what the library offers, what the agent's compiler declares and runs, and what it costs.
- `perf.py` — Per-capsule performance: Spike-model estimate, gSIM cycles, structural floor, intensity and modeled activity.
- `performance.py` — Measured performance over time: gSIM cycles per test case, speedups, utilization, movers and model gaps.
- `progress.py` — How far the experiment has come: grade progress, cohort coverage, the passing count over time.
- `resources.py` — Host resources of one experiment: per-core activity while it runs, process-tree history always.
- `stages.py` — Compiler-construction stages of one experiment, from a stage ledger or derived from existing records.
- `timeline.py` — Where the wall time went: critical-path attribution, parallelism and phase markers.
- `tokens.py` — Cumulative token use across every author agent: totals, a time series, rates, budget and limits.
- `tools.py` — Which of its tools the author agent actually uses: diagnostics, performance feedback, helpers.

<!-- Purpose/Modules derived from docstrings via build_tools/scripts/gen_package_docs.py.
     Add hand-written notes (invariants, gotchas) below. -->
