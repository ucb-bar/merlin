# AGENT.md — merlin/experiments/performance_contract

## Purpose
The task register and working notes for the **performance layer**: deriving what a target's legal
choices *cost*, alongside the existing machinery that derives what it *permits*. Spine target is
Atlas; Radiance and Gemmini are fan-out.

`TASKS.md` is the register — every task, its state, and the blocking order. Read it before starting
work here, and update the state in the same commit that changes it.

## What lives here
- `TASKS.md` — the task register (`DONE` / `PARTIAL` / `OPEN`, with blockers named).
- Method specs and working notes for this workstream.

## What does NOT belong here
- **Reusable library code** → `merlin/python/merlin/perf/` (and machine-side primitives stay in mlc).
- **Generated output** → `out/artifacts/` (products) or `out/runs/<target>/<suite>/` (runs). Never
  in-tree; the `check_artifact_layout` gate forbids `experiments/*/reports/` and `experiments/*/runs/`.
- **Durable rationale** → `docs/design/`. The cost decisions live in
  `docs/design/performance_budget_unit.md`, not here.

## Invariants
- **A task is not `DONE` because code exists.** `DONE` means implemented *and* verified by a test or a
  measured run. Three numbers in this tree looked settled and were wrong; the register exists to keep
  that visible.
- **A check that could not run is `not_run`** — never a pass, never a zero.
- **Report a per-query cost with its concurrency.** A 16-worker grade inflated the same arc query
  6.3× over its serial latency; a cost measured under parallelism is a throughput figure wearing a
  latency figure's clothes.
- **Cycles belong to the submission, not the capsule** (an 8.2× spread was measured on identical
  inputs). Freeze the capsule *set*; never freeze a cycle number.
- Every Atlas number carries a **source digest** — those pins report permanent drift by design, so a
  commit sha alone is not provenance.

## Scale warning
The whole Atlas capsule corpus is toy-scale — the largest tensor anywhere is 5400 elements. Any cost,
utilization or speedup measured on it describes 32×32 tiles and does **not** extrapolate to layers
without the scaling law being measured across both regimes. See `TASKS.md` N1, the hard blocker.
