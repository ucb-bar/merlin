# AGENT.md — experiments/studies/performance_contract

Status: active

## Purpose
Measurement scripts and validation requirements for the **performance layer**: deriving what a target's legal
choices *cost*, alongside the existing machinery that derives what it *permits*. Spine target is
Atlas; Radiance and Gemmini are fan-out.

`TASKS.md` describes input identities, measurement checks and reporting requirements.

## What lives here
- `TASKS.md` — the validation checklist for measured performance studies.
- Target-specific measurement scripts and method specifications.

## What does NOT belong here
- **Reusable library code** → `merlin/python/merlin/perf/` (and machine-side primitives stay in mlc).
- **Generated output** → `out/artifacts/` (products) or `out/runs/<target>/<suite>/` (runs). Never
  in-tree; the `check_artifact_layout` gate forbids `experiments/*/reports/` and `experiments/*/runs/`.
- **Durable rationale** → `docs/design/`. The cost decisions live in
  `docs/design/performance_budget_unit.md`, not here.

## Invariants
- **A task is not `DONE` because code exists.** `DONE` means implemented *and* verified by a test or a
  measured run. Source availability alone is not evidence of numerical or timing correctness.
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
without the scaling law being measured across both regimes. See `TASKS.md` for the required checks.
