---
title: Performance objectives and evaluation costs
kind: design
status: current
owner: core
last_verified: 2026-10-12
related: [capsule_phase_split, normal_search_feedback, phase2_cpu_timing_study]
code_refs: [src/merlin/targetgen/phase_policy.py, src/merlin/targetgen/cert_cost.py, packages/merlin-experiments/src/merlin_experiments/phase2/search_priorities.py]
---

# Performance objectives and evaluation costs

Keep three quantities separate: the cost of executing the compiled program,
the host cost of evaluating a candidate, and the remaining search budget.
Guest cycles cannot price compilation or simulator wall time. A fast functional
simulator cannot establish target timing.

## Define the objective

For contraction work, `phase_policy.priceable` requires derivable nonzero MACs.
A zero-MAC component may instead select `PerformanceObjective`: an observable
metric, unit, direction, complete cost boundary and provenance. Unknown work
remains refused. The objective admits a question; it supplies no measured cost,
throughput target or timing certificate.

State whether the boundary includes preparation, packing, dispatch,
synchronization and readback. Inclusive timers and contained stages cannot be
added without accounting for their overlap. Cold and warm measurements retain
their distinct contracts.

## Measure evaluation cost

Record the actual route, compiler and input identities, toolchain, engine,
resource allocation and completion scope. Include required setup, compilation,
observers and proven cleanup in a complete action cost. Keep queue wait distinct
from action service time. Timeout, cancellation or unknown cleanup cannot supply
a successful complete evaluation cost.

Do not divide a concurrent batch wall time into per-member costs, extrapolate a
small measured shape without a validated model, or compare costs from different
engines and resource configurations as if they were interchangeable.

The [CPU timing study](../guides/phase2_cpu_timing_study.md) records route costs
and empirical screening validation. Its whole-roster observations do not
establish per-member costs. Normal search priorities leave evaluation cost and
savings per evaluation second unknown until matched complete observations exist.

## Spend the budget

Use cheap compilation and numerical checks for development questions. Escalate
when a ranking depends on unknown physical service, overlap or an unqualified
regime. Qualified predictions can screen alternatives; complete measured
selection and final correctness remain separate gates.

Prioritize weighted potential savings against an unrefuted bound or matched
throughput observation. Distance from roofline is an upper bound on possible
savings, not a promise of attainable performance. Reserve time and resources for
complete confirmation and final evaluation before stopping exploration.
