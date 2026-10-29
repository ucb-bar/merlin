---
title: Capsule roles in correctness and performance evaluation
kind: design
status: current
owner: core
last_verified: 2026-10-12
related: [generating_capsules, phase2_test_justification, normal_search_feedback, performance_budget_unit]
code_refs: [src/merlin/targetgen/phase_policy.py, src/merlin/targetgen/cert_cost.py, experiments/templates/phase0/performance.yaml]
---

# Capsule roles in correctness and performance evaluation

Phase 1 establishes functional compilation for the admitted domain. Phase 2
improves its performance while preserving that domain and numerical contract.
A capsule may serve either purpose or both; its size alone does not establish
its role.

## Admission predicates

`phase_policy.certifiable` checks whether the selected correctness tier and
measured evaluation domain can cover the capsule within the supplied budget.
Missing engine, unsupported extent or absent cost evidence remains explicit.
A predicted affordable test has not executed and supplies no correctness result.

`phase_policy.priceable` checks whether the capsule poses a measurable
performance question. Its default requires derivable nonzero MAC work. A
component with genuinely zero MACs can instead receive a typed
`PerformanceObjective` specifying the metric, unit, direction, complete cost
boundary and provenance. Unknown work cannot be repaired by supplying a metric.
This allows movement, conversion and dispatch questions without invented MAC
utilization.

`phase_of`, `split_report` and `anchors` retain these independent predicates.
They classify eligibility; they do not issue numerical, timing or convergence
qualification. Unknown eligibility remains distinct from a demonstrated refusal.

## Choose tests for the property

- Functional tests exercise arithmetic, encoding, tails, reduction state,
  initialization, output coverage, resource boundaries and composition.
- Performance tests retain enough work and pressure to expose the declared
  bottleneck. Their purpose, comparator and falsifier are recorded during Phase 0.
- Larger compile-only cases inspect scaling, code size and interfaces without
  supplying numerical or timing evidence.
- Connected public programs test interactions that isolated kernels cannot
  establish. Protected validation programs remain outside authoring inputs.

A small case can reveal a correctness error while failing to expose a transfer
or residency bottleneck. A large case can expose that bottleneck while being
expensive to simulate. Use the documented fidelity ladder and qualified models
to plan development checks; retain all required promotion and final gates.

## Interpret coverage

Report member counts separately from witnessed semantic and interaction
obligations. Matching a shape or arithmetic form does not prove graph placement,
whole-program execution or time share. Unknown support and ungenerated required
families remain visible in the coverage report.

See [test justification](../guides/phase2_test_justification.md) for derivation
and representative selection, and [normal search feedback](normal_search_feedback.md)
for selective development evaluation and complete measured selection.
