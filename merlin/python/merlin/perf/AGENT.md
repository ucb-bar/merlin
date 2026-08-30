# AGENT.md — merlin/python/merlin/perf

## Purpose

The performance layer: what a target's legal choices *cost*. Derives archetypes and traits, emits a
performance contract whose terms carry provenance and a validity domain, composes a predicted cycle
count with gap attribution, and classifies workloads by which lever has headroom.

## The rule that governs everything here

Every module is a **generic, trait-gated analysis**. Gating is on **derived traits**, never on an
archetype name and never on a target name — an archetype is only a prior (it decides *which questions
to ask*); the RTL-derived traits decide *which of them apply*. The target is a parameter (`target=`)
threaded from the descriptor/manifest.

A tool that only works where it was written is manual overfitting with extra steps. The bar: the same
code runs on two targets of different archetypes and produces **different, correct** answers.

## What does not belong here

- Target-name literals of any kind. `check_no_target_name.py` scans this tree, and its advisory
  `--coupling` pass also flags imports and substring symbol hits in any file whose own path does not
  name a target — which `perf/` never will. **Do not add allowlist entries**; that list only shrinks.
- `import re`. Parse structurally.
- Assumed geometry, capacities, opcodes or latencies. Derive them, or record `UNKNOWN` and fail
  closed. `UNKNOWN` is a distinct inhabited state and must never be readable as `0.0`.
- Generated output. Products go to `out/artifacts/` via `merlin.common.paths` / `artifacts` helpers.

## Invariants

- **Never default a composition operator.** Textbook roofline takes `max` of compute and memory time,
  which assumes perfect overlap; a target that does not overlap sums them instead. Deriving `max` where
  the truth is `sum` understates runtime badly, and in the flattering direction.
- **Prefer moved bytes to algorithmic bytes.** Transfer amplification is real and large; a bound built
  on the bytes an algorithm needs rather than the bytes a program moves is optimistic by that factor.
- **Fixed terms are first-class.** Pipeline fill and drain are intercepts, not rates. A rate-only model
  mispredicts every small workload.
- **At least two points per fitted parameter.** A single rate cannot price a unit whose cost is a rate
  plus a fixed overhead.
- Tests go in an existing bucket — there is no `perf` bucket and the list is an enum. Contract, record
  and profile tests live in `merlin/tests/targetgen/`; envelope, attribution and analysis tests in
  `merlin/tests/dse/`. Resolve paths via `merlin.common.paths.repo_root()`.

## Where the task register lives

`merlin/experiments/performance_contract/TASKS.md` — every task, its state, and its blocker.
Rationale for the cost and oracle decisions: `docs/design/performance_budget_unit.md`.
