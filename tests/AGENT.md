# AGENT.md — tests

## Purpose

The compact permanent behavioral suite for Merlin. The complete owned repository
has a maximum of 150 collected cases, including parameterization. Active cases
live here; other project roots may not bypass the global collection budget.
Shared test fixtures/data live in `fixtures/` and `data/` here.

## What belongs here

- `test_*.py` (the suite), shared `fixtures/` and `data/` consumed by tests.

## What does not belong here

- Library/application source (`src/merlin/` or `packages/*/src/`).
- Generated outputs (those belong under the configured `out/` root).

## Invariants

- Resolve repo paths via `merlin.common.paths.repo_root()` for checkout fixtures and resource-specific resolvers for installed inputs, never `__file__` parents
  (so tests are location-independent).
- Run: `.venv/bin/python -m pytest tests`.

## Test review bar

The permanent suite protects distinct behavior and independent acceptance
obligations. A new helper or option does not automatically need a new test.

- Run `python build_tools/checks/check_test_budget.py --output out/build/python/ci/test-budget.json`
  before execution. The ceiling applies to actual cases, not files or functions.
- Replace overlapping coverage when adding an obligation at the ceiling. Do not
  bypass the budget with markers, skips, large assertion loops or hidden collectors.
- Runtime verification algorithms remain production code under
  `src/merlin/verification/` and the independent evaluation owners. Tests protect
  their behavior; test pruning must not relax numerical or evidence admission.

- Prefer a small real program through a public compiler/runtime boundary, with
  an independent numerical reference or a check of admitted IR, effects, memory,
  ABI or artifacts. Retain meaningful quantization, shape-tail and alias cases.
- Keep the smallest negative witness for each distinct refusal or integrity
  obligation. Source custody, evaluator isolation and a gate's actual failing
  direction are important behavior even when they perform no numerical work.
- Parameterize only materially different semantics or failure paths. Do not
  expand flag combinations, repeat expensive setup, or move redundant cases
  into a large parameter table merely to reduce the function count.
- Avoid assertions about private call counts, source spelling, prose, incidental
  filenames or complete JSON snapshots. Keep such assertions only when that
  exact property is a documented interface or source-integrity obligation.
- Extend an existing behavior test before adding another module. Fixtures
  should model the real producer contract; do not patch the function under test
  into returning the expected answer or fabricate missing acceptance evidence.
- Remove obsolete or overlapping coverage with an explicit replacement or
  retired-obligation rationale. A failing test is not itself obsolete. Historical
  replay uses its frozen sources and inputs; the current factory need not recreate
  retired releases. Preserve immutable paper inputs separately.
- Missing external prerequisites remain explicit. Do not add blanket skips,
  exclusions or weaker acceptance rules to obtain a green run.
