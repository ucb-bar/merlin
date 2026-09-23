# AGENT.md — merlin

## Purpose

Retained contracts, target inputs, native experiment engines and cross-subsystem tests.
The installed compiler core is owned by `src/merlin/`, not this directory.

## What belongs here

- `contract/` owns shared schemas, capsule contracts and reviewed project policy.
- `experiments/` retains native phase engines and research inputs; start experiment
  discovery at the root `experiments/catalog.yaml`, not by choosing a script here.
- `targets/` retains target sources whose out-of-tree removal is not yet qualified.
- `tests/` owns cross-subsystem regression tests; optional distributions also own
  focused tests under `packages/*/tests/`.
- `python/merlin` is a compatibility symlink to the canonical core, not another
  implementation tree. Edit core files through `src/merlin/`.

## What does not belong here

- Reusable compiler implementation belongs in `src/merlin/`; optional execution,
  analysis, mining and DSE belong in their owning `packages/*/src/` distribution.
  Phase orchestration belongs in `packages/merlin-experiments/src/merlin_experiments/`.
- Generated outputs belong under the configured `out/runs/`, `out/artifacts/` or
  `out/build/` roots, selected through `merlin.common.paths`. The old top-level
  `runs/`, `artifacts/` and `build/` destinations are retired.

## Invariants

- Keep compiler primitives independent of experiment workflows and target identities.
- Follow the local `AGENT.md` and root `CLAUDE.md`; do not relocate private grading
  inputs into public package resources or rewrite historical run evidence.
- See `docs/reference/repo_structure.md` for ownership and remaining migration
  limits, and `docs/guides/storage.md` for generated-output retention policy.
