# AGENT.md — src/merlin/mining

## Purpose

Compiler/runtime schedule-package primitives retained in core during staged cutover.
Search, agents, fork creation, campaigns and reports belong to `merlin.mining`.
One distribution owns these modules; compiler primitives keep their logical dependency boundary.
Retained workflow tests and historical ownership notes remain in `packages/merlin-mining/`.

## Modules

- `apply.py` — Apply an RVV package's codegen knobs to a workload build, via the existing build_app seam.
- `from_strategy.py` — Render a transform-dialect RVV schedule FROM knobs, and mint a versioned fork package.
- `registry.py` — Loader for ISOLATED, per-run RVV codegen packages.

The other `merlin.mining.*` implementations live in the optional mining workflow;
their stable import names do not imply core ownership. The `from_strategy.mint_fork`
compatibility operation still selects the workflow dependencies; schedule rendering does not.

<!-- Purpose/Modules derived from docstrings via build_tools/scripts/gen_package_docs.py.
     Add hand-written notes (invariants, gotchas) below. -->

## Research workflow responsibilities

Search, agents, campaigns, fork creation, measurement orchestration, and reports live here.
The compiler-facing initializer and runtime primitives keep their logical owner. Preserve
historical module names and class identity; do not create another implementation tree.
