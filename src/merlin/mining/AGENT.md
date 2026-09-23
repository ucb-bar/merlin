# AGENT.md — merlin/python/merlin/mining

## Purpose

Compiler/runtime schedule-package primitives retained in core during staged cutover.
Search, agents, fork creation, campaigns and reports belong to `packages/merlin-mining`.
The single core-owned initializer extends its package path, preserving historical module names.
See `packages/merlin-mining/README.md` for the ownership table and remaining extraction work.

## Modules

- `apply.py` — Apply an RVV package's codegen knobs to a workload build, via the existing build_app seam.
- `from_strategy.py` — Render a transform-dialect RVV schedule FROM knobs, and mint a versioned fork package.
- `registry.py` — Loader for ISOLATED, per-run RVV codegen packages.

The other `merlin.mining.*` implementations live in the optional mining distribution;
their stable import names do not imply core ownership. The `from_strategy.mint_fork`
compatibility operation still requires that distribution; schedule rendering does not.

<!-- Purpose/Modules derived from docstrings via build_tools/scripts/gen_package_docs.py.
     Add hand-written notes (invariants, gotchas) below. -->
