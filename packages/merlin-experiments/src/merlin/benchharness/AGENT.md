# AGENT.md — packages/merlin-experiments/src/merlin/benchharness

## Purpose

Shared, target-parametric benchmark-harness primitives.

Public standalone clients live in `merlin_experiments.phase1.tools`, not here.
`merlin.targetgen.tool_registry` owns their exact module/staged-name mapping;
resolve sources with `module_source_path()` and copy only declared client files.
Never stage this package into candidate workspaces.

`chia_tasks.py` owns only references returned to its caller. Use it inside `chia_run` so
bounded cancellation observation precedes collector teardown. Preserve partial child results,
never retry paid work implicitly, and never cancel borrowed cluster peers. A terminal Ray task
does not prove its native subprocess descendants stopped; record that qualification separately.

<!-- Purpose/Modules derived from docstrings via build_tools/scripts/gen_package_docs.py.
     Add hand-written notes (invariants, gotchas) below. -->
