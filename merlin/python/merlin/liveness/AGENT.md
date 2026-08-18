# AGENT.md — merlin/python/merlin/liveness

## Purpose

HW-agnostic *liveness / progress* oracle — an L2.5 tier between functional (L2) and RTL (L3).

## Modules

- `facts.py` — The single derivation seam for the liveness oracle: normalize a target's CIRCT/mlc-introspected
- `interconnect.py` — (A) Dynamic transaction-level liveness model.
- `oracle.py` — The unifying entry point: assess a program's silicon liveness against a target, merging the static
- `preconditions.py` — (B) Static silicon-precondition linter.
- `report.py` — Shared result types for the liveness oracle: a severity-ranked :class:`Finding` and the aggregate

<!-- Purpose/Modules derived from docstrings via build_tools/scripts/gen_package_docs.py.
     Add hand-written notes (invariants, gotchas) below. -->
