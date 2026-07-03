# AGENT.md — merlin/python/merlin/kernels

## Purpose

Kernel abstraction mining: ingest -> features -> emit.

## What belongs here

- Files appropriate to the purpose above.

## What does not belong here

- Vendored external kernel repos (pass by path / `MERLIN_<SRC>_REPO`, never vendor).
- Generated artifacts (write those to `runs/` or `artifacts/`).

## Invariants

- Keep this directory focused on its stated purpose.
- Every subdirectory must also contain an AGENT.md.
- Per-kernel extraction stays deterministic (regex/filename/AST), zero LLM calls; reusable logic lives here, not in `experiments/`.
