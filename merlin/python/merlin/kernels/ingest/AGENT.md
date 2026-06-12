# AGENT.md — merlin/python/merlin/kernels/ingest

## Purpose

Ingest kernels from XNNPACK/Autocomp/Triton/Exo/generic sources.

## What belongs here

- Files appropriate to the purpose above.

## What does not belong here

- Vendored external kernel repos (pass by path / `MERLIN_<SRC>_REPO`, never vendor).
- Generated artifacts (write those to `output/`).

## Invariants

- Keep this directory focused on its stated purpose.
- Every subdirectory must also contain an AGENT.md.
- Per-kernel extraction stays deterministic (regex/filename/AST), zero LLM calls; reusable logic lives here, not in `experiments/`.
