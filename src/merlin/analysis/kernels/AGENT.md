# src/merlin/analysis/kernels

## Purpose

Kernel abstraction mining: ingest -> features -> emit.

## What belongs here

- Files appropriate to the purpose above.
- Generic analyses that public compiler code needs (contraction shapes and extents, the role
  vocabulary, engine facets, endpoint binding, RoCC/intrinsics decoding, the disassembly
  tokenizer) live in `merlin.analysis`. This package is withheld from a full-compiler author.

## What does not belong here

- Vendored external kernel repos (pass by path / `MERLIN_<SRC>_REPO`, never vendor).
- Generated artifacts (write those to `runs/` or `artifacts/`).

## Invariants

- Keep this directory focused on its stated purpose.
- Every subdirectory must also contain an AGENT.md.
- Per-kernel extraction stays deterministic (regex/filename/AST), zero LLM calls; reusable logic lives here, not in `experiments/`.

## Research workflow responsibilities

Optional measurement controllers contribute to the core-owned `merlin.analysis.kernels` namespace.
Do not add a competing namespace initializer or duplicate compiler/runtime implementations.
Native driver resources remain in core when runtime backends consume the same bytes.
