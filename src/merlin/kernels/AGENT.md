# AGENT.md — merlin/python/merlin/kernels

## Purpose

Kernel abstraction mining: ingest -> features -> emit.

## What belongs here

- Files appropriate to the purpose above.
- Generic analyses that public compiler code needs (contraction shapes and extents, the role
  vocabulary, engine facets, endpoint binding, RoCC/intrinsics decoding, the disassembly
  tokenizer) live in `merlin.compile.analysis`; `shapes.py`, `roles.py`, `engines.py`,
  `endpoints.py` and `decode/{rocc,objdump,insn_header}.py` here are compatibility aliases.
  Do not add new logic to an alias; this package is withheld from a full-compiler author.

## What does not belong here

- Vendored external kernel repos (pass by path / `MERLIN_<SRC>_REPO`, never vendor).
- Generated artifacts (write those to `runs/` or `artifacts/`).

## Invariants

- Keep this directory focused on its stated purpose.
- Every subdirectory must also contain an AGENT.md.
- Per-kernel extraction stays deterministic (regex/filename/AST), zero LLM calls; reusable logic lives here, not in `experiments/`.
