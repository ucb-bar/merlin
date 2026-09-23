# AGENT.md — src/merlin

## Purpose

The canonical, installed `merlin` Python package. See the repository ownership map and
`docs/reference/architecture.md` before introducing a new dependency.

## What belongs here

- Shared compiler, scheduling dialects, capture contracts, target/toolchain resolution,
  generic runtime and verification primitives.

## What does not belong here

- Research orchestration, study-only analysis, agent grading or target-specific implementations.
- Generated artifacts (use `merlin.common.paths` and the configured `out/` root).

## Invariants

- Optional distributions may extend declared namespace paths but must not overwrite core files.
- Keep shared primitives independent of optional workflows and heavy frontend imports.
- Every subdirectory must also contain an AGENT.md.
