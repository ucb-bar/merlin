# AGENT.md — src/merlin

## Purpose

The canonical, installed `merlin` Python package. See the repository ownership map and
`docs/reference/architecture.md` before introducing a new dependency.

## What belongs here

- Shared compiler, scheduling dialects, capture contracts, target/toolchain resolution,
  generic runtime and verification primitives.
- Research orchestration, analysis and grading under their existing import identities;
  their dependencies and trusted access roles remain separate from compiler mechanisms.

## What does not belong here

- Target-specific implementations, independently owned simulators and evaluators.
- Generated artifacts (use `merlin.common.paths` and the configured `out/` root).

## Invariants

- One distribution owns every active Merlin implementation; no duplicate source trees.
- Wheel membership never grants candidate access to graders, references or hidden inputs.
- Keep shared primitives independent of optional workflows and heavy frontend imports.
- Every subdirectory must also contain an AGENT.md.
