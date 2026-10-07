# AGENT.md — src/merlin/common

## Purpose

Shared utilities: schemas, IO, source/resource paths, storage and access identities.
The common access registry contains only target-neutral identities; the experiments
sandbox extends it with mandatory packaged historical target denials before launch.
`frozen_imports` provides stdlib-only, process-local source import isolation for
trusted bootstraps. Callers own snapshot seals, invocation records and grading.
`source_membership` inventories ordinary Python source files for existing run
receipts; it owns no seal, target policy or optional-distribution dependency.
`ir_audit` owns opt-in exact-byte named-stage inspection records. Lowering callers
own stage serialization and explicit destination and sidecar selection; these observations
are not compiler certificates or a replacement for frozen-source seals.
Caller-produced compact stage views are explicitly non-executable and bind the exact
parent content hash; printer/framework dependencies remain outside this common utility.
Large xDSL dense inspection payloads are stored as exact raw bytes under content-addressed
`tensors/` files within one audit. Stage descriptors bind producer element types and shapes;
identical bytes share storage across stages. Completion rechecks generated tensor hashes.
This is inspection storage, not safetensors conversion or an executable reconstruction ABI.
`compile_trace` is the user-facing debugging switch over the same recording points: a request
(in `MERLIN_COMPILE_TRACE`, so child processes see it) names stages to dump and one to stop after.
Pipelines declare their stage names beside the code that records them; never list them here. Only
the session's own process raises the stop (`StopAfterStage`, a `BaseException`), so a stop is
never swallowed as a per-group failure.

## What belongs here

- Files appropriate to the purpose above.

## What does not belong here

- Workstream-specific logic (this is shared infrastructure only).
- Generated artifacts (write those to `runs/` or `artifacts/`).

## Invariants

- Keep this directory focused on its stated purpose.
- Every subdirectory must also contain an AGENT.md.
- Shared helpers (schema load/validate, yaml, llm summary) are real and dependency-light.
- CAS-backed snapshots share file modes. Before deleting a read-only bundle, use
  its snapshot cleanup owner; bare TemporaryDirectory cleanup can chmod shared
  files after an unlink failure and invalidate other sealed consumers.
- Frozen imports never fall through to a live owner or unchecked bytecode. This
  provenance boundary is not a Python sandbox and does not propagate to subprocesses
  without an explicit bootstrap. Keep experiment-specific launch policy out of core.
