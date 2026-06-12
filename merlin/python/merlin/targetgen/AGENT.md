# AGENT.md — merlin/python/merlin/targetgen

## Purpose

TargetGen pipeline: ingest -> evidence -> synthesize -> generate -> validate. Turns a
target's local docs/examples/source into human-reviewable plans and a generated
`merlin-target-<name>/` repo skeleton.

## What belongs here

- `cli.py` / `pipeline.py` — the deterministic, LLM-free entry points.
- `ingest/` — record inputs as a SourceManifest (no crawling, no vendoring).
- `evidence/` — deterministic file discovery + keyword concept detection.
- `synthesize/` — emit the five plans (target_contract / dialect / runtime_adapter / zephyr / llvm).
- `generate/` — write the target-repo skeleton + xDSL / MLIR / runtime adapter / Zephyr / LLVM.
- `validate/` — schema + structural checks and the validation report.

## What does not belong here

- LLM/API dependencies — TargetGen is deterministic.
- Claims of automatic correctness. Outputs are human-reviewable; non-toy synthesis is
  flagged `requires_human_review: true`.
- The Merlin core dialects or runtime ABI (those live elsewhere in the repo).

## Interfaces

- Produces artifacts that validate against `merlin/schemas/*.schema.yaml`.
- `pipeline.build(...)` returns a `BuildResult`; `cli.py` exposes `build` and `inspect`.
- Consumes `merlin.common` (paths/io/yaml/artifacts/schemas) and `merlin.validation`.

## Invariants

- Deterministic: same inputs -> byte-identical YAML artifacts.
- Targets implement adapters; TargetGen never generates an independent runtime model.
- Do not generate dialect ops directly from instruction names — synthesis stays conservative.
- Every subdirectory contains an AGENT.md.

## Testing expectations

- `merlin/python/tests/test_targetgen_toy.py` must pass (toy_npu build + inspect + simulate).

## Notes for future agents

- The reference target is `toy_npu` (concrete). gemmini/saturn/radiance are conservative
  keyword-detected skeletons until human review fills them in.
