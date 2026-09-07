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

## Two caches, and why they are different things

A grade reuses work at two levels, and conflating them is the mistake to avoid.

- `build_cache.py` — **a BUILD cache.** An ELF is a pure function of the emitted program, the command
  buffer, the operands, the recipe, the toolchain and the code that compiles them; all of that is in
  the key. A restored build is still executed and still judged, so the worst a wrong hit can do is run
  the wrong program. It restores the whole generated directory, not just the executable, because the
  agent reads what is in it.
- `tier_cache.py` — **a VERDICT cache.** It carries a tier's *result* for bytes already certified, so a
  wrong hit asserts an unearned verdict. Every rule in it is written in the direction of re-running,
  and it must stay that way.

Measured on `merlincirct_g4p1_biasabi_20260906` (819 screen-tier executions): the screen spent 4.06 s
median building against 0.153 s simulating — 27x more compiling than simulating, 3,981 s of build in
one run. That is why the build cache exists, and why relaxing the verdict cache to chase the same
saving would have been the wrong trade (carrying a screen saves 0.153 s).

Provenance is scoped, not cached: `merlin.common.provenance.observation_scope`, opened by `run_suite`,
makes one grade see one revision per checkout. That is a correctness property first — two capsules in
one grade must not be attributed to different hardware revisions — and a large speedup second.

## Testing expectations

- `merlin/python/tests/test_targetgen_toy.py` must pass (toy_npu build + inspect + simulate).

## Notes for future agents

- The reference target is `toy_npu` (concrete). gemmini/saturn/radiance are conservative
  keyword-detected skeletons until human review fills them in.
