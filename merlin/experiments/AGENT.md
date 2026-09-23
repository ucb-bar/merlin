# AGENT.md — merlin/experiments

## Purpose
Retained native experiment engines, benchmark harnesses and research inputs.
Start with the root `experiments/catalog.yaml` and `experiments/README.md` for
declarative phase selection. This directory is not a second experiment registry.
The native engines consume canonical core and optional package owners; their
remaining checkout dependencies mean that moving or deleting them requires
consumer and source-receipt qualification, not just import checks.

- Small workstream experiments: `kernel_policy/`, `gemmini_cert/`.
- Benchmark harnesses: `capsule_bench/` (the multi-target capsule benchmark — six targets, one
  harness), `agent_bench/` (target-agnostic reference scaffold), `gemmini_perf_bench/`,
  `muon_perf_bench_v0/`, `targetgen_evals/` (import-isolated; 0 real runs — do not cite it).
- Cross-compiler studies: `llm_kernel_vs_compiler_v0/`, `voyager_h2h/` (merlin vs the Voyager
  compiler, same-hardware and home-turf planes), `dataset_accuracy/` (full-validation-set accuracy;
  first milestone reproduces Voyager's Table 3 ImageNet cells with Voyager's own quantizer).

## Status (enforced)
Every experiment's AGENT.md carries a `Status:` line in its first 15 lines: `active`, `frozen`
(finished; its results are in `FINDINGS.md`, which must exist) or `reference` (a scaffold other
experiments copy, with no runs of its own). `check_structure.py` "experiment status" enforces it.
There is no `retired` status: retiring an experiment means deleting its directory, and git history is
the archive. Decide by citation, not commit age -- a quiet experiment a guide still cites is frozen,
not abandoned.

## What lives here
- Task specs, `input_bundles/`, method specs, per-target guides, and the harness drivers that run them.
- Kernel/capsule corpora that are experiment-specific inputs.

## What does NOT belong here
- **Reusable compiler primitives** → `src/merlin/`.
- **Phase orchestration and evaluated conformance** →
  `packages/merlin-experiments/src/merlin_experiments/`.
- **Optional compiler workflows and public staged clients** → their existing
  namespace owner under `packages/*/src/`; do not create a second implementation.
- **Generated output** → `out/runs/<target>/<suite>/` (runs) or `out/artifacts/` (products). Never
  in-tree — the `check_artifact_layout` gate forbids `experiments/*/reports/` and
  `experiments/*/runs/`. The top-level `runs/`/`artifacts/`/`build/` roots are retired.

## The rule (consumption direction)
Core compiler primitives must not import native engines or own experiment admission.
Shared public resources belong to their declared package owner; private corpora and
target-specific inputs remain explicit run inputs, not implicitly bundled dependencies.
Use the existing corpus workflow, frozen-source resolver and tool registry instead
of adding checkout-relative lookups or parallel implementations. Source and package
dependency gates enforce this direction; see `docs/reference/architecture.md`.

## Invariants
- Resolve repository/output paths through `merlin.common.paths`, package-owned
  helper source through `module_source_path()`, and target facts through descriptors.
  Never infer these from hardcoded `parents[N]`, a retired layout or a machine path.
- `targetgen_evals/` is import-isolated by design (zero `merlin.*` imports) — keep it that way.
- `agent_bench/` is the clean target-agnostic model the other benches should converge toward.

## In progress
Phase 0 derivation/admission, phase 1 session/provider/feedback operations and
phase 2 campaign/measurement/report policy have canonical experiments-package
owners. The large authoring and portfolio controllers still remain native.
Their installed execution, full checkpoint/resume and inherited frozen-child
transport must be qualified before declaring the migration complete.
`docs/reference/repo_structure.md` records that ownership boundary; consult it
before changing controller state, broker launches or formal grading.
