---
title: Repository and implementation map
kind: reference
status: current
owner: core
last_verified: 2026-11-09
related: [architecture, ../guides/01-inputs, ../guides/12-experiments]
code_refs: [src/merlin, experiments/catalog.yaml, examples, packages, tests, build_tools]
---

# Repository and implementation map

There is one active Python distribution, with source under `src/merlin/`.
Opening the repository directory does not supply a second Python package.
Start with [the guide](../README.md); use this map when you need to change code.

## Top-level directories

| Directory | Contents |
| --- | --- |
| `src/merlin/` | Installed compiler, runtime, verification and optional research workflows |
| `examples/` | Runnable walkthroughs, target inputs and selected target workflows |
| `experiments/` | Study catalog, definitions, retained corpora and research inputs |
| `tests/` | Compact permanent suite, capped at 150 collected cases |
| `build_tools/` | Packaging, source checks, documentation generation and toolchain descriptions |
| `docs/` | Twelve ordered chapters and a small reference section |
| `third_party/` | Explicit external dependencies, used only when the selected workflow needs them |
| `packages/` | Retired distribution metadata plus the separate timing observer and retained study support |
| `out/` | Generated runs, artifacts and build products |

The remaining historical directory names are not additional recommended install
commands. `packages/merlin-analysis`, `merlin-dse`, `merlin-experiments` and
`merlin-mining` contain no active Python distribution implementations. Their
workflow code is in the single `merlin` namespace. The timing observer remains a
separate distribution because its measurement authority is independent of the
compiler being evaluated.

## Follow a compilation

| Stage | Open first | What to find there |
| --- | --- | --- |
| Public request | [`api.py`](../../src/merlin/api.py) | Shared compile/inspect/run service |
| Inputs | [`compiler/request.py`](../../src/merlin/compiler/request.py), [`compiler/frontend/`](../../src/merlin/compiler/frontend/) | Input records, exports and import |
| Target meaning | [`target/spec/`](../../src/merlin/target/spec/), [`system/topology.py`](../../src/merlin/system/topology.py) | Numerical definitions, resources and memory |
| Preparation | [`compiler/quantization/`](../../src/merlin/compiler/quantization/), [`capture/`](../../src/merlin/capture/) | Precision contracts, preparation and shared capture |
| Discovery and selection | [`compiler/system.py`](../../src/merlin/compiler/system.py), [`compiler/planning/`](../../src/merlin/compiler/planning/) | Backend offers, fusion alternatives and placement |
| Partitioning | [`compiler/ir/boundary.py`](../../src/merlin/compiler/ir/boundary.py), [`compiler/partition/`](../../src/merlin/compiler/partition/) | Live values, effects and materialized regions |
| Transformations | [`transforms/`](../../src/merlin/transforms/), [`compiler/transforms/`](../../src/merlin/compiler/transforms/) | Shared rewrite algorithms |
| Scheduling | [`compiler/schedule/`](../../src/merlin/compiler/schedule/), [`sched/`](../../src/merlin/sched/) | Command schedules and counted kernel records |
| Pass invocation | [`compiler/passes/`](../../src/merlin/compiler/passes/) | Native MLIR and xDSL invocation |
| Scalar lowering | [`compiler/lower/scalar/`](../../src/merlin/compiler/lower/scalar/), [`compiler/lower/llvm/`](../../src/merlin/compiler/lower/llvm/) | Base scalar pipeline and reusable LLVM finalization |
| OOT lowering | [`compiler/backend/selected.py`](../../src/merlin/compiler/backend/selected.py), [`compiler/lower/regions.py`](../../src/merlin/compiler/lower/regions.py) | Selected recipe and backend emission boundary |
| Build | [`compiler/build/`](../../src/merlin/compiler/build/) | Objects, C control, image linking and Zephyr configuration |
| Execution | [`runtime/service.py`](../../src/merlin/runtime/service.py) | Existing-artifact execution and runner selection |
| Verification | [`verification/`](../../src/merlin/verification/), [`evaluation/`](../../src/merlin/evaluation/) | Semantic checks, independent references and assessment |

## Research and development code

| Family | Responsibility |
| --- | --- |
| `experiments/phase0/` and `experiments/corpus/` | Test/corpus construction from selected specifications and source evidence |
| `experiments/phase1/` | Compiler authoring, feedback, certification and author skills |
| `experiments/phase2/` | Optimization experiments and their selected execution/evaluation inputs |
| `kernels/` | Decoding, Common Compute Abstraction, comparisons and edit proposals |
| `mining/` | Reusable kernel search and optimization orchestration |
| `analysis/dse/` | Design-space exploration and evidence analysis |
| `resources/prompts/` | Versioned prompt inputs |
| `resources/optimizations/` | Authored optimization catalogs consumed by authoring tools |
| `resources/contracts/` and `resources/schemas/` | Installed public contracts and data schemas |

The root `experiments/` directory contains study data; `src/merlin/experiments/`
contains its implementation. They are not two copies of the same package.
Generated experiment output belongs under `out/runs/` or `out/artifacts/`.

## Older names that still need care

`baselines/` contains optional external-framework comparison runners and assets.
It does not supply Merlin's base scalar compiler. `rvvgen/` is a compatibility
namespace for `mining/`; it is not the RVV OOT implementation. `llvmlower/` and
`targetgen/` still contain active code as well as imports retained for existing
callers. A folder's age or name is not proof that its implementation is unused.

Use the canonical owners in the compilation table for new shared work. Target
instruction selection, encoders and device implementations belong in the OOT
repository. Independent execution references must not move into the candidate
compiler's editable inputs.

## Documentation and generated data

Handwritten chapters belong in `docs/guides/`. The CLI, module and schema
references are generated from code. Optimization catalogs are runtime resources,
not documentation. Point-in-time audit notes belong in `tmp/`; generated reports
belong in `out/`.

The documentation rewrite removed the former collection of overlapping guides
and design notes. Earlier explanations remain available through Git history;
use their recorded source revision when reproducing an older result. They do
not describe the current API by default.
