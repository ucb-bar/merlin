---
title: Repository and implementation map
kind: reference
status: current
owner: core
last_verified: 2026-11-09
related: [architecture, ../guides/01-inputs, ../guides/12-experiments]
code_refs: [src/merlin, experiments/catalog.yaml, examples, tools, tests, build_tools]
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
| `experiments/` | Nine authored catalog, definition, template and instruction files |
| `tests/` | Compact permanent suite, capped at 150 collected cases |
| `build_tools/` | Packaging, source checks, documentation generation and toolchain descriptions |
| `docs/` | Twelve ordered chapters and a small reference section |
| `third_party/` | Explicit external dependencies, used only when the selected workflow needs them |
| `tools/` | Standalone console and independent timing observer |
| `out/` | Ignored generated runs, corpora, input bundles, captures, caches and build products |

The compiler and research workflows ship in one Merlin distribution. The console
is a standalone Go tool; the timing observer has its own build and measurement
authority under `tools/timing/`. They are not separate copies of Merlin.

## Follow a compilation

| Stage | Open first | What to find there |
| --- | --- | --- |
| Public request | [`api.py`](../../src/merlin/api.py) | Shared compile/inspect/run service |
| Inputs | [`compiler/request.py`](../../src/merlin/compiler/request.py), [`compiler/frontend/`](../../src/merlin/compiler/frontend) | Input records, exports and import |
| Target meaning | [`target/spec/`](../../src/merlin/target/spec), [`system/topology.py`](../../src/merlin/system/topology.py) | Numerical definitions, resources and memory |
| Preparation | [`compiler/quantization/`](../../src/merlin/compiler/quantization), [`capture/`](../../src/merlin/capture) | Precision contracts, preparation and shared capture |
| Discovery and selection | [`compiler/system.py`](../../src/merlin/compiler/system.py), [`compiler/planning/`](../../src/merlin/compiler/planning) | Backend offers, fusion alternatives and placement |
| Partitioning | [`compiler/ir/boundary.py`](../../src/merlin/compiler/ir/boundary.py), [`compiler/partition/`](../../src/merlin/compiler/partition) | Live values, effects and materialized regions |
| Transformations | [`transforms/`](../../src/merlin/transforms), [`compiler/transforms/`](../../src/merlin/compiler/transforms) | Shared rewrite algorithms |
| Scheduling | [`compiler/schedule/`](../../src/merlin/compiler/schedule), [`sched/`](../../src/merlin/sched) | Command schedules and counted kernel records |
| Pass invocation | [`compiler/passes/`](../../src/merlin/compiler/passes) | Native MLIR and xDSL invocation |
| Scalar lowering | [`compiler/lower/scalar/`](../../src/merlin/compiler/lower/scalar), [`compiler/lower/llvm/`](../../src/merlin/compiler/lower/llvm) | Base scalar pipeline and reusable LLVM finalization |
| OOT lowering | [`compiler/backend/selected.py`](../../src/merlin/compiler/backend/selected.py), [`compiler/lower/regions.py`](../../src/merlin/compiler/lower/regions.py) | Selected recipe and backend emission boundary |
| Build | [`compiler/build/`](../../src/merlin/compiler/build) | Objects, C control, image linking and Zephyr configuration |
| Execution | [`runtime/service.py`](../../src/merlin/runtime/service.py) | Existing-artifact execution and runner selection |
| Verification | [`verification/`](../../src/merlin/verification), [`evaluation/`](../../src/merlin/evaluation) | Semantic checks, independent references and assessment |

## Research and development code

| Family | Responsibility |
| --- | --- |
| `development/testgen/` and `experiments/corpus/` | Test/corpus construction from selected specifications and source evidence |
| `development/backend/` | Compiler authoring, feedback, certification and author skills |
| `development/backend/optimization/` | Optimization experiments and their selected execution/evaluation inputs |
| `kernels/` | Decoding, Common Compute Abstraction, comparisons and edit proposals |
| `mining/` | Reusable kernel search and optimization orchestration |
| `analysis/dse/` | Design-space exploration and evidence analysis |
| `resources/prompts/` | Versioned prompt inputs |
| `resources/optimizations/` | Authored optimization catalogs consumed by authoring tools |
| `resources/contracts/` and `resources/schemas/` | Installed public contracts and data schemas |

The root `experiments/` tree contains only these nine authored files:

```text
experiments/
  catalog.yaml
  README.md
  AGENT.md
  definitions/
    baseline-functional-template.yaml
    AGENT.md
  templates/
    AGENT.md
    test-generation/
      performance.yaml
      README.md
      AGENT.md
```

Seven target definitions live beside their inputs in `examples/compiler-generation/`;
the catalog selects those and the generic definition. `src/merlin/experiments/`
owns selection, run records and tracking, while `development/testgen/` and
`development/backend/` own the engines. Generated corpora, captures and bundles
are selected explicitly from reviewed artifacts under ignored `out/`. They are
not default source-tree fixtures. Retained historical payloads require their
recorded inputs and original source snapshot.

## Implementation boundaries

External-framework comparisons live in `evaluation/frameworks/`; base scalar
compilation lives in `compiler/lower/scalar/`. Kernel search lives in `mining/`;
RVV device implementation belongs in the RVV OOT repository. The retired
`baselines`, `rvvgen` and split research package directories are gone.

The older `compile/`, `llvmlower/`, `semantic_compiler/` and `verify/` families
still include compatibility imports and some specialized implementations. New
shared code uses the owners in the compilation table. Do not duplicate an
algorithm just to preserve an old import path.

## Documentation and generated data

Handwritten chapters belong in `docs/guides/`. CLI, module and schema
references are generated into ignored `out/artifacts/documentation/`. Optimization catalogs are runtime resources,
not documentation. Point-in-time audit notes belong in `tmp/`; generated reports
belong in `out/`.

The documentation rewrite removed the former collection of overlapping guides
and design notes. Earlier explanations remain available through Git history;
use their recorded source revision when reproducing an older result. They do
not describe the current API by default.

Board measurement commands live in `tools/benchmarks/`; the reusable comparison
implementations remain in `src/merlin/compare/` and `evaluation/frameworks/`.
