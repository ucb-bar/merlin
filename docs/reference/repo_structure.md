---
title: Repository and implementation map
kind: reference
status: current
owner: core
last_verified: 2026-11-20
related: [architecture, ../guides/01-inputs, ../guides/12-experiments]
code_refs: [src/merlin, experiments/catalog.yaml, examples, tools, tests, build_tools]
---

# Repository and implementation map

There is one active Python distribution, with source under `src/merlin/`.
Opening the repository directory does not supply a second Python package.
Start with [the guide](../README.md); use this map when you need to change code.

The custom TTIR branch integrates `refactor/repository-layout` at
`2451b8134a7995d4e73efa3a465f975a1d532727`. Its release checkpoint `e5d522fc5`
and the custom branch's `e47c4fd16` have the identical tree
`989d5c0363c924a5d9e9a473ff9b52fe3864857f`. That shared content reconciles the
TTIR extensions with the new owners. Both actual histories remain merge parents;
published history is not rewritten.

Original TTIR regression sources remain pinned at
`53e7b01158c918dbd6d4dad0ac38ba0a901f16ba` for explicit workflow replay, instead
of restoring them wholesale into the compact permanent suite. The input,
lowering and runtime chapters describe retained interfaces and changed paths.
Run the opt-in regression replay with the selected compiler environment:

```sh
python build_tools/checks/replay_ttir_frontend.py --repo . --python /path/to/merlin-env/bin/python
```

The checked selection pins 98 original source members and a reviewed adapter.
It relocates imports/source pins and checks private seed copies in two cases whose
old zero-rewrite expectations already fail on the original branch. All numerical,
effect and custody checks remain active. Each replay records source hashes, the
adapter hash, environment interpreter and pytest output under `out/build/ttir-replay/`.
Native compiler, cross compiler and simulator paths are explicit environment selections;
unavailable prerequisites remain reported by the original tests.

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
| Inputs | [`compiler/compilation.py`](../../src/merlin/compiler/compilation.py), [`compiler/frontend/`](../../src/merlin/compiler/frontend) | Input records, exports and import |
| Target meaning | [`target/spec/`](../../src/merlin/target/spec), [`target/system.py`](../../src/merlin/target/system.py) | Numerical definitions, resources and memory |
| Preparation | [`compiler/quantization/`](../../src/merlin/compiler/quantization), [`capture/`](../../src/merlin/capture) | Precision contracts, preparation and shared capture |
| Discovery and selection | [`compiler/system.py`](../../src/merlin/compiler/system.py), [`compiler/planning/`](../../src/merlin/compiler/planning) | Backend offers, fusion alternatives and placement |
| Partitioning | [`compiler/ir/boundary.py`](../../src/merlin/compiler/ir/boundary.py), [`compiler/partition/`](../../src/merlin/compiler/partition) | Live values, effects and materialized regions |
| Canonical OOT input | [`compiler/lower/interface.py`](../../src/merlin/compiler/lower/interface.py), [`compiler/dialects/interface.py`](../../src/merlin/compiler/dialects/interface.py) | Complete source coverage, shared Interface IR conversion and grammar |
| Transformations | [`compiler/transforms/`](../../src/merlin/compiler/transforms) | Shared rewrite algorithms |
| Scheduling | [`compiler/schedule/`](../../src/merlin/compiler/schedule), [`compiler/kernel_ir/`](../../src/merlin/compiler/kernel_ir) | Command schedules and counted kernel records |
| Pass invocation | [`compiler/lower/native/invocation.py`](../../src/merlin/compiler/lower/native/invocation.py), [`compiler/frontend/preprocess.py`](../../src/merlin/compiler/frontend/preprocess.py) | Native MLIR invocation and shared xDSL preparation |
| Scalar lowering | [`compiler/lower/native/pipeline.py`](../../src/merlin/compiler/lower/native/pipeline.py), [`compiler/lower/native/passes.py`](../../src/merlin/compiler/lower/native/passes.py) | Scalar pipeline and reusable LLVM finalization |
| OOT lowering | [`compiler/backend/providers.py`](../../src/merlin/compiler/backend/providers.py), [`compiler/backend/source_proposals.py`](../../src/merlin/compiler/backend/source_proposals.py), [`compiler/backend/c_abi.py`](../../src/merlin/compiler/backend/c_abi.py) | Provider selection, implementation offers and callable region boundaries |
| Build | [`compiler/build/`](../../src/merlin/compiler/build) | Objects, C control, image linking and Zephyr configuration |
| Execution | [`runtime/execution/service.py`](../../src/merlin/runtime/execution/service.py) | Existing-artifact execution and runner selection |
| Verification | [`verification/`](../../src/merlin/verification), [`evaluation/`](../../src/merlin/evaluation) | Semantic checks, independent references and assessment |

## Preparation and target ownership

`capture/adapters/` connects supported inputs to their framework converter.
`capture/preparation/` owns the selected worker, export/state observations and
capture receipts. Model conversion uses model2MLIR's existing `m2m.convert` API.
Merlin's recipe policy and numerical admission stay in `compiler/quantization/`;
the TorchAO adapters and arithmetic rewrites live below `quantization/torchao/`.

Shared instruction observations live in `target/rtl/instructions/`. They derive
and check selected source declarations. Instruction emission stays in the OOT
provider: Kodiak owns the shared matrix encoder, and Iris declares that dependency.
Independent arithmetic references live in `verification/references/`, outside
the compiler implementations being evaluated.

The shared pass library is in `compiler/transforms/`, `compiler/schedule/` and
`compiler/lower/options/`. Prompt inputs, procedural skills and optimization
catalogs live in `resources/prompts/`, `resources/skills/` and
`resources/optimizations/`. Backends supply the target facts, effects and encoders
that the shared algorithms consume.

Backends share the selected Merlin installation. Backend-specific changes to a
shared pass live in the OOT package under `src/<backend>/transforms/overrides/`,
selected explicitly by its pipeline. The base API stays in Merlin; an OOT package
does not include a private edited SDK. Merlin's `compiler/lower/interface.py`
owns upstream conversion and canonical Interface IR parsing/admission. OOT
compilers receive source kind `merlin.interface.v1`; they own the next conversion
to target operations, scheduling and physical ABI. The shared conversion covers
one complete static signed i8-by-i8 matmul or batched matmul from a zero accumulator.
It admits an i32 result or the explicitly matched ordered readout, including
supported constant scale/bias parameters and saturating i8 output. Uncovered
operations, dynamic qparams and extra live results receive an explicit refusal.

Gemmini's `conversion/from_interface.py` selects its primitive implementation
from that canonical IR. `conversion/typed.py` and `conversion/to_llvm.py` own
typed target compilation; `lower/primitive/` retains the kernel strategies.
Static source bias/channel scales remain SSA operands; `ir/readout.py` owns
their ordered numerical schema and `lower/quant/readout_match.py` has the single
structural matcher. Gemmini's `conversion/readout.py` proves target store scales
and any integer preload before strategy selection.
`conversion/from_source.py` delegates upstream conversion to Merlin for existing
callers. Historical direct adapters remain explicit paths. These boundary edits
are source-only and do not establish whole-model or hardware qualification.

`compiler/backend/interface.py` owns the shared compiler-description records.
Discovery consumes those declarations without importing target implementations
into Merlin. Host authoring requirements live in
`development/targetgen/authoring/compiler_contract.py`; `prompt.py` owns prompt
rendering and preserves historical public imports. The interface source is
installed with Merlin. SDK30 declares its reviewed public dependency set:
`compiler/backend/interface.py` exposes capability, artifact, proposal and C-call
records; `compiler/ir/interface_contract.py` owns parsing and structural admission.
The host driver, source preparation, planner and evaluator are outside that public
API dependency set. Export and installation execution remain unqualified.

The callback-based whole-program route uses `compiler/partition/callback_regions.py`
for boundaries and records, `compiler/build/host_regions.py` for outlining and
host compilation, and `compiler/build/llvm_bundle.py` for compatible LLVM
linking. Target matching, ABI adaptation and completion hooks remain in OOT.

`common/access.py` holds the reviewed source-access policy. Its generated
historical name correspondences live in `common/source_aliases.py`; that table
does not grant new access. Private families use already-withheld owner prefixes
instead of maintaining a separate alias for every child module. Retired spellings
remain covered by those prefixes or explicit denies. Source-move ancestry for repository checks lives in
`build_tools/checks/source_moves.json`, including every predecessor of a merge.

## Research and development code

| Family | Responsibility |
| --- | --- |
| `development/testgen/` and `experiments/corpus/` | Test/corpus construction from selected specifications and source evidence |
| `development/backend/` | Compiler authoring, feedback, certification and author skills |
| `development/backend/optimization/` | Optimization experiments and their selected execution/evaluation inputs |
| `analysis/kernels/` | Decoding, Common Compute Abstraction, comparisons and edit proposals |
| `analysis/mining/` | Schedule-package preparation, fork lineage and reference reconstruction |
| `analysis/optimization/` | Design-space exploration and evidence analysis |
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

Within `experiments/corpus/`, `preparation.py` assembles source pools and
`scaffold.py` stages the selected SDK and task resources. `admission.py` owns
cohort and resource admission. `release.py` owns prepare, inspect, review and seal;
`release_binding.py` holds private records and exact-offload bindings.

## Implementation boundaries

Keep one coherent responsibility per module. Files may contain up to 1,000
lines when their organization remains clear; size alone is not a reason to
split a concept across files. Consolidate redundant mechanisms before adding
another module, and separate code when its responsibilities or dependencies
need independent ownership.

External-framework comparisons live in `evaluation/frameworks/`; base scalar
compilation lives in `compiler/lower/native/`. Schedule-package preparation lives in `analysis/mining/`;
RVV device implementation belongs in the RVV OOT repository. The retired
`baselines`, `rvvgen` and split research package directories are gone.

Historical module names remain in source-move and access records so that ownership
and withheld identities survive the reorganization. They are not additional source
trees. Use the owners in the compilation table for current code, and the recorded
original source snapshot when reproducing a historical run.

## Documentation and generated data

Handwritten chapters belong in `docs/guides/`. CLI, module and schema
references are generated into ignored `out/artifacts/documentation/`. Optimization catalogs are runtime resources,
not documentation. Point-in-time audit notes belong in `tmp/`; generated reports
belong in `out/`.

The documentation rewrite removed the former collection of overlapping guides
and design notes. Earlier explanations remain available through Git history;
use their recorded source revision when reproducing an older result. They do
not describe the current API by default.

Board measurement commands live in `tools/benchmarks/`. Comparison baselines live in
the separate merlin-baselines repository.
