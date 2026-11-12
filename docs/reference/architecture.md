---
title: Compiler architecture
kind: reference
status: current
owner: compiler
last_verified: 2026-11-09
related: [repo_structure, ../guides/04-planning, ../guides/06-lowering]
code_refs: [src/merlin/api.py, src/merlin/compiler, src/merlin/system, src/merlin/runtime, src/merlin/verification, src/merlin/xdsl_dialects]
---

# Compiler architecture

Start with the [ordered guide](../README.md) for commands and examples. This
page describes the boundaries between the components that implement that flow.

```text
source program + precision intent + selected target system
    -> capture / preparation / import
    -> implementation discovery and region/resource planning
    -> selected partition and dispatch boundaries
    -> shared control + scalar or OOT region lowering
    -> artifact publication / optional image linking
    -> execution and independent verification
```

## Public entry points

`merlin.api.compile`, `inspect`, `inspect_backends`, `inspect_spec`, `run` and
`collect` are shared with the public CLI. An installed backend implements
`validate` and `compile`. An explicitly constructed `SystemCompiler` adds
source-level proposal discovery; `PlannedCompiler` consumes a selected plan.
The output can be IR, an object, a device binary, commands or a complete image.

The saved-model route also retains `merlin.model` and existing capture/build
entry points. Do not infer identical options or coverage from their common use
of shared lowering code.

## Ownership and dependencies

| Component | Owns | Must not decide |
| --- | --- | --- |
| Target/specification | Selected hardware, numerical meaning, topology and evidence | Which unverified backend claim becomes hardware truth |
| Frontend/preparation | Source capture, numerical realization and import | Device placement over source nodes that import may replace |
| Planning | Compatible region groupings, implementations, resources and boundaries | A new numerical meaning to rescue an unsupported implementation |
| Shared transforms | Reusable graph, loop, memory and scheduling algorithms | Target encodings or workload-name shortcuts |
| OOT backend | Implemented target capabilities, proposals, target lowering and encoders | Application acceptance thresholds or independent verification results |
| Build/runtime | Objects, images, deployment, invocation and observed completion | Whether a process exit alone proves numerical correctness |
| Verification/evaluation | Semantic checks, references and execution assessment | Candidate compiler optimization policy |
| Experiment orchestration | Authoring rounds, selected inputs and recorded studies | A second implementation of the shared compiler |

Shared compiler mechanisms remain below experiment orchestration. LLVM
finalization remains below scalar, vector and control compilation; it must not
import the driver or backend loader. Framework dependencies are imported only
by the workflows that need them. A package being installed does not grant a
candidate access to graders, hidden inputs or independent references.

## Compiler development

`development/testgen/` derives compiler tests from selected specifications and source
evidence. `development/backend/` owns the native authoring engine, which develops
correctness and performance together. `experiments/` inside the Python package selects
and tracks those workflows; the nine authored files at the repository root only provide
the catalog, generic definition and template. External-framework comparisons belong in
`evaluation/frameworks/`; the console and independent timing observer live in `tools/`.

Generated corpora, bundles, captures, compiler snapshots and caches live under ignored
`out/`. A run binds its explicitly selected reviewed inputs. Historical artifacts retain
their original bytes and source identities; they are not discovered as fresh defaults.

## Representations

The xDSL prototypes live in `src/merlin/xdsl_dialects/`:

| Namespace | Role |
| --- | --- |
| `contract` | Facts, capabilities and obligations |
| `schedule` | Selected decisions and bindings |
| `interface` | Target-independent hardware/software abstractions |
| `runtime` | Generic launch, events and measurements |
| `dse` | Interface candidates and measured variant results |

These roles do not imply that every compile route materializes each dialect in
that order. Some routes use typed Python records and call a selected OOT service.
Target dialects are supplied by their backends.

`merlin.compiler.kernel_ir.ir` uses Python records and counted `mk` text. Shared block
scheduling uses explicit load/compute/store records in `compiler/schedule/`.
These are different representations from xDSL `schedule` operations and the
upstream MLIR Transform dialect; adapters must be explicit.

## Composition

A SoC integration selects component providers and contributes integration policy,
transfers and completion requirements. It reuses the component compilers rather
than copying them. Scalar and RVV implementations can share one CPU resource.
Availability, addressability and concurrency follow the selected topology.

A selected region's artifact must agree with its callable boundary. A raw binary
needs an admitted loader/wrapper; LLVM is only one possible output path.
Numerical fidelity, complete target lowering and observed device execution are
separate verification obligations, described in [chapter 9](../guides/09-verification.md).
