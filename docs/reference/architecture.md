---
title: Architecture
kind: reference
status: current
owner: core
last_verified: 2026-09-20
related: [repo_structure, core_dialects, lowering_pipeline]
code_refs: [src/merlin, packages, experiments/catalog.yaml]
---

# Architecture

Merlin generates target compilers through three phases: derive capsules (0), create and certify
a functional compiler (1), and optimize its performance (2). Start at
[`experiments/catalog.yaml`](../../experiments/catalog.yaml) through `merlin experiment`.
Shared artifacts (schemas) connect those phases and the supporting research workstreams.
Two compiler planes remain: xDSL for prototyping and MLIR/C++ for stable code.

## Implementation ownership

Shared compiler IR, scheduling, contracts, runtime and target/toolchain resolution live in
`src/merlin`. Optional distributions in `packages/` own experiments and trusted evaluation,
DSE, mining, and analysis. Their historical `merlin.*` import names remain compatible;
an import prefix alone does not identify its distribution. `merlin/python/merlin` is a
legacy symlink to core, not a second implementation.

The catalog defines experiments; `merlin/experiments/` still contains checkout-dependent
native engines and resources. Reviewed phase-0 handoff and frozen phase-1/2 inputs preserve
the existing grading authority. Installed commands and synthetic tests do not establish
hardware readiness or qualify a compiler. See [repository structure](repo_structure.md)
for navigation and the remaining qualification boundaries.

## Data flow

```
External kernels/repos --> integration adapters --> kernel_record / abstraction_candidate / policy_rule
ISA/docs/RTL           --> targetgen            --> target_contract --> dialect_plan --> dialect scaffold
workload_region        --> design_pressure      --> design_pressure --> candidate_contracts
candidate + cost model  --> dse                  --> dse_result --> exploitability_report
```

## Compiler flow (intended)

```
linalg / tensor / scf
  -> contract     (facts, obligations, legality)
  -> schedule     (chosen compiler decisions)
  -> interface    (target-independent HW/SW abstractions)
  -> <target dialect>    (e.g. toynpu)
  -> runtime      (command buffers, dispatch, waits, profiling)
  -> binary / simulator / external runner
```

## Principles

- Core source lives under `src/merlin`; optional research source lives under `packages/`.
- Experiment definitions have one catalog; generated state belongs under `out/{runs,artifacts,build}`.
- Target-specific support belongs OOT where qualified; retained reference resources are not
  proof that an exported package is a standalone compiler.
- Coordinate through schemas, not prose.
- Integrations are adapters, never vendored repos.
- Prototype in xDSL; promote to MLIR/C++ only when stable.
- A concept becomes a dialect only when it must survive passes, be verified/transformed, and
  lower. Otherwise it is a schema/YAML artifact first.
