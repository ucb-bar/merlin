---
title: Architecture
kind: reference
status: current
owner: core
last_verified: 2026-10-05
related: [repo_structure, core_dialects, lowering_pipeline]
code_refs: [src/merlin, packages, experiments/catalog.yaml]
---

# Architecture

Merlin derives capsules and develops target compilers through shared authoring,
evaluation and optimization infrastructure. The joint experiment combines
correctness and performance development in one evolving compiler using the Phase 1
engine; separate Phase 1/Phase 2 workflows and module names remain available. Start at
[`experiments/catalog.yaml`](../../experiments/catalog.yaml) through `merlin experiment`.
Shared artifacts (schemas) connect those phases and the supporting research workstreams.
Two compiler planes remain: xDSL for prototyping and MLIR/C++ for stable code.

## Implementation ownership

Shared compiler IR, scheduling, contracts, runtime and target/toolchain resolution live in
`src/merlin`. Optional distributions in `packages/` own experiments and trusted evaluation,
DSE, mining, and analysis. Their historical `merlin.*` import names remain compatible;
an import prefix alone does not identify its distribution. `merlin/python/merlin` is a
legacy symlink to core, not a second implementation.

### Compiler and target boundary

| Owner | Implementation |
| --- | --- |
| Merlin | Reusable host code generation, packing, requantization, graph/global optimizations, dispatch, buffer ownership, device compilation orchestration, and runtime infrastructure |
| Target OOT MLIR dialect repository | Target dialect operations, instruction encodings, device kernels/schedules, hardware layout/resource facts, target ABI glue, and target execution support |

An optimization that can be selected independently of the accelerator belongs in Merlin,
including one first measured on a particular device. A mixed optimization uses an explicit
contract: Merlin supplies the generic algorithm and verifies its semantic obligations; the
provider supplies target legality facts, instruction selection and implementation. Generic
OOT prototypes move into Merlin when promoted, with provider delegation replacing duplicated
code. Optional numeric policies retain explicit selection and their original accuracy gates.

Host CPU implementations are shared independently of accelerator selection. Two
targets with the same host ISA, ABI and numerical/effect capabilities reuse the
same host code generation and runtime; a different host selects an explicitly
supported CPU implementation. Device layout and ABI bridges remain target-owned.
See [shared host arithmetic](host_cpu_arithmetic.md). Shared host support does not
admit a target compiler seed, device schedule or protected performance answer.

The OOT dialect is a general compiler backend. Production decisions follow operation
semantics, shapes, layouts, numeric contracts and hardware capabilities, with legality and
resource checks. They must not depend on model names, captured provenance IDs, golden outputs
or benchmark constants. Constant and shape specialization derives from the current input IR.
Experiment drivers may select source-bound candidates; promotion requires general transforms
and cost models tested on independent cases, including tails and refused/fallback cases.
Provenance IDs support traceability and exact source-to-device binding, independently of
the semantic and cost-based decision to apply an optimization.

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
