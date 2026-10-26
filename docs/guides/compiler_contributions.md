---
title: Contributing compiler transformations and coverage
kind: guide
status: current
owner: compiler
last_verified: 2026-10-26
related: [../reference/shared_compiler_passes, conditional_compiler_pipelines, verified_compiler_tooling, phase0_specification, ../design/compiler_verification, artifact_evaluation]
code_refs:
  - src/merlin/transforms/evaluated_exploration.py
  - src/merlin/sched/primitives/loop_handles.py
  - src/merlin/llvmlower/source_program.py
  - src/merlin/transforms/pipeline_state.py
  - src/merlin/transforms/pipeline_policy.py
  - src/merlin/compile/scheduling/dense_source.py
  - src/merlin/compile/fusion_workflow.py
  - src/merlin/compile/connected_regions.py
  - src/merlin/compile/connected_host.py
  - src/merlin/verify/lean_checker.py
  - src/merlin/targetgen/software_spec.py
  - src/merlin/targetgen/quantization_spec.py
  - packages/merlin-experiments/src/merlin_experiments/phase0/source_requirement_ledger.py
---

# Start contributing to the compiler

The joint experiment develops one evolving target compiler using shared Merlin
passes, graph/scheduling representations and host code generation. It reuses the
native Phase 1 loop for correctness and performance work together. An available
API is not evidence that an authored compiler consumes it or improves performance.

Local development can start without a full campaign or FPGA. Follow repository
instructions and [development setup](getting_started.md), then choose a boundary:

| Work | Start in code | Reference and focused checks |
| --- | --- | --- |
| Pass selection and efficient generated code | [pipeline_policy.py](../../src/merlin/transforms/pipeline_policy.py): `evaluate_variants`; [dense_source.py](../../src/merlin/compile/scheduling/dense_source.py): `schedule_source_dense`; [fusion_workflow.py](../../src/merlin/compile/fusion_workflow.py): `realize_candidate` | [Shared passes](../reference/shared_compiler_passes.md), [conditional pipelines](conditional_compiler_pipelines.md); `test_pipeline_policy.py`, `test_dense_source_catalog.py`, `test_fusion_exploration.py` |
| Graph and state preservation | [connected_regions.py](../../src/merlin/compile/connected_regions.py): `plan_connected_region`; [lean_checker.py](../../src/merlin/verify/lean_checker.py): `verify_pipeline_change`; [Lean sources](../../src/merlin/verify/lean/Merlin/) | [Verification boundaries](../design/compiler_verification.md), [toolchain integration](verified_compiler_tooling.md); `test_connected_regions.py`, `test_lean_state_effects.py` |
| Phase 0 coverage and quantization | [software_spec.py](../../src/merlin/targetgen/software_spec.py): `admit_operation`; [quantization_spec.py](../../src/merlin/targetgen/quantization_spec.py): `build_quantization_contract`; [Phase 0](../../packages/merlin-experiments/src/merlin_experiments/phase0/) | [Specification](phase0_specification.md), [quantization methodology](../design/target_quantization_methodology.md); `test_quantization_spec.py`, `test_phase0_support_accounting.py` |

Test files are under `merlin/tests/ir/`, `merlin/tests/infra/` or
`merlin/tests/targetgen/`; run the affected family, not a full campaign for every
edit. Lean execution additionally requires the explicitly selected toolchain.

Use source semantics, shapes and explicit hardware facts to derive applicability.
Keep generic transforms, packing and host code in Merlin; target encodings,
physical layouts/effects and ABI belong in the target backend. Preserve exact
numerical policy and every visible graph output. A transformation may need a
new abstraction, but its returned product must reach an actual consumer.

Efficient generated programs can motivate structural hypotheses without access
to their compiler internals. Sweep applicable pass configurations/orderings and
use independent examples; do not transplant reference kernels into shared passes.
Keep excluded reference implementations and scores outside automatic author inputs
and describe reference-guided study treatments explicitly.

Return one coherent change with a before/after example, applicability conditions,
focused positive/negative checks and limits. Bind observations to exact compiler,
input and runtime identities. Provisional model estimates are not physical gains;
graph proofs do not discharge unmodeled arithmetic or target execution.

For an actual campaign, use [artifact setup](artifact_evaluation.md) and
[terminal monitoring](experiment_dashboard.md). Fresh-machine inputs and physical
validation remain separate requirements; pass tests do not establish full-model
correctness or reproducibility of a historical result.

## Connect construction to evaluation

`merlin.llvmlower.source_program.prepare_source_program` preserves the original
sibling integer source, its host epilogues, complete outputs and logical pointer
ABI. `emit_source_program` consumes actual device and stock-host emitter results.
The initial adapter supports rank-two sibling integer contractions; other graph
families retain their existing route or refuse. ABI and link checks do not prove
arithmetic, unavoidable device work or physical effects.

Use `merlin.sched.primitives.loop_handles.locate_loop` for an explicit structural
occurrence when sibling loop names repeat. Handles bind a kernel digest and
expire after rewriting; `transform_loop` returns replacement occurrences for
supported split, unroll and ordered-partition composition. Ambiguous legacy name
selection still refuses.

`merlin.transforms.evaluated_exploration.explore_evaluated` joins actual registered
pass factories to external evaluations. It deduplicates emitted bytes and returns
the evaluated winning product under matching source, inputs, ABI, reference,
facts, library, provider and whole-call scope identities. UNKNOWN, incomplete
windows and numerical failures cannot win. Counts, traffic and utilization are
diagnostics; complete elapsed cycles determine the performance comparison.

The compact scheduling SDK exposes these APIs only when its selected inventory
contains their modules. The editable full compiler has a separate reviewed
inventory. Inspect availability before choosing a route; do not infer a grant
from an advertised name or copy a target implementation into the shared library.
