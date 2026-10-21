---
title: Optimize staged contractions in device LLVM
kind: guide
status: current
owner: llvmlower
last_verified: 2026-10-21
related: [shared_pass_compiler_generation, default_pass_evaluation]
code_refs:
  - src/merlin/llvmlower/staged_dense_llvm.py
  - src/merlin/llvmlower/target_staged_dense.py
  - src/merlin/llvmlower/counted_llvm_cfg.py
  - src/merlin/llvmlower/compact_dense_llvm.py
  - src/merlin/compile/scheduling/staged_dense.py
  - src/merlin/compile/scheduling/staged_rhs_residency.py
---

# Optimize staged contractions in device LLVM

Use this route when a generated compiler emits LLVM containing counted loops,
but does not expose the finite instruction-module consumer. The ordinary
`DeviceRouting.llvm_transform_chain` hook already feeds selected LLVM into
object compilation. The original compiler package stays unchanged.

This route is opt-in and outside the frozen authoring SDKs. It does not discover
instruction meanings from assembly text or automatically qualify a backend.

## Supply the target boundary

The original integration supplies a `StagedDenseTargetAdapter`:

- `project(module)` reads actual operations, addresses and configuration and
  returns the complete ordered staged effects. Unknown or unaccounted work
  returns `None`. A retained plan cannot stand in for this projection.
- `emit(module, original_effects, selected_effects)` returns a new LLVM module
  implementing the selected effects, keeping reusable loops within the selected
  static-code budget. Physical instruction encoding belongs to this provider.

Projection must account for the whole module, including globals, configuration,
completion, unmodified functions and opaque effects. `complete_effects=True`
declares that obligation; it is not evidence that the adapter satisfies it.
Qualify the projector and emitter independently against actual output and
execution before relying on them for a real target.

`StagedDenseContract` separates four selected identities: hardware, software,
numerical policy and execution. Structural inputs include primitive dimensions,
operand and accumulator widths and **assigned usable** local rows. Raw physical
capacity does not establish which rows this invocation owns. Software inputs
include descriptor visibility at compute consumption, accumulator continuation,
identity readout, immutable RHS ownership and disjoint external buffers.
The staged checker does not infer these laws from RTL dimensions.

## Select shared passes

The following uses existing algorithms; provider callbacks and contract creation
are integration inputs, not parts of the shared optimizer:

```python
from dataclasses import replace

from merlin.compile.scheduling.staged_dense import (
    StagedDenseRewrite,
    eliminate_redundant_rhs_loads,
    order_output_tiles_n_before_m,
)
from merlin.compile.scheduling.staged_rhs_residency import place_rhs_stripe
from merlin.llvmlower.host_transform_chain import HostLLVMArtifact, HostLLVMTransformChain
from merlin.llvmlower.staged_dense_llvm import StagedDenseLLVMTransform
from merlin.llvmlower.target_staged_dense import StagedDenseTargetAdapter

adapter = StagedDenseTargetAdapter(project, emit, complete_effects=True)
selection = StagedDenseLLVMTransform(
    problem=problem,
    contract=contract,
    adapter=adapter,
    translator=HostLLVMArtifact.capture(selected_mlir_translate),
    semantic_proofs=original_qualification_artifacts,
    implementation_sources=original_adapter_dependencies,
    rewrites=(
        StagedDenseRewrite("rhs_placement", place_rhs_stripe),
        StagedDenseRewrite("output_order", order_output_tiles_n_before_m),
        StagedDenseRewrite("rhs_reuse", eliminate_redundant_rhs_loads),
    ),
)
stage = selection.stage()
device = replace(
    device,
    llvm_transform_chain=HostLLVMTransformChain((stage.contract,), (stage,)),
)
```

Stripe placement gives different reduction panels distinct assigned rows.
Column-major output-tile order puts uses of the same RHS near one another.
Reuse then removes a load only while the same immutable source cells are still
present. The independent staged checker checks every ordered reduction and
complete output after each rewrite. Insufficient assigned capacity or absent
ownership prevents the relevant optimization. Unknown costs are not cycle
estimates; this selection establishes no automatic profitability rule.

Phase 2 may select admitted rewrite combinations and policy choices under the
same frozen library and verification rules. It cannot change the meanings of
the effects, assigned capacities, numerical policy or checking budgets.

## Check consumed output

The bridge imports the actual LLVM file produced by the compiler. It checks
the source effects, applies the selected rewrites, emits LLVM and independently
reimports the file that object compilation will consume. Actual projected
effects must equal the checked selected effects and pass complete staged
verification. Ignored changes, wrong addresses, missing work and changed
source, callbacks, tools or evidence refuse the build.

The initial completion check permits physical local-row reassignment, reordering
whole completed output tiles and removing redundant RHS loads. It preserves
every existing completion fence and the remaining per-tile effect order.
Removing waits or overlapping incomplete tiles requires a separate qualified
asynchronous route; arithmetic agreement alone does not authorize it.

The existing transform-chain receipt retains the selected LLVM and translation
invocations. Object optimization, compilation, symbol renaming, final linking,
linked-image admission and numerical execution retain their ordinary checks.
The bridge does not replace those checks with a planning report.

Unknown projection preserves the original LLVM bytes exactly. Invalid known
effects or a mismatching emission fail explicitly. Empty or ineffective rewrite
selections retain the original bytes. No alternative compiler is tried after
a selected tool failure.

## Compile symbolic compact schedules

`CompactDenseLLVMTransform` connects the existing compact dense planner and
symbolic verifier to the same ordinary device-object hook. Use it for larger
contractions whose counted loops fit the selected static budget. It checks
coverage, complete ordered reductions, tails, physical backing and latest
writers without enumerating all scalar products.

The original provider supplies a qualified `DenseLoopAdapter`, assigned
`DenseLoopStorage`, `DenseProblem` and `DenseResources`. The problem describes
rectangular primitive dimensions and element widths; resources describe usable
budgets. Structural RTL facts inform those values. The software contract and
independent qualification establish exact arithmetic, continuation, transfer
layout and completion. Capacity alone grants none of these semantics.

```python
from merlin.llvmlower.compact_dense_llvm import CompactDenseLLVMTransform

selection = CompactDenseLLVMTransform(
    problem=problem,
    resources=assigned_resources,
    storage=assigned_storage,
    adapter=qualified_dense_adapter,
    project_module=project_actual_module,
    emit_module=emit_selected_module,
    translator=HostLLVMArtifact.capture(selected_mlir_translate),
    semantic_proofs=original_qualification_artifacts,
    complete_effects=True,
    external_buffers_disjoint=True,
    implementation_sources=original_adapter_dependencies,
)
stage = selection.stage()
# Install stage through HostLLVMTransformChain, as above.
```

`project_module` must account for the complete current module and reconstruct
its actual `Kernel`. `emit_module` receives that module and the selected Kernel.
The bridge checks the emitted module and reimports the LLVM file selected for
object compilation; an emitter that ignores blocking or changes a pointer
refuses. `source_storage` may describe the baseline assignment when it differs
from the candidate assignment. Both assignments must have original ownership.

The default planner ranks legal blocks by requested operand bytes, then command
count; these are not latency estimates. Phase 2 can compare admitted loop
orders and enable `reuse_pe_weights=True` when the original adapter declares
qualified retained-weight semantics. It may tune legal blocking controls through
the same dense planner. It cannot enlarge hardware assignments, relax numerical
policy, invent completion rules or change verification limits to admit a plan.

`group_dma=True` additionally permits bounded wide operand transfers when the
original adapter supplies `DenseLoadGrouping`. That law declares source strides
and base alignments, physical destination block pitches, transfer limits and an
independently qualified blocked-copy mapping. Ordinary rectangular-copy support
does not establish this mapping. The original projector decodes each actual
wide instruction into its primitive load effects; the verifier checks those
effects and counts one physical command. Unavailable capabilities preserve the
original LLVM. Both grouping and PE retention are disabled by default.

This route currently requires declared synchronous logical operations and exact
accumulator continuation. Raw queued DMA commands do not satisfy that premise
merely because the target has dependency queues. A qualified compound operation
may include its completion boundary. Overlap requires an asynchronous route.
Explicit zero initialization requires the adapter's separately qualified zero
law; bias seeds and arbitrary epilogues remain unsupported here. Missing
effects, disjointness or completion retains the original LLVM; recognized
invalid programs and mismatching emissions refuse. Neutral native tests establish
blocking, PE reuse and wide transfers through actual object execution, not
target qualification.

## Scope and limits

The initial staged route supports finite checked contractions with explicit
descriptor issue/consume, initialized accumulators and raw readout. It is not
an arbitrary LLVM optimizer, a general asynchronous scheduler or a substitute
for target-authored fusion and completion laws. A plain GEMM does not supply
producer-packing, epilogue or multi-output fusion opportunities.

Verification bounds are explicit and independent of code-generation policy.
The default product ceiling is 4,194,304 scalar products; a 512-cubed GEMM is
outside that finite domain. Compact emission alone does not supply a symbolic
proof for a larger reduction. Do not unroll a large kernel or silently increase
the limits to claim coverage.

For an original provider implementing a symbolic route,
`project_counted_function` reads canonical LLVM counted loops into existing
`Kernel`, `Loop` and `Call` IR without visiting dynamic iterations. It supports
positive constant steps, affine nonwrapping i64 operands, explicit pointer
identities and induction-equality selects for initialization or tails. Its
instruction observer must account for actual calls; unknown effects and control
flow refuse projection. Global addresses require explicit original bindings.
This reader supplies control-flow representation only: it does not establish
instruction semantics, completion, buffer ownership or dense correctness.
`canonical_counted_kernel` normalizes induction-variable names when comparing
selected and reimported schedules, preserving call order and operands.

Synthetic native tests establish shared mechanism behavior. A bounded real-target
execution establishes only its tested configuration, arithmetic and inputs.
Neither establishes all-model support, hardware portability or a latency gain.
Report executed commands and requested traffic separately from hardware timing;
functional Spike counters are not accelerator latency.
