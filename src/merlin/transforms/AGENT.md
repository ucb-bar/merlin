# AGENT.md — src/merlin/transforms

This package owns target-independent, structural xDSL transformations and analyses.
`pipeline_state.py` owns shared state/order/invalidation types. `pipeline.py`
reexports the identical objects and retains the host verification broker.
Keep the state owner independent of host proof/execution orchestration.
Hardware policies, instruction selection and device resource assignments stay in OOT
repositories. Derive facts from the current IR; unknown effects and unsupported layouts
must remain conservative. A structural witness is not a formal proof receipt.
Exact integer row batching preserves the original typed module quantization
annotation without interpreting it or granting any numerical transformation.

`dense_reduction_layout.rewrite_module` preserves trailing-channel storage for
sole-use single-column signed integer contractions. It exchanges the operands
and parallel axes while retaining the original reduction sequence and external
result shape. Invoke before contraction outlining; opaque effects, other users
or incompatible index orders retain the source. The ordinary `merlin-opt`
catalog exposes `merlin-column-reduction-layout`. No target schedule is supplied.
The same consumer can demand an existing channel-last layout through pure
pointwise chains and projected channel broadcasts. Exact affine maps and the
original scalar bodies remain; arbitrary broadcast layouts are not inferred.
Its source-orientation request is consumed by group statement and logical host
binding. Immutable left operands retain their manifest ownership and preparation;
the request still requires the ordinary checked matrix maps and readout contract.
Scalar per-tensor quantization can carry the layout without expanding its scalar
arithmetic; axis-dependent quantization remains a boundary. Original rounding,
saturation and output quantization stay intact. `HostGraphPolicy` exposes the
opt-in `column_reduction_layout` choice under existing layout/order permissions.

`pipeline_policy.py` selects bounded semantic paths through `PipelineSession`.
Source/context-stamped shape, graph, layout, numerical, capability and resource
facts guide applicability; model/capsule identities and private scores are not
predicate facets. Registry entries expose actual `PassSpec` contracts. Explicit
fallback applies only before execution; a failed pass cannot reuse partial IR.
Return the selected current output analysis, not a disconnected planned schedule.
Keep each session at its declared IR level; native lowering adapters produce the
next IR and verify its boundary. Compact counted loops are never expanded merely
to use a finite schedule pass. Existing host verification obligations remain.
`evaluate_variants` explores explicit alternatives on distinct source-equivalent
private IR, yielding each completion separately. It does not rank, grant guards,
invoke simulators or recover failed partial IR. Unknown predicates remain unknown;
the caller supplies independent artifact execution and timing ownership.

Keep compilation space proportional to IR size and rank, rather than tensor elements or
tile count. All covered changes must go through the host verification hook, except an
explicitly unverified planning invocation. Tests belong in `merlin/tests/ir/`.

V2 adds guarded upstream integer CSE/DCE and invariant motion, plus bounded
ordered unrolling. Never infer numerical reassociation or totality from an op
name alone. Region/CFG memory arguments remain possible aliases until forwarding
is proved; only function entry arguments are classified as external roots.

Resource allocation uses explicit OOT representations, geometry and completion
constraints. Emitted typed assignments need independent reconstruction checks
and terminal lowering consumption. Structural allocation checks create no Lean
receipt, asynchronous schedule or hardware qualification.
