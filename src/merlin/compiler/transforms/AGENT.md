# src/merlin/compiler/transforms

Own reusable transformations, their applicability analysis and materialization.
Preserve numerical ordering, effects, source lineage and explicit permissions.
Backend resources, instruction encodings and target-specific policies stay in OOT.

`host.py` composes native scalar transformations implemented by
`host_contraction.py`, `host_reduction.py`, `host_broadcast.py` and
`host_integer_math.py`. Their source literals execute only in the caller's
selected native context. Preserve the exact source bytes and concatenation order:
ordinary lowering runners and separately reviewed host-library releases consume
the same algorithms. Historical compile/LLVM-lowering imports are exact aliases;
new callers use these real implementation owners. Explicit numerical/effect
choices and independently selected host capabilities retain their obligations.

`pointwise.py` and `reduction.py` own portable tile geometry and byte transport
for explicit semantic requests. They preserve source affine maps, seed bytes,
padding and ordered boundaries without choosing arithmetic or target instructions.
The C transport consumers use these owners; semantic_compiler's historical modules
are exact aliases and published records retain their pickle names. Active native
snapshot receipts observe both aliases and actual implementations. Their packing
checks do not qualify a device kernel or authorize floating-point reassociation.

`contraction_k.py` preserves ordered reduction slices; `contraction_mn.py`
partitions independent output tiles; `contraction_mnk.py` composes those plans
and represents counted whole-program tiling. `panels.py` selects bounded streamed
M/N/K panels from explicit capacities and padding requirements. The caller must
qualify numerical continuation and padding; these owners do not infer identities
or target instructions. Legacy semantic_compiler modules are exact aliases;
published exceptions and records retain their pickle names. Native snapshots
observe actual implementations and aliases. Preserve reduction sequence, original
boundaries, tail/head padding and explicit accumulator-carry obligations.

`recomputation.py` owns bounded source recomputation. Its independent full-module
checker remains in `merlin.compiler.transforms.recomputation_check` during verification
ownership migration. Keep the checker and actual implementation sources pinned;
candidate construction is not execution, cost or hardware qualification.

Legacy module imports may alias the canonical module to preserve issued record,
function and monkeypatch identity. Historical SDK inventories remain immutable.

`multioutput.py` owns connected source fanout discovery and checked compilation
with two live outputs. Its independent semantic checker remains in
`merlin.compiler.schedule.multioutput_fusion_check`. Pin the actual algorithm,
compatibility alias and existing checker/source roster; the old module aliases
this owner exactly, including issued-record state. Preserve published record
pickle spellings. Missing composition permission retains individual lowering.

`row_normalization.py` owns exact row-island materialization and ordinary pass
orchestration. `row_normalization_source.py` reads closed source islands and
their affine row projections. Independent replay remains in
`merlin.compiler.transforms.row_normalization_check`; source admission is not its proof.
The historical module aliases the orchestration owner exactly. Its dispatch
bindings (including matcher and emitter) remain patchable there; pure matching
helpers now resolve globals in their defining source-analysis module and are
reexported as the identical functions. Both record classes retain historical
pickle names. Ordinary receipts observe both current owners, the old alias and
the existing checker/pipeline paths; these hashes do not attest arbitrary loaded
bytecode or provide live authority after the observation. Frozen SDK inventories
and historical full-source snapshots remain unchanged.

## Former merlin.transforms

This package owns target-independent, structural xDSL transformations and analyses.
`pipeline_state.py` owns shared state/order/invalidation types. `pipeline.py`
reexports the identical objects and retains the host verification broker.
Keep the state owner independent of host proof/execution orchestration.
Hardware policies, instruction selection and device resource assignments stay in OOT
repositories. Derive facts from the current IR; unknown effects and unsupported layouts
must remain conservative. A structural witness is not a formal proof receipt.
Exact integer row batching preserves the original typed module quantization
annotation without interpreting it or granting any numerical transformation.

`recomputation.py` is an exact compatibility module alias. The active algorithm
and its source custody live in `merlin.compiler.transforms.recomputation`.
`recomputation_check.py` retains the independent full-module checker during the
separate verification ownership migration; do not move its authority into the
candidate transform or grant additional authoring access to make imports pass.

`row_normalization.py` is an exact alias of the shared compiler transform owner.
Actual source-island matching and row projection live in
`compiler/transforms/row_normalization_source.py`; materialization and ordinary
pass orchestration live in `compiler/transforms/row_normalization.py`. The
independent replay checker remains `row_normalization_check.py`. Preserve exact
record identities and retain the actual implementation and alias observations.

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
explicitly unverified planning invocation. Tests belong in `tests/ir/`.

V2 adds guarded upstream integer CSE/DCE and invariant motion, plus bounded
ordered unrolling. Never infer numerical reassociation or totality from an op
name alone. Region/CFG memory arguments remain possible aliases until forwarding
is proved; only function entry arguments are classified as external roots.

Resource allocation uses explicit OOT representations, geometry and completion
constraints. Emitted typed assignments need independent reconstruction checks
and terminal lowering consumption. Structural allocation checks create no Lean
receipt, asynchronous schedule or hardware qualification.
