---
title: "Source views in execution-owned model binding"
kind: design
status: current
owner: core
last_verified: 2026-11-17
related: [execution_owned_results]
code_refs: [merlin/tests/ir/test_execution_rank_views.py, src/merlin/frontends/execution_source_coordinates.py, merlin/tests/ir/test_execution_source_coordinates.py, src/merlin/frontends/execution_runtime_views.py, merlin/tests/ir/test_execution_runtime_views.py, src/merlin/frontends/execution_call_views.py, src/merlin/frontends/execution_call_contracts.py, src/merlin/frontends/execution_model_correspondence.py, src/merlin/frontends/derived_source_views.py, src/merlin/frontends/derived_runtime_arguments.py, src/merlin/frontends/derived_compile_inputs.py, merlin/tests/ir/test_execution_call_views.py, merlin/tests/ir/test_execution_source_layouts.py]
---

# Source views in execution-owned model binding

Explicit v4 execution-owned correspondence additionally accepts
`execution_view_binding_profile: t2m.execution-checked-call-views.v1`. It is
required exactly when calls contain dense policy or source-coordinate records.
Legacy v4 readers reject these extra fields. Bindings without views retain their
previous expansion; public tensor, receipt and signature schemas are unchanged.

The consumer independently rebuilds every dense adapter, original kernel body
and surrounding parent operation. Same-rank casts preserve executable dimension
guards. Rank-changing dense views require equal static element counts, exact
element types and unencoded tensors. Kernel dimensions must be actual original
captured scalar attributes. Dynamic result narrowing checks actual dimensions
before the reshape. Runtime scalar values or evaluator arrays never supply
compiler facts. Rank-zero and empty values retain their actual standard
semantics and no element accesses are invented for empty results.

Source views reuse the existing independent parent-SSA coordinate recovery.
Complete canonical transpose/slice paths must recover the declared owning root,
shape, strides and offset; captured scalar fields and the original source
allocation-pointer binding must match. Every parent schema operand remains
accounted for. Logical relationships do not authorize physical public aliases
or mutation; source pointer sharing comes from the retained executable kernel.

The complete reconstructed model must match types, operations, attributes and
locations. Provenance and declared intent explain the source; neither grants
rewrite, effect, lifetime or numeric authority. Standard native bufferization,
deallocation and allocation-budget checks remain independently mandatory.
Compiler staging still excludes evaluator values and result/reference records.

Implementation review reused LLVM revision
`a47bddccec30255619bb8c37fa59700e661d4e66`, TensorOps.cpp's reshape verification
and tensor/ops.mlir's rank-zero and dynamic reshape cases. The existing source
coordinate checker remains the owner of transpose/slice/root recovery. There
is no producer analysis dependency in the consumer.

Tests cover explicit profile refusal and independent native transport of i64/f32
words, including nonfinite values, scalar ranks and empty shapes. External
frontend integrations additionally exercise complete learned-state models and
source views feeding content-sized count/scan/scatter. These are separate from
arbitrary source equivalence and accelerator execution/performance requirements.
Runtime dense adapters have the separate v2 contract below. Rank-reducing/negative source slices,
encoded or width-changing model views, physical/mutating public views and
general ownership/effect compositions remain separate work. Source-layout binding
has the separately checked extension below.

## Checked runtime dense views

`dense-row-major-runtime-v1` explicitly selects execution view profile v2.
Kernel input extents come from their actual checked runtime scalar operands;
multiple dynamic input axes and changes of same-rank geometry retain a real
collapse/expand. Compatible dynamic types alone cannot prove equal geometry.
Kernel output extents remain actual computed tensor dimensions. A changed result
rank may infer one unknown axis only when its other static axes have nonzero
product. Multiple unknown result axes or a zero static factor require additional
original coordinate information; no annotation or sample invents that information.

Every dimension is nonnegative and every nonempty product uses standard
`arith.mului_extended` on `index`, checking both the high word and the signed
range before reshaping. Empty domains select unit factors before multiplication,
so large unused extents cannot introduce overflow or element accesses. Actual
source and destination counts must match, including divisibility of inferred
axes. Scalar, empty and dynamic views remain ordinary tensor operations, without
changed arithmetic order, source tiling or physical alias permission.

The independent consumer reconstructs the complete guards, dimension operands,
collapse/expand and source locations. Missing/downgraded profiles, removed checks
or substituted dimensions refuse even after outer hashes are repaired. V1 static
adapters retain their original behavior. Versioned source fingerprints also bind
`execution_runtime_views.py`; actual native and applicable target execution remain
separate gates.

## Runtime source coordinates

Per-call `source_view_policy: runtime-logical-coordinates-v1` selects execution
view profile v3. The independent reader derives logical dense-root strides,
transpose permutations and rank-preserving positive slice offsets from actual
parent SSA. Checked dimensions, extended products and nonnegative additions
retain index representability. Every resulting shape, stride and offset must
equal its actual checked kernel scalar operand and the original executable
allocation-root pointer binding. The root ABI keeps independently dynamic axes;
a flattened count never supplies their geometry.

Rank-reducing extraction additionally projects only statically unit axes.
The reader reconstructs TensorOps.cpp's reverse axis matching from actual mixed
sizes and result types. `retained_axes` appears in the corresponding step record
only when rank changes; original rank-preserving records keep their meaning.
Full source-coordinate bounds and offset contributions remain checked before
projecting shape and strides. Retained unit axes keep their original logical
strides; they cannot be confused with the selected axis. Scalar results retain
the complete owning-root offset with an empty axis list.

This is an additive form within execution view profile v3 and coordinate schema
v1. Consumers lacking rank reduction reject it. It grants no physical public
alias, mutation or arbitrary reshape permission. A kernel's one-element ranked
result can separately use the existing checked dense adapter to preserve the
original scalar model result; exact element counts remain executable.

Empty slice refinement accepts only a standard `scf.if` with exactly an
extraction/yield branch and an empty/yield branch, equal result types, equal
dynamic sizes and dominating source coordinates. The branch condition must
equal the conjunction of actual positive sizes. This avoids executing clipped
boundary extractions; arbitrary branches and effects cannot claim a view.
The original conditional and every surrounding operation remain in the model.

Records use `t2m.runtime-source-coordinates.v1`; cloned kernels retain
`t2m.runtime-source-tensor-views.v1`. Both describe logical value coordinates,
without public physical alias or mutation permission. Complete SSA/type/location
comparison, source guards, bounds, initialization, ownership and lifetime checks
remain required. Unknown/downgraded policies or forged records refuse; old v4
readers reject the explicit fields. Public model types, receipt v2 and signature
v6 retain their existing schemas. The selected consumer fingerprint additionally
binds `execution_source_coordinates.py`.

Independent native tests cover exact wide-i64 words, empty boundary/root shapes,
actual coordinate/condition aborts and impure or falsified-record refusals.
External frontend integration separately qualifies i64/f32 dynamic-matrix models
with three actual Triton view updates and all nine snapshots/results. This does
not establish negative slices, arbitrary runtime strides,
physical/mutating aliases, general effects or accelerator performance.

## Semantic source layouts

V4 additionally reuses `source_layout_binding_profile:
t2m.captured-source-layout.v1`. A packed source map can depend on the original
target and warp/thread/CTA options: these determine source grouping, not where
the consumer must execute the operation. Every selected option must bind to a
distinct exact scalar attribute of the original typed model call. The serialized
`source_layout_arguments` and `source_layout_bindings` must agree completely.

The consumer reuses its existing `checked_source_layout_call` proof and then
checks the full original parent/kernel/adapter SSA and locations. The profile
is present exactly when a retained kernel has a source-layout specialization.
Unknown profiles, incomplete selections, unrelated arguments and claims on
unspecialized kernels refuse. Older v4 readers reject the additional call fields.
This extends binding while keeping public model types, receipt v2 and signature
v6 unchanged. The executable grouping, effects, numeric policies, allocation and
lifetime checks retain their existing independent obligations. Source geometry
does not authorize consumer placement, physical aliases or GPU equivalence.
