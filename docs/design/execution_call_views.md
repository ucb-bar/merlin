---
title: "Source views in execution-owned model binding"
kind: design
status: current
owner: core
last_verified: 2026-11-17
related: [execution_owned_results]
code_refs: [src/merlin/frontends/execution_call_views.py, src/merlin/frontends/execution_call_contracts.py, src/merlin/frontends/execution_model_correspondence.py, src/merlin/frontends/derived_source_views.py, src/merlin/frontends/derived_compile_inputs.py, merlin/tests/ir/test_execution_call_views.py]
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
Fully runtime reshape coordinates, rank-reducing/negative source slices,
encoded or width-changing model views, physical/mutating public views and
general source-layout/ownership/effect compositions remain separate work.
