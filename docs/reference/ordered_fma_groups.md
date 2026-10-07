---
title: Source closure for guarded ordered floating contractions
kind: reference
status: current
owner: llvmlower
last_verified: 2026-10-06
related:
  - docs/reference/architecture.md
code_refs:
  - src/merlin/llvmlower/ordered_fma_groups.py
  - src/merlin/llvmlower/ordered_fma_rewrite.py
  - merlin/tests/ir/test_ordered_fma_groups.py
  - src/merlin/llvmlower/ordered_fma_group_outline.py
  - merlin/tests/ir/test_ordered_fma_group_outline.py
  - src/merlin/llvmlower/ordered_bf16_group_binding.py
  - merlin/tests/ir/test_ordered_bf16_group_binding.py
  - src/merlin/llvmlower/quantized_consumer_frontier.py
  - merlin/tests/ir/test_quantized_consumer_frontier.py
  - src/merlin/llvmlower/consumer_observed_group_writer.py
  - merlin/tests/ir/test_consumer_observed_group_writer.py
  - src/merlin/llvmlower/private_workspace_pool.py
  - merlin/tests/ir/test_private_writer_workspace.py
  - merlin/tests/ir/test_private_workspace_pool.py
  - src/merlin/llvmlower/closed_group_writer.py
  - merlin/tests/ir/test_closed_group_source_fallback.py
---

# Source closure for guarded ordered floating contractions

`analyze_ordered_fma_groups` analyzes live typed source operations. It selects
canonical positive-zero, increasing-reduction-axis binary32 FMA contractions
of widened BF16 operands through the existing ordered FMA matcher. It does not
create a second IR or change arithmetic, dispatch, scheduling or buffer ownership.

## Exact source frontier

Static pure parallel generics with complete output maps, registered reductions
and tensor views remain in the source DAG. Every intermediate operation, cast,
coefficient and operand order remains part of the obligation. A BF16 result is
an endpoint unless its view chain feeds another selected contraction. This lets
an intermediate probability tensor connect a score contraction to its value
contractions without inventing an earlier rounded endpoint. Shared actual
consumers merge groups; equal shapes alone do not.

Live binary32 outputs and unsupported uses are recorded explicitly. An external
call never gains purity from its symbol name. Unknown scalar effects, unsupported
maps, dynamic or empty result domains, strict floating scopes and nonempty
fastmath permissions refuse closure. Registered reductions remain source nodes:
analysis grants no reassociation or reduction-order change.

## Retained witness and remaining obligations

`validate_group_source` rechecks contraction matching and the retained live
operations, regions, operands, types, result uses, attributes and properties.
Changed coefficients, zero seeds, inserted scalar operations or new float uses
refuse the old witness. The witness also pins the enclosing operation attributes
and properties plus the upward block/region ownership chain, so a later strict
floating environment annotation or region move refuses the earlier analysis. Binding to a complete source file and provider ABI is
still a separate compiler obligation.

`closed_bf16_endpoints` is only a necessary source condition. A provider must
independently prove the entire endpoint's numeric certificate, intermediate
rounding, fallback semantics, floating environment, complete writes, alias and
lifetime contracts. A closed group is not permission to replace a live float
path with an early BF16 conversion. Input inventory includes original output
initializers; an ABI builder must prove which scalar bodies read them before
omitting an argument. No default routing or performance policy is enabled.

## Source-exact preparation

`outline_ordered_fma_group` is an explicit preparation seam for a closed group.
It validates the entire source witness, a shared single-block parent, complete
endpoints and dominance of the call's external inputs and uses before mutation.
It moves the original operations into an ordinary private function and copies
the source function's attributes. Upstream compilation still owns this function;
no accelerator provider or numerical relaxation is selected by partitioning it.

`read_group_inputs` omits only unused output-init block arguments of the proved
pure parallel generics. Reduction seeds remain reads. Exact scalar constant
splats/fills are materialized locally, preserving their attributes and values;
other source uses of the original constant tensor stay live. Omitted pointwise
initializers become local empty tensors. Each original cast, FMA, reduction,
coefficient and operand order remains inside the function. Ordinary upstream
bufferization owns buffers and copies; the utility infers no physical no-alias,
full-writer external ABI or exception flag permissions.

The preparation returns its actual call, function, input values, original moved
operations and omitted/localized initializer inventory for a provider to bind.
A guarded external implementation still requires an independent complete
endpoint certificate, fallback, ownership and ABI proof plus full numeric
qualification. Default pipeline behavior is unchanged.

## Normal build binding and coverage

`SourceExactGroupPreparation` can be supplied explicitly as the ordinary
`DeviceRouting.prepared_transform` callback. It derives eligible BF16 groups
from current source arithmetic, types and closure, then emits actual source
calls and retained ordinary function bodies. Ordinals and digests distinguish
source instances; they never choose a numerical strategy. Equal-shaped groups
remain separately bound while a target provider may independently share
identical physical integer kernels.

The preparation receipt pins the original prepared file, selected file, original
operation ordinals, complete typed contraction and endpoint contracts, actual
call count, input ABI and retained function digest. `verify_group_call_coverage`
rechecks the record digest, actual calls, declarations, signature and function
body. A coefficient change or an extra same-symbol call refuses coverage.
The source-exact control reports `ordinary_source_cpu` and explicitly states
that no numeric certificate or target product writer has been installed.
Provider activation requires additional qualified arithmetic/effect/ownership
proofs and actual compiled call/catalog evidence; this preparation is not an
acceleration or whole-model performance claim.

## Explicit complete-source writers

`install_closed_group_writers` checks an explicitly supplied endpoint writer
contract against the retained source function, call ABI and enclosing numeric
context before mutation. Its complete typed function fingerprint retains every
operand dependency, scalar property, cast, coefficient and dimension; function
names and `prov.*` metadata do not choose an implementation. Equal shapes with
different arithmetic cannot share a writer. Unselected groups keep their source
bodies. Physical writer sharing requires equal complete contract and typed ABI.

The caller supplies mathematical endpoint and effect witnesses, a complete
source fallback, RNE returned-value policy, implementation identity, full writes,
input preservation, borrowed lifetimes and allocation alignment. The binder
checks identities and obligations; it does not prove these external witnesses.
Its fresh writer bridge lets ordinary upstream own the allocation and lifetime,
and passes a writable output to the borrowed ranked C writer. No FP flag or
physical alias permissions are inferred from source closure.

Installation records the source function digest, selected wrapper digest and
exact supplied contract. Native functional screens, target implementation
coverage and hardware timing remain separately qualified evidence. This explicit
API adds no default route and accepts no source-name or shape-only selector.

An explicit `source_fallback_symbol` retains the complete original typed source
function before replacement. Its ordinary C interface is exported for the
borrowed provider to invoke on numerical or runtime refusal. The model entry's
tensor ABI stays unchanged. Original source casts, reductions and floating
attributes remain; source dispatch tags are removed from this helper to prevent
redispatch. Equal fallback symbols share only equal complete source semantics,
typed input/output ABI and enclosing numerical context. All declaration and
generated C-interface collisions refuse before any source replacement.

The normal upstream pipeline still owns this helper's lowering and allocation.
Its buffer-results conversion appends destination descriptor arguments after
the original input descriptors. Actual compiled native tests call the retained
helper through a borrowed C writer and compare every result bit against the
unchanged source implementation. The provider must invoke fallback before
publishing an incomplete result; retained source code alone grants no numerical
certificate, runtime eligibility, target capability or performance claim.

## Integer consumer observations

`analyze_quantized_consumer_frontier` retains the original source use DAG from
explicit floating tensor producers to an integer tensor observation. It preserves
all typed scalar operations, intermediate precision, registered reductions,
indexing maps and tensor coordinates. Static insertions into a fresh carrier
must cover every element exactly once; dynamic, overlapping or incomplete
assemblies refuse. Pure source pointwise bodies and views retain their exact
arithmetic. Unknown calls/effects and enclosing strict floating scopes require
additional contracts and refuse this analysis.

Every dependent value with a use outside the selected consumer remains a live
observation. This includes floating quantization scales, residual values and raw
source/view escapes. `source_uses_closed` reports only whether a raw unquantized
source escape remains; it does not prove any numerical replacement. A provider
must prove all observation values, including every integer word and escaping
scale, through the retained complete source DAG before returning a different
floating producer value. Returning a value within a small floating tolerance is
insufficient for observational equivalence at this integer frontier.

`validate_quantized_consumer_frontier` binds source uses and types, complete
consumer operation/region bodies and properties, exact assembly coordinates,
and enclosing numeric attributes plus the upward block/region ownership chain.
A changed rounding operation, inserted residual use, modified coordinate or
later strict floating annotation invalidates the retained witness. This analysis
creates no alternate IR, mutates no source, installs no default route, and grants
no floating flag, buffer ownership, alias or source fallback permissions.

`quantized_consumer_semantic_sha256` serializes the complete live source DAG
without cloning operations. Local SSA references retain operand order, scalar
block arguments, types, attributes, properties, coordinates and every live
observation. Only provenance metadata is omitted. This permits an explicitly
supplied numeric consumer theorem to bind equal complete source semantics;
shapes or source names alone cannot establish that match. Enclosing numerical
context remains independently pinned by the live validation witness.

### Explicit consumer-observed writer binding

`install_consumer_observed_group_writers` requires live consumer witnesses before
installing a producer whose floating output may differ. Each supplied witness
pins the complete consumer semantic fingerprint, every observation type, the
numerical theorem identity, effect proof and explicit RNE returned-value policy.
Every selected actual source call must be covered exactly once; foreign live
DAGs, residual/view escapes, incomplete observation lists, stale contexts and
unmatched writer proofs refuse before any source body is replaced.

The original integer/scale consumer remains in the caller. Existing writer
validation still requires the complete producer source, physical implementation,
full writes, input preservation, borrowed lifetime and complete source fallback.
The wrapper records each validated consumer/context/observation obligation. The
supplied mathematical theorem remains an external proof obligation; this binder
adds no default routing, numerical theorem, target qualification or cycle claim.

`find_quantized_consumer_frontiers` discovers candidate observations from explicit
producer SSA values and their actual typed dependencies. It does not assume a
producer count, adjacent call order, model identity or source label. Unsupported
paths remain unselected; extra raw/residual uses remain in each returned witness
for the caller to refuse. Discovery grants no numerical certificate or writer
installation. The explicit binder still validates each complete live witness
before replacing any selected source body.

`ConsumerObservedGroupPreparation` composes these proofs with the normal
prepared-source callback. Explicit immutable producer and consumer contracts
match complete semantics and observation types. Every actual source dependency
is discovered before installation; unknown consumers, unmatched mathematical
proofs and residual escapes keep their original source execution. Ambiguous
proof or live coverage refuses before replacement. The source fallback, numeric
theorem, target implementation and performance qualification remain separate
obligations; no normal build changes unless this callback is explicitly supplied.

## Explicit private workspace ownership

`PrivateWorkspaceContract` supplies bounded byte capacity, alignment and the
implementation effect witness. Each borrowed call must initialize before read,
retain no workspace aliases and finish every use before returning. The generated
full-writer wrapper owns a disjoint local rank-one byte allocation and appends its
descriptor only to the borrowed C ABI. Original tensor arguments and returned
result identity remain unchanged. Unused private bytes need not be written.
Normal upstream buffer deallocation owns release; inserting `memref.dealloc`
before one-shot bufferization is unsupported. Actual compiled native tests check
alignment, disjointness, all outputs, repeated calls and release after completion.

`pool_private_writer_workspaces` optionally moves storage ownership to a common
public function block. It proves all uses through direct private helper calls,
retains the public ABI, propagates explicit dynamic rank-one memrefs through
private ABIs and uses the maximum required bytes/alignment per concurrent slot.
Different calls share storage only; each call still initializes its own state.
Unknown symbolic escapes, extra workspace uses, recursion, multiple owners and
nested control flow refuse before pooling mutation. Borrowed C checks capacity
at least its required bytes. Ordinary upstream owns the terminal release.

The bare-metal bump allocator's `free` is a no-op until inference reset. A local
workspace released in the IR therefore still consumes arena capacity for every
call. One explicitly owned pool avoids that repeated allocation without changing
the global allocator. `ConsumerObservedGroupPreparation` accepts the explicit
`reuse_private_workspaces=True` opt-in. Default empty-workspace generation and
existing writer contract hashes stay compatible. Numerical proof, source fallback,
actual target resources and whole performance remain separately qualified.
The combined preparation validates pooling ownership on the original live source
call graph before replacing any producer. Unsupported multiple owners or symbolic
escapes therefore leave the source module unchanged, without cloning a second IR.
