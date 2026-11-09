---
title: "Prepared operand owners"
kind: reference
status: current
owner: core
last_verified: 2026-11-09
related: [prepared_polynomial_batch]
code_refs:
  - src/merlin/llvmlower/prepared_operand_owner.py
  - src/merlin/llvmlower/prepared_operand_effects.py
  - src/merlin/llvmlower/prepared_operand_schedule.py
  - src/merlin/llvmlower/prepared_attention_rhs.py
  - src/merlin/llvmlower/tensor_preparation_identity.py
---

# Prepared operand owners

`PreparedOperandOwner` models an invocation-local owner for a prepared immutable
tensor view. It extends the read-only `TensorPreparationOpportunity` witness
with an explicit representation, disjoint storage layout and ordered consumer
lifetime. It emits no allocation, packing, cache lookup or operation deletion.

The representation binds format, numeric and effect evidence. The normal
compiler/provider binding must independently validate that evidence: initialized
private storage, immutable source and prepared spans, stable rounding environment,
synchronous consumption, no retained pointers and no writes through aliases.
The effect validator raises on unknown or invalid evidence. A digest is evidence
identity, not a proof of those effects. The prototype cannot detect unauthorized
external memory writes; such writes must be ruled out by the binding proof.

One fresh owner is created per invocation epoch. Every consumer presents its
actual source operation, the same epoch, representation and physical allocation.
The source/view/context and complete block schedule are checked against retained
snapshots. Reordering, new escaping uses and changed operations refuse. Storage
sizes and alignment are checked; overlapping source/storage intervals refuse.
Disjoint byte spans derive from the layout. Every synchronous consumer is used
once; the last use or any failed admission invalidates the owner permanently.

Physical addresses are checked only for allocation consistency and disjointness.
They never select a cached value. A later invocation cannot reuse an old epoch,
even when addresses or input values happen to match. The existing source
fallback remains required whenever a producer or consumer refuses.

A production lowering still needs a dominating allocation, preparation before
the first consumer, explicit handle forwarding through the typed ABI, and release
after the final consumer. Complete cost includes allocation, initialization,
packing, all consumers, reconstruction, metadata and final release. The owner state alone grants no performance or provider qualification.

## Explicit physical attention RHS preparation

The explicit `prepare_readonly_rhs` emitter option adds a separate preparation
entry, owner size/alignment queries, and an owner-consuming entry. The ordinary
provider entry keeps its original arguments and does not reuse prepared data.
The option requires the encoded-row and exact probability-point contracts.
Normal placement is installed separately by the typed borrow planner and
installer; the C emitter alone cannot establish source lifetime or effects.

The producer copies source values, computes canonical signed-radix planes,
reconstructed binary32/binary64 rows, scales and numeric equality flags once.
The private layout keeps each head and source view separate, including transposed
right-hand planes. Consumers use the prepared original/reconstructed spans for
bounds, without changing left-hand probability uncertainty or source replay.
Preparation validates finite source values and the existing rounding environment.

The source binder must establish immutable K/V owners across all admitted calls.
Runtime descriptor checks enforce consistency, not proof of content immutability.
No equality of addresses or captured values grants reuse. Unknown writes or
escapes invalidate the compiler owner witness and require ordinary source fallback.
The experiment measures preparation, its extra allocation, and every consumer
inside the complete four-consumer ROI. No hardware-cycle or normal whole-model
qualification follows from that instruction screen.

## Source effects before physical binding

`validate_prepared_source_lifetime` checks the complete typed source functions
and every operation between the first and last borrow. It follows defined
nonrecursive calls, admits tensor-only structured scalar regions, and refuses
opaque calls, buffers, address values and unknown effects. This source check
passed the actual complete attention source grammar; it is not a certificate
for an external C implementation. Normal integration must separately check the
selected generated producer and borrowed consumer, then emit explicit preparation
and lease arguments. Private workspace pooling alone grants no content reuse.

## Explicit normal-call placement

`plan_prepared_borrows` captures the original immutable tensor bundles before
writer replacement. `install_prepared_borrows` then checks the unchanged source
calls and private writer call tree, invokes the mandatory physical validator,
and emits one public-owner allocation, explicit preparation operations, and
constant epoch/consumer arguments. Epochs do not overlap. Address-only epoch
slots occupy the final bytes of the same pool; no global call counter or pointer
lookup selects reusable contents. The public model function type is unchanged.

The actual upstream lowering and native tests exercise preparation, four borrows,
source preservation and terminal release at O0 and O2 across repeated invocations.
These small ownership tests do not qualify the attention provider. Its generated
C bridge must still bind the pool capacity/epoch slots, initialize validity even
on refused preparation, validate borrowed readonly/noescape behavior, and invoke
the retained source helper on failure before the normal full-model gate.

## Current compiler admission

Planning binds the complete original sequential source calls, exact immutable
views and ancestor attributes. Installation checks the retained contexts and
ordinary writer call tree before mutation, reserves all top-level symbol names
and bounds the aligned private allocation in the signed-index domain. A physical
validator must inspect the selected preparation/borrow implementation; a hash or
truthy return is not permission. Explicit shared-view materialization copies only
proved read arguments once per epoch, leaving tensor ownership and release to
upstream bufferization. It is default-off and is not a pointer cache.

The normal upstream O0/O2 fixtures exercise nonuniform strided views, repeated
invocations, ordered leases, input preservation and terminal deallocation.
Portable attention fixtures cover different extents, masks, dirty storage,
nonfinite refusal and stale epochs; the returned carrier is checked through the
original quantized observer. Refusal must call the retained source writer before
publishing output. Normal ABI wrappers and target products belong to the selected
provider. Their complete link, numeric, effect and resource evidence is still
required; these APIs do not enable a target or predict whole-model cycles.

For navigation, start with `plan_prepared_borrows` and
`install_prepared_borrows` in `prepared_operand_schedule.py`. Source effect
closure is checked by `prepared_operand_effects.py`; linear owner state lives
in `prepared_operand_owner.py`. `prepared_attention_rhs.py` implements the
optional owned C representation and explicit invalidation/borrow entrypoints.
The physical validator remains mandatory at the installation boundary.
